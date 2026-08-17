import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from miaplpy.load_ifgram import (
    _IfgramReadAdapter,
    copy_available_geometry_files,
    gamma_subset_yx_to_box,
)
from miaplpy.load_slc_geometry import (
    add_lonlat_gamma,
    sanitize_geometry_radar_lat_lon,
)
from miaplpy.miaplpyApp import miaplpyTimeSeriesAnalysis
from miaplpy.objects.utils import normalize_compression


class TestRuntimeCompatibility(unittest.TestCase):

    def test_complex_interferogram_is_loaded_as_wrapped_phase(self):
        class FakeIfgram:
            def read(self, family, *args, **kwargs):
                return np.array([[1 + 0j, 1j, -1 + 0j]], dtype=np.complex64), {}

        adapter = _IfgramReadAdapter(FakeIfgram())
        data, _ = adapter.read('wrapPhase')

        np.testing.assert_allclose(data, [[0.0, np.pi / 2, np.pi]])
        self.assertEqual(data.dtype, np.float32)

    def test_all_available_geometry_files_are_copied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            destination = root / 'destination'
            source.mkdir()
            (source / 'geometryRadar.h5').write_bytes(b'radar')
            (source / 'geometryGeo.h5').write_bytes(b'geo')

            count = copy_available_geometry_files(source, destination)

            self.assertEqual(count, 2)
            self.assertEqual((destination / 'geometryRadar.h5').read_bytes(), b'radar')
            self.assertEqual((destination / 'geometryGeo.h5').read_bytes(), b'geo')

    def test_compression_aliases_are_h5py_compatible(self):
        for value in [None, False, '', 'auto', 'default', 'no', 'none', 'False']:
            with self.subTest(value=value):
                self.assertIsNone(normalize_compression(value))

        self.assertEqual(normalize_compression('gzip'), 'gzip')
        self.assertEqual(normalize_compression(' LZF '), 'lzf')
        with self.assertRaisesRegex(ValueError, 'Unsupported.*compression'):
            normalize_compression('default-filter')

    def test_gamma_subset_uses_mintpy_xy_order(self):
        self.assertEqual(
            gamma_subset_yx_to_box('0:1270,0:984'),
            (0, 0, 984, 1270),
        )

    def test_generated_ifgram_template_replaces_auto_and_preserves_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis = miaplpyTimeSeriesAnalysis.__new__(miaplpyTimeSeriesAnalysis)
            analysis.workDir = str(root)
            analysis.ifgram_dir = str(root / 'inverted' / 'interferograms_single_reference')
            analysis.template = {}
            analysis.customTemplate = {
                'miaplpy.load.unwFile': 'auto',
                'miaplpy.load.corFile': '/custom/coherence/*.cor',
            }

            analysis._write_generated_ifgram_template()

            runtime = Path(analysis.generated_ifgram_template).read_text(encoding='utf-8')
            self.assertIn(
                'miaplpy.load.unwFile = {}/filt_fine.unw'.format(
                    os.path.join(analysis.ifgram_dir, '*')
                ),
                runtime,
            )
            self.assertIn(
                'miaplpy.load.corFile = /custom/coherence/*.cor',
                runtime,
            )

    def test_parallel_wait_block_propagates_a_background_failure(self):
        script = ['set -e\n', 'pids=()\n', 'false &\n', 'pids+=("$!")\n']
        script.extend(miaplpyTimeSeriesAnalysis._background_wait_commands())

        with tempfile.TemporaryDirectory() as temp_dir:
            run_file = Path(temp_dir) / 'run.sh'
            run_file.write_text(''.join(script), encoding='utf-8')
            result = subprocess.run(['bash', str(run_file)], check=False)

        self.assertNotEqual(result.returncode, 0)

    def test_gamma_lonlat_postprocessing_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            geom_file = Path(temp_dir) / 'geometryGeo.h5'
            with h5py.File(geom_file, 'w') as fobj:
                fobj.attrs.update({
                    'X_FIRST': '114.0',
                    'X_STEP': '0.1',
                    'Y_FIRST': '41.0',
                    'Y_STEP': '-0.1',
                    'WIDTH': '3',
                    'LENGTH': '2',
                })
                fobj.create_dataset(
                    'azimuthCoord', data=np.zeros((2, 3), dtype=np.float32)
                )

            add_lonlat_gamma(geom_file)
            add_lonlat_gamma(geom_file)

            with h5py.File(geom_file, 'r') as fobj:
                self.assertEqual(fobj['longitude'].shape, (2, 3))
                self.assertEqual(fobj['latitude'].shape, (2, 3))

    def test_nonfinite_radar_lookup_coordinates_become_zero_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            geom_file = Path(temp_dir) / 'geometryRadar.h5'
            with h5py.File(geom_file, 'w') as fobj:
                fobj.create_dataset(
                    'latitude',
                    data=np.array([[41.0, np.nan], [np.inf, 40.0]], dtype=np.float32),
                )
                fobj.create_dataset(
                    'longitude',
                    data=np.array([[114.0, 115.0], [116.0, np.nan]], dtype=np.float32),
                )

            count = sanitize_geometry_radar_lat_lon(geom_file)

            self.assertEqual(count, 3)
            with h5py.File(geom_file, 'r') as fobj:
                np.testing.assert_array_equal(
                    fobj['latitude'][:], [[41.0, 0.0], [0.0, 0.0]]
                )
                np.testing.assert_array_equal(
                    fobj['longitude'][:], [[114.0, 0.0], [0.0, 0.0]]
                )


if __name__ == '__main__':
    unittest.main()
