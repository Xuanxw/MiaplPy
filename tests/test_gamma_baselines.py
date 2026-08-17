import tempfile
import unittest
import h5py
import numpy as np
from pathlib import Path
from unittest import mock

from miaplpy import load_slc_geometry
from miaplpy.miaplpyApp import miaplpyTimeSeriesAnalysis
from miaplpy.objects.utils import read_initial_info
from miaplpy.find_short_baselines import (
    _get_gamma_bperp_files,
    get_baselines_dict,
    get_gamma_baselines_dict,
)


def write_base_perp(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '  line   range      B_t       B_c        B_n     look angle      '
        'bpara       bperp       blen\n'
        '************************************************************************\n'
        f'0 0 0 0 0 0 0 {value} 0\n'
        f'1 0 0 0 0 0 0 {value} 0\n'
    )
    return path


class TestGammaBaselines(unittest.TestCase):

    def test_existing_slc_stack_supplies_baselines_without_source_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            inputs_dir = work_dir / 'inputs'
            inputs_dir.mkdir()
            stack_file = inputs_dir / 'slcStack.h5'
            with h5py.File(stack_file, 'w') as h5:
                h5.create_dataset('slc', shape=(2, 1, 1), dtype=np.complex64)
                h5.create_dataset('date', data=np.array(['20200101', '20200113'], dtype='S8'))
                h5.create_dataset('bperp', data=np.array([10.0, 15.5], dtype=np.float32))
                h5.attrs['LENGTH'] = '1'
                h5.attrs['WIDTH'] = '1'
                h5.attrs['PROCESSOR'] = 'gamma'
                h5.attrs['REF_DATE'] = '20200101'

            dates, _, _, baselines = read_initial_info(
                str(work_dir),
                str(work_dir / 'missing-template.cfg'),
                return_baselines=True,
            )

            self.assertEqual(dates, ['20200101', '20200113'])
            self.assertEqual(baselines, {'20200101': 0.0, '20200113': 5.5})

    def test_single_acquisition_does_not_require_pair_file(self):
        baselines, dates = get_baselines_dict(
            '/does/not/exist',
            processor='gamma',
            date_list=['20200101'],
        )

        self.assertEqual(baselines, {'20200101': 0.0})
        self.assertEqual(dates, ['20200101'])

    def test_solves_chain_as_acquisition_baselines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = [
                write_base_perp(root / '20260223_20260317.base_perp', 82.2095),
                write_base_perp(root / '20260110_20260201.base_perp', 60.88505),
                write_base_perp(root / '20260201_20260223.base_perp', -19.79045),
            ]

            baselines = get_gamma_baselines_dict(
                files,
                date_list=['20260110', '20260201', '20260223', '20260317'],
            )

            expected = {
                '20260110': 0.0,
                '20260201': 60.88505,
                '20260223': 41.09460,
                '20260317': 123.30410,
            }
            for date, value in expected.items():
                self.assertAlmostEqual(baselines[date], value, places=5)

    def test_accepts_hyphenated_pair_in_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_base_perp(
                Path(temp_dir) / '20200101-20200113' / 'pair.base_perp',
                -12.5,
            )

            baselines = get_gamma_baselines_dict(
                [path],
                date_list=['20200101', '20200113'],
            )

            self.assertEqual(baselines, {'20200101': 0.0, '20200113': -12.5})

    def test_rejects_disconnected_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = [
                write_base_perp(root / '20200101_20200113.base_perp', 1.0),
                write_base_perp(root / '20200201_20200213.base_perp', 2.0),
            ]

            with self.assertRaisesRegex(ValueError, 'disconnected'):
                get_gamma_baselines_dict(
                    files,
                    date_list=['20200101', '20200113', '20200201', '20200213'],
                )

    def test_rejects_inconsistent_cycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = [
                write_base_perp(root / '20200101_20200113.base_perp', 10.0),
                write_base_perp(root / '20200113_20200125.base_perp', 10.0),
                write_base_perp(root / '20200101_20200125.base_perp', 30.0),
            ]

            with self.assertRaisesRegex(ValueError, 'Inconsistent GAMMA baseline'):
                get_gamma_baselines_dict(files, residual_tolerance=0.1)

    def test_subset_dates_may_be_connected_through_an_excluded_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = [
                write_base_perp(root / '20200101_20200113.base_perp', 10.0),
                write_base_perp(root / '20200101_20200125.base_perp', 25.0),
            ]

            baselines = get_gamma_baselines_dict(
                files,
                date_list=['20200113', '20200125'],
            )

            self.assertEqual(baselines, {'20200113': 0.0, '20200125': 15.0})

    def test_finds_nested_pair_files_from_explicit_pattern(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = write_base_perp(
                root / 'interferograms' / '20200101_20200113' / 'pair.base_perp',
                1.0,
            )

            files = _get_gamma_bperp_files(
                bperp_file=str(root / 'interferograms' / '**' / '*.base_perp')
            )

            self.assertEqual(files, [str(expected)])

    def test_date_pair_may_be_in_an_ancestor_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_base_perp(
                Path(temp_dir) / '20200101_20200113' / 'nested' / 'pair.base_perp',
                2.5,
            )

            baselines = get_gamma_baselines_dict([path])

            self.assertEqual(baselines, {'20200101': 0.0, '20200113': 2.5})

    def test_explicit_baseline_model_is_rejected_as_not_projected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / '20200101_20200113.baseline'
            baseline.touch()

            with self.assertRaisesRegex(FileNotFoundError, 'No GAMMA .base_perp'):
                _get_gamma_bperp_files(bperp_file=str(baseline))

    def test_rejects_nonfinite_baseline_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_base_perp(
                Path(temp_dir) / '20200101_20200113.base_perp',
                'nan',
            )

            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                get_gamma_baselines_dict([path])

    def test_gamma_scalar_bperp_is_not_loaded_as_geometry_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dem_file = root / 'height.rdc.dem'
            dem_file.touch()
            bperp_file = write_base_perp(
                root / '20200101_20200113.base_perp',
                1.0,
            )
            inputs = {
                'processor': 'gamma',
                'miaplpy.load.demFile': str(dem_file),
                'miaplpy.load.bperpFile': str(bperp_file),
            }
            metadata = {'LENGTH': '1', 'WIDTH': '1', 'PROCESSOR': 'gamma'}
            mapping = load_slc_geometry.datasetName2templateKey.copy()

            with mock.patch.dict(
                load_slc_geometry.datasetName2templateKey,
                mapping,
                clear=True,
            ), mock.patch(
                'miaplpy.load_slc_geometry.mut.read_attribute',
                return_value=metadata,
            ), mock.patch(
                'miaplpy.objects.geometryStack.read_attribute',
                return_value=metadata,
            ):
                radar_geometry, _ = load_slc_geometry.read_inps_dict2geometry_dict_object(inputs)

            self.assertNotIn('bperp', radar_geometry.datasetDict)

    def test_pair_selection_uses_solved_gamma_baselines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dates = ['20260110', '20260201', '20260223', '20260317']
            write_base_perp(root / '20260110_20260201.base_perp', 60.88505)
            write_base_perp(root / '20260201_20260223.base_perp', -19.79045)
            write_base_perp(root / '20260223_20260317.base_perp', 82.2095)

            analysis = object.__new__(miaplpyTimeSeriesAnalysis)
            analysis.workDir = str(root)
            analysis.out_dir_network = str(root / 'network')
            Path(analysis.out_dir_network).mkdir()
            analysis.date_list = dates
            analysis.date_list_text = str(root / 'date_list.txt')
            analysis.num_images = len(dates)
            analysis.baselines = get_gamma_baselines_dict(
                list(root.glob('*.base_perp')),
                date_list=dates,
            )
            analysis.template = {
                'miaplpy.load.processor': 'gamma',
                'miaplpy.load.baselineDir': str(root / 'unused-baselines'),
                'miaplpy.load.bperpFile': str(root / '*.base_perp'),
                'miaplpy.interferograms.list': None,
                'miaplpy.interferograms.networkType': 'single_reference',
                'miaplpy.interferograms.referenceDate': False,
                'miaplpy.interferograms.oneYear': False,
            }

            with mock.patch('miaplpy.miaplpyApp.fb.plot_baselines') as plot:
                _, pairs = analysis.get_interferogram_pairs()

            self.assertEqual(
                pairs,
                [
                    ('20260223', '20260110'),
                    ('20260223', '20260201'),
                    ('20260223', '20260317'),
                ],
            )
            plotted = plot.call_args.kwargs['baselines']
            self.assertAlmostEqual(plotted['20260317'], 123.30410, places=5)
            self.assertIsNone(plot.call_args.kwargs['baseline_dir'])


if __name__ == '__main__':
    unittest.main()
