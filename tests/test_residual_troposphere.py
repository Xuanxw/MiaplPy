import shutil
import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from miaplpy.miaplpyApp import miaplpyTimeSeriesAnalysis
from miaplpy.residual_troposphere import (
    PROVENANCE_PREFIX,
    discover_final_timeseries,
    output_paths,
    run_residual_trop_hgt,
)


class TestResidualTroposphere(unittest.TestCase):

    def test_output_names_follow_the_mintpy_product_chain(self):
        corrected, velocity = output_paths(
            Path('/project/timeseries_ERA5_ramp_demErr.h5'),
            Path('/project'),
        )

        self.assertEqual(
            corrected, Path('/project/timeseries_ERA5_ramp_demErr_tropHgt.h5')
        )
        self.assertEqual(
            velocity, Path('/project/velocity_ERA5_ramp_demErr_tropHgt.h5')
        )

    def test_discovers_the_timeseries_used_by_final_velocity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source = work_dir / 'timeseries_ERA5_ramp_demErr.h5'
            source.touch()
            with h5py.File(work_dir / 'velocity.h5', 'w') as fobj:
                fobj.attrs['FILE_PATH'] = str(source)

            self.assertEqual(discover_final_timeseries(work_dir), source)

    def test_runs_correction_velocity_and_reuses_current_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source = work_dir / 'timeseries_ERA5_ramp_demErr.h5'
            with h5py.File(source, 'w') as fobj:
                fobj.create_dataset(
                    'timeseries', data=np.zeros((2, 3, 4), dtype=np.float32)
                )
            with h5py.File(work_dir / 'velocity.h5', 'w') as fobj:
                fobj.attrs['FILE_PATH'] = str(source)
            for name in ('geometryRadar.h5', 'maskTempCoh.h5'):
                with h5py.File(work_dir / name, 'w') as fobj:
                    fobj.create_dataset('data', data=np.ones((3, 4)))
            template = work_dir / 'smallbaselineApp.cfg'
            template.write_text('mintpy.timeFunc.polynomial = 1\n', encoding='utf-8')

            def fake_tropo(args):
                shutil.copy2(args[0], args[args.index('-o') + 1])

            def fake_velocity(args):
                outfile = args[args.index('-o') + 1]
                with h5py.File(outfile, 'w') as fobj:
                    fobj.create_dataset('velocity', data=np.zeros((3, 4)))
                    fobj.attrs['FILE_PATH'] = args[0]

            with mock.patch(
                    'mintpy.cli.tropo_phase_elevation.main',
                    side_effect=fake_tropo) as tropo_main, mock.patch(
                    'mintpy.cli.timeseries2velocity.main',
                    side_effect=fake_velocity) as velocity_main:
                corrected, velocity = run_residual_trop_hgt(
                    work_dir,
                    geometry_file='geometryRadar.h5',
                    mask_file='maskTempCoh.h5',
                    template_file=template,
                    geocode=False,
                )
                run_residual_trop_hgt(
                    work_dir,
                    geometry_file='geometryRadar.h5',
                    mask_file='maskTempCoh.h5',
                    template_file=template,
                    geocode=False,
                )

            self.assertEqual(tropo_main.call_count, 1)
            self.assertEqual(velocity_main.call_count, 1)
            tropo_args = tropo_main.call_args.args[0]
            self.assertEqual(tropo_args[tropo_args.index('-t') + 1], '0.3')
            velocity_args = velocity_main.call_args.args[0]
            self.assertEqual(velocity_args[velocity_args.index('-t') + 1], str(template))
            with h5py.File(corrected, 'r') as fobj:
                self.assertEqual(
                    fobj.attrs[f'{PROVENANCE_PREFIX}minCorrelation'], '0.3'
                )
            self.assertTrue(velocity.is_file())

    def _analysis_for_runfile(self, root, method='auto'):
        analysis = miaplpyTimeSeriesAnalysis.__new__(miaplpyTimeSeriesAnalysis)
        analysis.run_dir = str(root / 'run_files')
        analysis.out_dir_network = str(root / 'network_single_reference')
        analysis.templateFile_mintpy = str(root / 'custom_smallbaselineApp.cfg')
        analysis.org_custom_template = None
        analysis.text_cmd = ''
        analysis.write_job = False
        analysis.template = {
            'mintpy.troposphericDelay.method': method,
            'miaplpy.timeseries.residualTropHgt': True,
            'miaplpy.timeseries.residualTropHgt.polyOrder': 1,
            'miaplpy.timeseries.residualTropHgt.looks': 8,
            'miaplpy.timeseries.residualTropHgt.minCorrelation': 0.3,
            'miaplpy.timeseries.residualTropHgt.maskFile': 'maskTempCoh.h5',
        }
        return analysis

    def test_runfile_appends_residual_correction_after_mintpy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis = self._analysis_for_runfile(root)

            commands = analysis.run_timeseries_correction(
                'timeseries_correction', None
            )

            self.assertEqual(len(commands), 2)
            self.assertIn('smallbaselineApp.py', commands[0])
            self.assertIn('python -m miaplpy.residual_troposphere', commands[1])
            self.assertIn('--min-correlation 0.3', commands[1])
            self.assertLess(
                ''.join(commands).index('smallbaselineApp.py'),
                ''.join(commands).index('residual_troposphere'),
            )

    def test_runfile_does_not_double_apply_non_pyaps_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._analysis_for_runfile(
                Path(temp_dir), method='height_correlation'
            )

            commands = analysis.run_timeseries_correction(
                'timeseries_correction', None
            )

            self.assertEqual(len(commands), 1)
            self.assertNotIn('residual_troposphere', commands[0])

    def test_workflow_run_logs_and_executes_multiple_correction_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._analysis_for_runfile(Path(temp_dir))
            analysis.cwd = os.getcwd()

            with mock.patch('miaplpy.miaplpyApp.subprocess.run') as run_mock:
                analysis.run(steps=['timeseries_correction'])

            run_mock.assert_called_once()
            command = run_mock.call_args.args[0]
            self.assertEqual(command[:4], ['bash', '-e', '-o', 'pipefail'])


if __name__ == '__main__':
    unittest.main()
