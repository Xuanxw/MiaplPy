import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from miaplpy.miaplpyApp import miaplpyTimeSeriesAnalysis
from miaplpy.objects.arg_parser import MiaplPyParser, resolve_miaplpy_source_paths


class TestTemplatePaths(unittest.TestCase):

    def test_reuses_mintpy_template_without_losing_original_path_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_dir = root / 'project' / 'miaplpy'
            template_dir.mkdir(parents=True)
            original_template = template_dir / 'gamma.txt'
            original_template.touch()
            (template_dir / 'custom_smallbaselineApp.cfg').touch()

            analysis = miaplpyTimeSeriesAnalysis(
                str(original_template),
                str(root / 'alternate-work'),
                inps=Namespace(
                    write_job=False,
                    run_flag=False,
                    copy_to_tmp=False,
                ),
            )

            self.assertEqual(analysis.org_custom_template, str(original_template))

    def test_default_work_dir_retains_legacy_nested_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir) / 'project' / 'miaplpy'
            template_dir.mkdir(parents=True)
            template_file = template_dir / 'gamma.txt'
            template_file.touch()

            parsed = MiaplPyParser(
                [str(template_file), '--dostep', 'load_data'],
                script='miaplpy_app',
            ).parse()

            self.assertEqual(parsed.workDir, str(template_dir / 'miaplpy'))

    def test_default_work_dir_keeps_standard_project_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / 'project'
            project_dir.mkdir()
            template_file = project_dir / 'gamma.txt'
            template_file.touch()

            parsed = MiaplPyParser(
                [str(template_file), '--dostep', 'load_data'],
                script='miaplpy_app',
            ).parse()

            self.assertEqual(parsed.workDir, str(project_dir / 'miaplpy'))

    def test_explicit_work_dir_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_dir = root / 'project' / 'miaplpy'
            template_dir.mkdir(parents=True)
            template_file = template_dir / 'gamma.txt'
            template_file.touch()
            explicit_work_dir = root / 'custom-work'

            parsed = MiaplPyParser(
                [
                    str(template_file),
                    '--dir',
                    str(explicit_work_dir),
                    '--dostep',
                    'load_data',
                ],
                script='miaplpy_app',
            ).parse()

            self.assertEqual(parsed.workDir, str(explicit_work_dir))

    def test_resolve_source_paths_relative_to_custom_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_file = root / 'config' / 'gamma.txt'
            (root / 'RSLC_CROP' / '20200101').mkdir(parents=True)
            (root / 'RSLC_CROP' / '20200101' / '20200101.rslc').touch()
            (root / 'CROP_GEOCODE' / 'geometry').mkdir(parents=True)
            (root / 'CROP_GEOCODE' / 'geometry' / 'height.rdc.dem').touch()
            (root / 'CROP_GEOCODE' / 'geometry' / 'lookup.UTM_TO_RDC').touch()
            template = {
                'miaplpy.load.slcFile': '../RSLC_CROP/*/*.rslc',
                'miaplpy.load.demFile': '../CROP_GEOCODE/geometry/*.rdc.dem',
                'miaplpy.load.lookupXFile': '../CROP_GEOCODE/geometry/*.UTM_TO_RDC',
            }

            work_dir = root / 'config' / 'miaplpy'
            resolved = resolve_miaplpy_source_paths(template, template_file, work_dir)

            self.assertEqual(
                resolved['miaplpy.load.slcFile'],
                os.path.abspath(root / 'RSLC_CROP' / '*' / '*.rslc'),
            )
            self.assertEqual(
                resolved['miaplpy.load.demFile'],
                os.path.abspath(root / 'CROP_GEOCODE' / 'geometry' / '*.rdc.dem'),
            )
            self.assertEqual(
                resolved['miaplpy.load.lookupXFile'],
                os.path.abspath(root / 'CROP_GEOCODE' / 'geometry' / '*.UTM_TO_RDC'),
            )

    def test_keeps_absolute_special_and_generated_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            absolute_slc = os.path.abspath(root / 'slc' / '*.rslc')
            template = {
                'miaplpy.load.slcFile': absolute_slc,
                'miaplpy.load.demFile': 'auto',
                'miaplpy.load.waterMaskFile': 'None',
                'miaplpy.load.unwFile': 'inverted/interferograms/*/*.unw',
            }

            resolved = resolve_miaplpy_source_paths(
                template,
                root / 'gamma.txt',
                root / 'miaplpy',
            )

            self.assertEqual(resolved['miaplpy.load.slcFile'], absolute_slc)
            self.assertEqual(resolved['miaplpy.load.demFile'], 'auto')
            self.assertEqual(resolved['miaplpy.load.waterMaskFile'], 'None')
            self.assertEqual(
                resolved['miaplpy.load.unwFile'],
                'inverted/interferograms/*/*.unw',
            )
            self.assertEqual(template['miaplpy.load.demFile'], 'auto')

    def test_prefers_legacy_work_directory_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / 'project'
            slc_dir = project / 'rslc'
            work_dir = project / 'miaplpy'
            slc_dir.mkdir(parents=True)
            work_dir.mkdir()
            (slc_dir / '20200101.rslc').touch()
            template = {'miaplpy.load.slcFile': '../rslc/*.rslc'}

            resolved = resolve_miaplpy_source_paths(
                template,
                project / 'gamma_parameters.txt',
                work_dir,
            )

            self.assertEqual(
                resolved['miaplpy.load.slcFile'],
                str(slc_dir / '*.rslc'),
            )

    def test_auto_paths_are_available_to_the_main_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / 'miaplpy'
            work_dir.mkdir()
            template_file = work_dir / 'miaplpyApp.cfg'
            template_file.write_text(
                'miaplpy.load.autoPath = yes\n'
                'miaplpy.load.processor = gamma\n'
                'miaplpy.load.bperpFile = auto\n'
            )

            analysis = object.__new__(miaplpyTimeSeriesAnalysis)
            analysis.org_custom_template = None
            analysis.workDir = str(work_dir)
            analysis.templateFile = str(template_file)
            analysis.template_mintpy = None

            def expand_auto_path(processor, work_dir, template):
                self.assertEqual(processor, 'gamma')
                expanded = template.copy()
                expanded['miaplpy.load.bperpFile'] = '/data/baselines/*/*.base_perp'
                return expanded

            with mock.patch(
                'miaplpy.miaplpyApp.auto_path.get_auto_path',
                side_effect=expand_auto_path,
            ) as get_auto_path:
                analysis._read_template_miaplpy()

            get_auto_path.assert_called_once()
            self.assertEqual(
                analysis.template['miaplpy.load.bperpFile'],
                '/data/baselines/*/*.base_perp',
            )


if __name__ == '__main__':
    unittest.main()
