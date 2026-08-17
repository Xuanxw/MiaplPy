import tempfile
import unittest
import contextlib
import importlib
import io
import sys
from pathlib import Path
from unittest import mock

from miaplpy.objects.utils import prepare_metadata
from miaplpy.prep_slc_gamma import cmd_line_parse, read_gamma_slc_par


class TestGammaMetadata(unittest.TestCase):

    def test_gamma_date_is_zero_padded_to_yyyymmdd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            par_file = Path(temp_dir) / 'scene.rslc.par'
            par_file.write_text(
                'date: 2025 4 3 13 57 32.5375\n',
                encoding='utf-8',
            )

            metadata = read_gamma_slc_par(par_file, skiprows=0)

            self.assertEqual(metadata['date'], '20250403')

    def test_gamma_date_rejects_invalid_calendar_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            par_file = Path(temp_dir) / 'scene.rslc.par'
            par_file.write_text('date: 2025 2 30\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'Invalid GAMMA date'):
                read_gamma_slc_par(par_file, skiprows=0)

    def test_importing_gamma_prep_preserves_redirected_stdout(self):
        from miaplpy import prep_slc_gamma

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            before = sys.stdout
            importlib.reload(prep_slc_gamma)
            self.assertIs(sys.stdout, before)

    def test_cmd_line_parse_honors_supplied_arguments(self):
        parsed = cmd_line_parse(['-s', 'one.rslc', '-r', 'height.rdc.dem'])

        self.assertEqual(parsed['slc_files'], ['one.rslc'])
        self.assertEqual(parsed['radar_geom_files'], ['height.rdc.dem'])

    def test_prepare_metadata_expands_globs_before_calling_gamma_prep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'slc').mkdir()
            (root / 'geometry').mkdir()
            for name in ['20200102.rslc', '20200101.rslc']:
                (root / 'slc' / name).touch()
            (root / 'geometry' / 'height.rdc.dem').touch()
            (root / 'geometry' / 'lookup.UTM_TO_RDC').touch()
            inputs = {
                'processor': 'gamma',
                'miaplpy.load.slcFile': str(root / 'slc' / '*.rslc'),
                'miaplpy.load.demFile': str(root / 'geometry' / '*.rdc.dem'),
                'miaplpy.load.lookupXFile': str(root / 'geometry' / '*.UTM_TO_RDC'),
            }

            with mock.patch('miaplpy.prep_slc_gamma.main') as prep_main:
                prepare_metadata(inputs)

            args = prep_main.call_args.args[0]
            self.assertEqual(
                args,
                [
                    '-s',
                    str(root / 'slc' / '20200101.rslc'),
                    str(root / 'slc' / '20200102.rslc'),
                    '-r',
                    str(root / 'geometry' / 'height.rdc.dem'),
                    '-g',
                    str(root / 'geometry' / 'lookup.UTM_TO_RDC'),
                ],
            )
            self.assertFalse(any('*' in value for value in args))

    def test_prepare_metadata_rejects_an_unmatched_glob(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = {
                'processor': 'gamma',
                'miaplpy.load.slcFile': str(root / 'slc' / '*.rslc'),
                'miaplpy.load.demFile': str(root / 'geometry' / '*.rdc.dem'),
                'miaplpy.load.lookupXFile': str(root / 'geometry' / '*.UTM_TO_RDC'),
            }

            with mock.patch('miaplpy.prep_slc_gamma.main') as prep_main:
                with self.assertRaisesRegex(FileNotFoundError, 'miaplpy.load.slcFile'):
                    prepare_metadata(inputs)

            prep_main.assert_not_called()


if __name__ == '__main__':
    unittest.main()
