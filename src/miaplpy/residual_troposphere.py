#!/usr/bin/env python3
"""Apply an empirical residual height correction after MintPy/PyAPS.

MintPy exposes PyAPS and height-correlation correction as alternative
``troposphericDelay.method`` values.  MiaplPy uses this small post-processing
step to apply the latter to the final time-series produced by the former,
without overwriting any of the standard MintPy products.
"""

import argparse
import os
from pathlib import Path

import h5py


PROVENANCE_PREFIX = 'miaplpy.residualTropHgt.'


def _absolute_path(path, work_dir):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.is_absolute():
        path = Path(work_dir) / path
    return path.resolve()


def discover_final_timeseries(work_dir):
    """Return the final time-series used to create MintPy ``velocity.h5``."""
    work_dir = Path(work_dir).resolve()
    velocity_file = work_dir / 'velocity.h5'
    if not velocity_file.is_file():
        raise FileNotFoundError(
            f'Cannot locate MintPy final velocity file: {velocity_file}'
        )

    with h5py.File(velocity_file, 'r') as fobj:
        source = fobj.attrs.get('FILE_PATH')
    if isinstance(source, bytes):
        source = source.decode()
    if not source:
        raise ValueError(
            f'{velocity_file} does not record the input time-series in FILE_PATH'
        )

    source = _absolute_path(source, work_dir)
    if not source.is_file():
        raise FileNotFoundError(
            f'MintPy final time-series recorded by {velocity_file} is missing: {source}'
        )
    if source.stem.endswith('_tropHgt'):
        raise ValueError(
            f'MintPy velocity already points to a tropHgt product: {source}'
        )
    return source


def output_paths(source_file, work_dir):
    """Return non-destructive time-series and velocity output paths."""
    source_file = Path(source_file)
    work_dir = Path(work_dir)
    corrected = work_dir / f'{source_file.stem}_tropHgt.h5'
    if corrected.stem.startswith('timeseries'):
        velocity_stem = corrected.stem.replace('timeseries', 'velocity', 1)
    else:
        velocity_stem = f'velocity_{corrected.stem}'
    return corrected, work_dir / f'{velocity_stem}.h5'


def _provenance(params):
    return {
        f'{PROVENANCE_PREFIX}sourceFile': str(params['source_file']),
        f'{PROVENANCE_PREFIX}geometryFile': str(params['geometry_file']),
        f'{PROVENANCE_PREFIX}maskFile': str(params['mask_file']),
        f'{PROVENANCE_PREFIX}polyOrder': str(params['poly_order']),
        f'{PROVENANCE_PREFIX}looks': str(params['looks']),
        f'{PROVENANCE_PREFIX}minCorrelation': str(params['min_correlation']),
    }


def _is_current(outfile, inputs, provenance):
    outfile = Path(outfile)
    if not outfile.is_file():
        return False
    if outfile.stat().st_mtime < max(Path(fname).stat().st_mtime for fname in inputs):
        return False
    try:
        with h5py.File(outfile, 'r') as fobj:
            for key, value in provenance.items():
                actual = fobj.attrs.get(key)
                if isinstance(actual, bytes):
                    actual = actual.decode()
                if str(actual) != str(value):
                    return False
    except (OSError, ValueError):
        return False
    return True


def _write_provenance(outfile, provenance):
    with h5py.File(outfile, 'r+') as fobj:
        for key, value in provenance.items():
            fobj.attrs[key] = value


def run_residual_trop_hgt(
        work_dir,
        geometry_file='inputs/geometryRadar.h5',
        mask_file='maskTempCoh.h5',
        poly_order=1,
        looks=8,
        min_correlation=0.3,
        template_file='smallbaselineApp.cfg',
        geocode=True,
        force=False):
    """Apply residual height correction and regenerate velocity/geocoded files."""
    work_dir = Path(work_dir).resolve()
    source_file = discover_final_timeseries(work_dir)
    geometry_file = _absolute_path(geometry_file, work_dir)
    mask_file = _absolute_path(mask_file, work_dir)
    template_file = _absolute_path(template_file, work_dir)

    for label, fname in (
            ('geometry', geometry_file),
            ('mask', mask_file),
            ('MintPy template', template_file)):
        if not fname.is_file():
            raise FileNotFoundError(f'Residual tropHgt {label} file is missing: {fname}')

    corrected_file, velocity_file = output_paths(source_file, work_dir)
    params = {
        'source_file': source_file,
        'geometry_file': geometry_file,
        'mask_file': mask_file,
        'poly_order': int(poly_order),
        'looks': int(looks),
        'min_correlation': float(min_correlation),
    }
    if params['poly_order'] not in (1, 2, 3):
        raise ValueError('Residual tropHgt polyOrder must be 1, 2, or 3')
    if params['looks'] <= 0:
        raise ValueError('Residual tropHgt looks must be a positive integer')
    if not 0 <= params['min_correlation'] <= 1:
        raise ValueError('Residual tropHgt minCorrelation must be in [0, 1]')

    provenance = _provenance(params)
    input_files = [source_file, geometry_file, mask_file]

    from mintpy.cli import timeseries2velocity, tropo_phase_elevation

    if force or not _is_current(corrected_file, input_files, provenance):
        print('-' * 50)
        print('Apply post-PyAPS residual height-correlated tropospheric correction')
        print(f'input time-series : {source_file}')
        print(f'output time-series: {corrected_file}')
        tropo_phase_elevation.main([
            str(source_file),
            '-g', str(geometry_file),
            '-m', str(mask_file),
            '-l', str(params['looks']),
            '--poly-order', str(params['poly_order']),
            '-t', str(params['min_correlation']),
            '-o', str(corrected_file),
        ])
        _write_provenance(corrected_file, provenance)
    else:
        print(f'Skip residual tropHgt; output is current: {corrected_file}')

    velocity_provenance = {
        **provenance,
        f'{PROVENANCE_PREFIX}sourceFile': str(corrected_file),
    }
    if force or not _is_current(
            velocity_file, [corrected_file, template_file], velocity_provenance):
        print(f'Generate residual tropHgt velocity: {velocity_file}')
        timeseries2velocity.main([
            str(corrected_file),
            '-t', str(template_file),
            '-o', str(velocity_file),
        ])
        _write_provenance(velocity_file, velocity_provenance)
    else:
        print(f'Skip residual tropHgt velocity; output is current: {velocity_file}')

    if geocode:
        from mintpy.cli import geocode as mintpy_geocode

        geo_dir = work_dir / 'geo'
        geo_dir.mkdir(exist_ok=True)
        mintpy_geocode.main([
            str(corrected_file),
            str(velocity_file),
            '-l', str(geometry_file),
            '-t', str(template_file),
            '--outdir', str(geo_dir),
            '--update',
        ])

    return corrected_file, velocity_file


def create_parser():
    parser = argparse.ArgumentParser(
        description='Apply residual height-correlated correction after MintPy/PyAPS.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--geometry', default='inputs/geometryRadar.h5')
    parser.add_argument('--mask', default='maskTempCoh.h5')
    parser.add_argument('--poly-order', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--looks', type=int, default=8)
    parser.add_argument('--min-correlation', type=float, default=0.3)
    parser.add_argument('--template', default='smallbaselineApp.cfg')
    parser.add_argument('--no-geocode', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main(iargs=None):
    inps = create_parser().parse_args(args=iargs)
    return run_residual_trop_hgt(
        work_dir=inps.work_dir,
        geometry_file=inps.geometry,
        mask_file=inps.mask,
        poly_order=inps.poly_order,
        looks=inps.looks,
        min_correlation=inps.min_correlation,
        template_file=inps.template,
        geocode=not inps.no_geocode,
        force=inps.force,
    )


if __name__ == '__main__':
    main()
