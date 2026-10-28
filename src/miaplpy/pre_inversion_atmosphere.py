"""GAMMA local height regression before network inversion (phase in radians).

The source stack is never modified. A separate stack is published atomically
only after model validation and temporal-network consistency checks succeed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt

PREFIX = 'miaplpy.atmosphere.'
DATASET = 'unwrapPhase_atm'


def enabled(template):
    value = str(template.get(PREFIX + 'method', 'no')).strip().lower()
    if value == 'auto':
        value = 'gamma'
    if value not in ('no', 'false', 'none', 'gamma'):
        raise ValueError('miaplpy.atmosphere.method must be gamma or no')
    return value == 'gamma'


def source_dataset(template):
    method = str(template.get('mintpy.unwrapError.method', 'no')).lower().replace(' ', '')
    suffix = {'no': '', 'false': '', 'none': '', '': '',
              'bridging': '_bridging', 'phase_closure': '_phaseClosure',
              'bridging+phase_closure': '_bridging_phaseClosure'}
    if method not in suffix:
        raise ValueError('Unresolved unwrap-error method: ' + method)
    return 'unwrapPhase' + suffix[method]


def read_par(path, key):
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == key + ':':
            return int(float(fields[1]))
    raise ValueError(f'Missing {key} in {path}')


def read_float(path, shape):
    values = np.fromfile(path, dtype='>f4')
    if values.size != int(np.prod(shape)):
        raise ValueError(f'Wrong GAMMA raster size: {path}')
    return values.reshape(shape).astype(np.float32)


def command(binary, args, directory):
    log_name = Path(binary).name
    if log_name == 'fill_gaps':
        log_name += '_' + Path(str(args[0])).name
    with (directory / (log_name + '.log')).open('w') as log:
        log.write(json.dumps([str(binary), *map(str, args)]) + '\n')
        log.flush()
        subprocess.run([str(binary), *map(str, args)], cwd=directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


class GammaBackend:
    """Adapter for the installed GAMMA 2020 atm_mod_2d/atm_sim_2d interface."""
    def __init__(self, directory):
        self.binaries = {}
        for name in ('atm_mod_2d', 'atm_sim_2d', 'fill_gaps'):
            binary = str(Path(directory) / name) if directory else shutil.which(name)
            if name == 'fill_gaps' and directory and not os.access(binary, os.X_OK):
                # GAMMA ships fill_gaps in ISP/bin, the model tools in DIFF/bin.
                sibling = Path(directory).resolve().parent.parent / 'ISP/bin/fill_gaps'
                binary = str(sibling) if sibling.is_file() else shutil.which(name)
            if not binary or not os.access(binary, os.X_OK):
                raise FileNotFoundError(f'{name} is unavailable; set {PREFIX}gammaBin')
            self.binaries[name] = binary
            banner = subprocess.run([binary], capture_output=True, text=True)
            usage = banner.stdout + banner.stderr
            expected = {'atm_mod_2d': '[mfrac]', 'atm_sim_2d': '<a1> <atm_phase>',
                        'fill_gaps': '[ds_data]'}[name]
            if expected not in usage or (name == 'atm_sim_2d' and '<model>' in usage):
                raise RuntimeError(f'Unsupported GAMMA interface: {name}; inspect its usage')

    def estimate(self, phase, height, coherence, fit_mask, diff_par, directory, cfg):
        directory.mkdir()
        par = directory / 'model.diff_par'
        shutil.copyfile(diff_par, par)  # GAMMA rewrites model-grid metadata.
        # GAMMA treats zero phase as missing. A constant shift preserves slopes
        # and cancels when the simulated model is referenced below.
        shift = 1.0
        while np.any((phase[fit_mask] + shift) == 0):
            shift += 1.0
        export = np.where(fit_mask, phase + shift, 0)
        # atm_sim_2d treats a zero DEM height as missing too. A uniform
        # elevation shift changes only the fitted intercept for linear model 0.
        height_shift = 0.0
        while np.any(height[np.isfinite(height)] + height_shift == 0):
            height_shift += 1.0
        gamma_height = np.where(np.isfinite(height), height + height_shift, 0)
        for name, values in [('phase', export), ('height', gamma_height),
                             ('coherence', np.where(fit_mask, coherence, 0))]:
            np.asarray(values, dtype='>f4').tofile(directory / name)
        names = ['a0', 'a1', 'sigma', 'sigma_h', 's1']
        args = ['phase', 'height', 'coherence', par.name, '-', '0', *names,
                cfg['rangeWindow'], cfg['azimuthWindow'], cfg['rangeStep'],
                cfg['azimuthStep'], '7000', '1.350', cfg['minCoherence'],
                cfg['minValidFraction'], '-', '-', '1']
        command(self.binaries['atm_mod_2d'], args, directory)
        shape = (read_par(par, 'offset_estimation_azimuth_samples'),
                 read_par(par, 'offset_estimation_range_samples'))
        if min(shape) < 1:
            raise RuntimeError('GAMMA produced an empty model grid')
        grids = {name: read_float(directory / name, shape) for name in names}
        good = (np.isfinite(grids['a0']) & np.isfinite(grids['a1']) &
                np.isfinite(grids['sigma']) & (grids['sigma'] >= 0) &
                np.isfinite(grids['sigma_h']) &
                (grids['sigma_h'] >= cfg['minHeightStd']))
        if not good.any() or good.mean() < cfg['minValidFraction']:
            raise RuntimeError('Insufficient valid atmospheric model patches / height support')
        # Distance is a support check only; interpolation is delegated to GAMMA.
        distance = distance_transform_edt(~good)
        if distance.max() > cfg['maxFillDistance']:
            raise RuntimeError('Atmospheric model gaps exceed maxFillDistance')
        for name in ('a0', 'a1'):
            # Invalid windows use GAMMA's zero-gap convention. Preserve the raw
            # regression products and each invocation's input/output and log.
            masked = np.where(good, grids[name], 0)
            np.asarray(masked, dtype='>f4').tofile(directory / (name + '_masked'))
            command(self.binaries['fill_gaps'],
                    [name + '_masked', shape[1], name + '_filled',
                     '0', '4', '-', '0', '100', '-', '400', '-'], directory)
            filled = read_float(directory / (name + '_filled'), shape)
            if not np.isfinite(filled).all():
                raise RuntimeError(f'fill_gaps produced nonfinite {name} parameters')
        command(self.binaries['atm_sim_2d'],
                [par.name, 'height', 'a0_filled', 'a1_filled', 'atmosphere', '-'], directory)
        return read_float(directory / 'atmosphere', phase.shape), {
            'validPatchFraction': float(good.mean()),
            'maxFillDistance': float(distance.max()), 'phaseShift': shift,
            'heightShift': height_shift, 'gapFillBackend': 'gamma_fill_gaps',
            'gapFillMethod': 4}


def network_matrix(date_pairs):
    if date_pairs.ndim != 2 or date_pairs.shape[1] != 2 or not len(date_pairs):
        raise ValueError('No retained date pairs / malformed date array')
    dates = sorted(set(date_pairs.ravel()))
    index = {date: i for i, date in enumerate(dates)}
    matrix = np.zeros((len(date_pairs), len(dates)))
    for row, (first, second) in enumerate(date_pairs):
        if first == second:
            raise ValueError('Self-pair in interferogram network')
        matrix[row, index[first]] = -1
        matrix[row, index[second]] = 1
    matrix = matrix[:, 1:]
    if np.linalg.matrix_rank(matrix) != len(dates) - 1:
        raise ValueError('Atmospheric correction requires a connected retained network')
    return matrix


def check_temporal(models, matrix, valid, threshold):
    """Check raw unwrapped model consistency, NOT modulo 2pi or after projection."""
    rank = np.linalg.matrix_rank(matrix)
    cycles = matrix.shape[0] - rank
    if cycles == 0:
        return {'status': 'not_testable_tree_network', 'cycles': 0}
    projector = matrix @ np.linalg.pinv(matrix)
    maximum, total, count = 0., 0., 0
    for row in range(0, models.shape[1], 32):
        block = np.asarray(models[:, row:row + 32, :], dtype=np.float64)
        support = np.asarray(valid[:, row:row + 32, :]).all(axis=0)
        values = block[:, support]
        if values.size:
            residual = values - projector @ values
            rms = np.sqrt(np.mean(residual ** 2, axis=0))
            maximum = max(maximum, float(rms.max()))
            total += float(np.sum(residual ** 2))
            count += residual.size
    if not count:
        raise RuntimeError('No common valid pixels for temporal consistency check')
    result = {'status': 'pass' if maximum <= threshold else 'fail',
              'cycles': int(cycles), 'maxRmsRad': maximum,
              'rmsRad': float(np.sqrt(total / count)), 'thresholdRad': threshold}
    return result


def run_correction(stack_file, geometry_file, template, output_file=None, backend=None):
    """Return a corrected stack; overwrite only the dedicated output, never input."""
    stack_file, geometry_file = Path(stack_file).resolve(), Path(geometry_file).resolve()
    output_file = Path(output_file or stack_file.with_name('ifgramStack_atm.h5')).resolve()
    if output_file == stack_file:
        raise ValueError('Atmospheric output must differ from the source stack')
    base = stack_file.parent.parent
    def input_path(key):
        value = template.get(PREFIX + key)
        if value is None or str(value).lower() in ('auto', 'none', 'no', ''):
            raise ValueError(f'Set {PREFIX}{key} explicitly (relative to network work directory)')
        p = Path(os.path.expandvars(os.path.expanduser(str(value))))
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    par, mask_file = input_path('diffParFile'), input_path('controlMaskFile')
    defaults = {'rangeWindow': 512, 'azimuthWindow': 512, 'rangeStep': 64,
                'azimuthStep': 64, 'minCoherence': .3, 'minValidFraction': .2,
                'minHeightStd': 20., 'maxFillDistance': 2., 'maxTemporalRms': .5}
    cfg = {}
    for key, default in defaults.items():
        raw = template.get(PREFIX + key, default)
        raw = default if str(raw).lower() == 'auto' else raw
        cfg[key] = type(default)(raw)
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('minCoherence', 'minValidFraction'):
        if cfg[key] > 1:
            raise ValueError(f'{key} must be <= 1')
    if cfg['rangeStep'] > cfg['rangeWindow'] or cfg['azimuthStep'] > cfg['azimuthWindow']:
        raise ValueError('Model steps must not exceed window sizes')
    src = source_dataset(template)
    gamma_bin = str(template.get(PREFIX + 'gammaBin', ''))
    gamma_bin = '' if gamma_bin.lower() in ('auto', 'none', 'no') else gamma_bin
    backend = backend or GammaBackend(gamma_bin)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Failed work directories remain available for diagnosis; no output is published.
    work = Path(tempfile.mkdtemp(prefix='atmosphere_', dir=base))
    partial = work / 'ifgramStack_atm.partial.h5'
    report = {'source': str(stack_file), 'sourceDataset': src, 'output': str(output_file),
              'config': cfg, 'pairs': [], 'status': 'failed'}
    try:
        with h5py.File(stack_file, 'r') as source, h5py.File(geometry_file, 'r') as geometry:
            if PREFIX + 'sourceDataset' in source.attrs:
                raise ValueError('Refusing to correct an already corrected stack')
            shape = source[src].shape
            if len(shape) != 3 or geometry['height'].shape != shape[1:]:
                raise ValueError('Phase and height grids must match')
            if (read_par(par, 'az_samp_1'), read_par(par, 'range_samp_1')) != shape[1:]:
                raise ValueError('DIFF_par dimensions do not match phase grid')
            if cfg['rangeWindow'] > shape[2] or cfg['azimuthWindow'] > shape[1]:
                raise ValueError('Model windows exceed image dimensions; configure smaller physical support')
            height = geometry['height'][:]
            with h5py.File(mask_file, 'r') as masks:
                values = masks['mask'][:]
                control = np.isfinite(values) & (values != 0)
            if control.shape != shape[1:] or not control.any():
                raise ValueError('Control mask is empty or has the wrong grid')
            ref = (int(source.attrs['REF_Y']), int(source.attrs['REF_X']))
            if not (0 <= ref[0] < shape[1] and 0 <= ref[1] < shape[2]):
                raise ValueError('Reference pixel is outside phase grid')
            retained = source['dropIfgram'][:].astype(bool) if 'dropIfgram' in source else np.ones(shape[0], bool)
            pairs = source['date'][:].astype(str)
            matrix = network_matrix(pairs[retained])
            if retained.sum() < 1:
                raise ValueError('No retained interferograms')
            shutil.copyfile(stack_file, partial)
            with h5py.File(partial, 'r+') as out:
                corrected = out.create_dataset(DATASET, shape=shape, dtype='f4', chunks=(1, min(128, shape[1]), min(128, shape[2])))
                model = out.create_dataset('atmospherePhase', shape=shape, dtype='f4', chunks=corrected.chunks)
                validity = out.create_dataset(DATASET + 'Valid', shape=shape, dtype='bool', chunks=corrected.chunks)
                for i in range(shape[0]):
                    phase = source[src][i]
                    cc = source['coherence'][i]
                    valid = np.isfinite(phase) & np.isfinite(height) & np.isfinite(cc)
                    if src + 'Valid' in source:
                        valid &= source[src + 'Valid'][i].astype(bool)
                    elif 'connectComponent' in source:
                        valid &= source['connectComponent'][i] > 0
                    else:
                        valid &= phase != 0  # legacy source convention only
                    validity[i] = valid
                    corrected[i] = phase
                    if not retained[i]:
                        continue
                    if not valid[ref]:
                        raise ValueError(f'Invalid reference pixel in pair {pairs[i]}')
                    fit = valid & control & (cc >= cfg['minCoherence'])
                    if fit.mean() < cfg['minValidFraction']:
                        raise RuntimeError(f'Insufficient control samples for pair {pairs[i]}')
                    atmosphere, diagnostics = backend.estimate(phase, height, cc, fit, par, work / f'pair_{i:05d}', cfg)
                    if atmosphere.shape != phase.shape or not np.isfinite(atmosphere[valid]).all():
                        raise RuntimeError('Nonfinite / wrong-size GAMMA atmosphere model')
                    atmosphere = atmosphere - atmosphere[ref]
                    model[i] = np.where(valid, atmosphere, 0)
                    corrected[i] = np.where(valid, phase - atmosphere, phase)
                    report['pairs'].append({'index': i, 'dates': pairs[i].tolist(), **diagnostics})
                # Use retained edges only; stream rows rather than loading the stack.
                class Retained:
                    def __init__(self, ds):
                        self.ds, self.shape = ds, (int(retained.sum()), *shape[1:])
                    def __getitem__(self, key):
                        return self.ds[(np.flatnonzero(retained), *key[1:])]
                check = check_temporal(Retained(model), matrix, Retained(validity), cfg['maxTemporalRms'])
                report['temporalConsistency'] = check
                if check['status'] == 'fail':
                    raise RuntimeError('Atmospheric model temporal consistency exceeds maxTemporalRms; see report')
                if check['cycles'] == 0:
                    print('WARNING: tree network has no cycles; temporal consistency is not testable')
                for key, value in source[src].attrs.items():
                    corrected.attrs[key] = value
                corrected.attrs['MODIFICATION_TIME'] = str(time.time())
                out.attrs[PREFIX + 'sourceDataset'] = src
                out.attrs[PREFIX + 'sourceFile'] = str(stack_file)
                out.attrs[PREFIX + 'config'] = json.dumps(cfg, sort_keys=True)
                out.attrs[PREFIX + 'report'] = str(work / 'report.json')
                out.attrs[PREFIX + 'outputDataset'] = DATASET
        report['status'] = 'success'
        os.replace(partial, output_file)
        return output_file
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        (work / 'report.json').write_text(json.dumps(report, indent=2))
        print('Atmospheric correction report:', work / 'report.json')



def validate_timeseries_lineage(stack_file, timeseries_file):
    """Refuse backend correction on raw, stale or differently sourced time series."""
    with h5py.File(stack_file, 'r') as stack, h5py.File(timeseries_file, 'r') as ts:
        if ts.attrs.get('mintpy.networkInversion.obsDatasetName') != DATASET:
            raise ValueError('Backend requires time series inverted from unwrapPhase_atm')
        if ts.attrs.get('miaplpy.networkInversion.complete') != 'yes':
            raise ValueError('Corrected-phase time-series inversion is incomplete; rerun invert_network')
        for key in ('sourceFile', 'sourceDataset', 'report', 'config', 'outputDataset'):
            name = PREFIX + key
            if name not in stack.attrs or ts.attrs.get(name) != stack.attrs[name]:
                raise ValueError(f'Stale / mismatched atmospheric time-series provenance: {name}')
    print('Verified corrected-phase time-series lineage:', timeseries_file)


def main(iargs=None):
    from mintpy.utils import readfile, utils as ut
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stack')
    parser.add_argument('--geometry')
    parser.add_argument('--template')
    parser.add_argument('--check-timeseries', help='Validate backend input provenance only')
    parser.add_argument('--output')
    args = parser.parse_args(iargs)
    if args.check_timeseries:
        validate_timeseries_lineage(args.stack, args.check_timeseries)
        return
    if not args.geometry or not args.template:
        parser.error('--geometry and --template are required for correction')
    template = ut.check_template_auto_value(readfile.read_template(args.template))
    if not enabled(template):
        raise ValueError('Set miaplpy.atmosphere.method = gamma to run correction')
    run_correction(args.stack, args.geometry, template, args.output)


if __name__ == '__main__':
    main()
