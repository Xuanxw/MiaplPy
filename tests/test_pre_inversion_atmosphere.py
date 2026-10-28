import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest

from miaplpy.pre_inversion_atmosphere import (
    PREFIX, DATASET, GammaBackend, check_temporal, enabled, network_matrix,
    run_correction, source_dataset,
)


@pytest.fixture
def scene(tmp_path):
    shape = (64, 64)
    height = np.broadcast_to(np.arange(64, dtype='f4')[:, None] * 10, shape).copy()
    phases = np.array([height * .01, height * .02, height * .03], dtype='f4')
    (tmp_path / 'inputs').mkdir()
    stack = tmp_path / 'inputs/ifgramStack.h5'
    with h5py.File(stack, 'w') as f:
        f.attrs.update(REF_X=0, REF_Y=0, LENGTH=64, WIDTH=64, FILE_TYPE='ifgramStack', WAVELENGTH=.056)
        f.create_dataset('date', data=np.array([['20200101', '20200102'], ['20200102', '20200103'], ['20200101', '20200103']], dtype='S8'))
        f.create_dataset('dropIfgram', data=np.ones(3, bool))
        f.create_dataset('bperp', data=np.zeros(3))
        f.create_dataset('unwrapPhase', data=phases)
        f.create_dataset('coherence', data=np.ones_like(phases))
        f.create_dataset('connectComponent', data=np.ones_like(phases, dtype='i2'))
    geometry = tmp_path / 'geometry.h5'
    with h5py.File(geometry, 'w') as f:
        f.create_dataset('height', data=height)
    mask = tmp_path / 'control.h5'
    with h5py.File(mask, 'w') as f:
        f.create_dataset('mask', data=np.ones(shape, bool))
    par = tmp_path / 'input.diff_par'
    par.write_text('Gamma DIFF&GEO Processing Parameters\nrange_samp_1: 64\naz_samp_1: 64\noffset_estimation_range_samples: 2\noffset_estimation_azimuth_samples: 2\n')
    cfg = {PREFIX + 'method': 'gamma', PREFIX + 'diffParFile': str(par),
           PREFIX + 'controlMaskFile': str(mask), PREFIX + 'rangeWindow': 32,
           PREFIX + 'azimuthWindow': 32, PREFIX + 'rangeStep': 16,
           PREFIX + 'azimuthStep': 16, PREFIX + 'maxTemporalRms': .001}
    return stack, geometry, cfg, height


class FakeGamma:
    def __init__(self, bad=False):
        self.i, self.bad = 0, bad

    def estimate(self, phase, height, cc, fit, par, directory, cfg):
        directory.mkdir()
        self.i += 1
        scale = [1., 2., 4. if self.bad else 3.][self.i - 1]
        return height * (.01 * scale), {'validPatchFraction': 1.}


def test_preserves_source_and_corrects_zero_phase(scene):
    source, geometry, cfg, _ = scene
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = run_correction(source, geometry, cfg, backend=FakeGamma())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with h5py.File(out) as f:
        assert np.max(np.abs(f[DATASET][:])) < 2e-6
        assert f[DATASET + 'Valid'][:].all()
        with h5py.File(source) as original:
            np.testing.assert_array_equal(f['unwrapPhase'][:], original['unwrapPhase'][:])
        report = json.loads(Path(f.attrs[PREFIX + 'report']).read_text())
        assert report['temporalConsistency']['status'] == 'pass'
    # A real zero observation remains finite in the actual inversion reader.
    from mintpy.objects import ifgramStack
    from miaplpy.dev.ifgram_inversion_L1L2 import read_stack_obs
    stack = ifgramStack(str(out))
    stack.open(print_msg=False)
    obs = read_stack_obs(stack, (0, 0, 64, 64), np.zeros(3), DATASET, print_msg=False)
    assert np.isfinite(obs).all()
    assert abs(obs[0, 0]) < 1e-6


def test_temporal_failure_does_not_publish(scene):
    source, geometry, cfg, _ = scene
    with pytest.raises(RuntimeError, match='temporal consistency'):
        run_correction(source, geometry, cfg, backend=FakeGamma(bad=True))
    assert not source.with_name('ifgramStack_atm.h5').exists()
    reports = list(source.parent.parent.glob('atmosphere_*/report.json'))
    assert json.loads(reports[0].read_text())['temporalConsistency']['status'] == 'fail'


def test_invalid_reference_stops(scene):
    source, geometry, cfg, _ = scene
    with h5py.File(source, 'r+') as f:
        f['connectComponent'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='Invalid reference'):
        run_correction(source, geometry, cfg, backend=FakeGamma())


def test_dropped_pair_is_preserved(scene):
    source, geometry, cfg, _ = scene
    with h5py.File(source, 'r+') as f:
        f['dropIfgram'][2] = False
    out = run_correction(source, geometry, cfg, backend=FakeGamma())
    with h5py.File(out) as f:
        np.testing.assert_array_equal(f[DATASET][2], f['unwrapPhase'][2])
        assert not f['atmospherePhase'][2].any()
        report = json.loads(Path(f.attrs[PREFIX + 'report']).read_text())
        assert report['temporalConsistency']['status'] == 'not_testable_tree_network'


def test_refuses_repeat_correction(scene):
    source, geometry, cfg, _ = scene
    out = run_correction(source, geometry, cfg, backend=FakeGamma())
    with pytest.raises(ValueError, match='already corrected'):
        run_correction(out, geometry, cfg, output_file=source.parent / 'twice.h5', backend=FakeGamma())


def test_network_and_dataset_contracts():
    with pytest.raises(ValueError, match='connected'):
        network_matrix(np.array([['a', 'b'], ['c', 'd']]))
    assert enabled({PREFIX + 'method': 'gamma'})
    assert not enabled({PREFIX + 'method': 'no'})
    assert source_dataset({'mintpy.unwrapError.method': 'bridging+phase_closure'}) == 'unwrapPhase_bridging_phaseClosure'


def test_modelling_failure_preserves_previous_output(scene):
    source, geometry, cfg, _ = scene
    out = source.with_name('ifgramStack_atm.h5')
    out.write_bytes(b'previous accepted result')
    backend = mock.Mock()
    backend.estimate.side_effect = RuntimeError('GAMMA failed')
    with pytest.raises(RuntimeError, match='GAMMA failed'):
        run_correction(source, geometry, cfg, backend=backend)
    assert out.read_bytes() == b'previous accepted result'


def test_gamma_command_contract_and_low_height_rejection(scene):
    source, geometry, cfg, height = scene
    backend = GammaBackend.__new__(GammaBackend)
    backend.binaries = {'atm_mod_2d': 'atm_mod_2d', 'atm_sim_2d': 'atm_sim_2d'}
    from miaplpy.pre_inversion_atmosphere import command
    calls = []
    def fake_command(binary, args, directory):
        calls.append((binary, args))
        for name in ('a0', 'a1', 'sigma', 'sigma_h', 's1'):
            np.ones((2, 2), dtype='>f4').tofile(directory / name)
    with mock.patch('miaplpy.pre_inversion_atmosphere.command', side_effect=fake_command):
        with pytest.raises(RuntimeError, match='height support'):
            run_correction(source, geometry, cfg, backend=backend)
    args = calls[0][1]
    assert args[5] == '0'  # linear model
    assert args[-5:] == [.3, .2, '-', '-', '1']


def test_runfile_routes_corrected_stack_and_disables_backend(tmp_path):
    from miaplpy.miaplpyApp import miaplpyTimeSeriesAnalysis
    analysis = miaplpyTimeSeriesAnalysis.__new__(miaplpyTimeSeriesAnalysis)
    analysis.run_dir = str(tmp_path / 'run_files')
    analysis.out_dir_network = str(tmp_path / 'network')
    analysis.workDir = str(tmp_path)
    template = tmp_path / 'custom.cfg'
    template.write_text('mintpy.troposphericDelay.method = pyaps\n')
    analysis.templateFile_mintpy = str(template)
    analysis.org_custom_template = None
    analysis.text_cmd, analysis.write_job = '', False
    analysis.template = {PREFIX + 'method': 'gamma',
        'miaplpy.timeseries.tempCohType': 'full', 'miaplpy.timeseries.minTempCoh': .5,
        'miaplpy.timeseries.residualNorm': 'L1', 'miaplpy.timeseries.L1smoothingFactor': .001,
        'miaplpy.timeseries.L2weightFunc': 'var', 'miaplpy.timeseries.minNormVelocity': True,
        'miaplpy.timeseries.shadowMask': False, 'miaplpy.timeseries.residualTropHgt': True,
        'mintpy.troposphericDelay.method': 'pyaps'}
    analysis.run_network_inversion('invert_network', None)
    runfile = (tmp_path / 'run_files/run_08_miaplpy_invert_network').read_text()
    assert runfile.index('pre_inversion_atmosphere') < runfile.index('network_inversion.py')
    assert '--dset unwrapPhase_atm' in runfile
    assert 'inputs/ifgramStack_atm.h5' in runfile
    commands = analysis.run_timeseries_correction('timeseries_correction', None)
    assert len(commands) == 1
    assert 'residual_troposphere' not in commands[0]
    effective = (tmp_path / 'run_files/mintpy_after_atmosphere.cfg').read_text()
    assert effective.rstrip().endswith('mintpy.troposphericDelay.method = no')
    assert template.read_text() == 'mintpy.troposphericDelay.method = pyaps\n'


@pytest.mark.skipif(not os.getenv('MIAPLPY_GAMMA_BIN'), reason='requires installed licensed GAMMA')
def test_real_gamma_synthetic_height_signal(scene):
    source, geometry, cfg, _ = scene
    cfg[PREFIX + 'gammaBin'] = os.environ['MIAPLPY_GAMMA_BIN']
    cfg[PREFIX + 'rangeWindow'] = 32
    cfg[PREFIX + 'azimuthWindow'] = 32
    cfg[PREFIX + 'maxTemporalRms'] = .05
    out = run_correction(source, geometry, cfg)
    with h5py.File(out) as f:
        assert np.max(np.abs(f[DATASET][:])) < .05


@pytest.mark.skipif(not os.getenv('MIAPLPY_GAMMA_BIN'), reason='requires installed licensed GAMMA')
def test_real_gamma_preserves_excluded_deformation(scene):
    source, geometry, cfg, height = scene
    bump = np.zeros_like(height)
    bump[24:40, 24:40] = 1.5
    with h5py.File(source, 'r+') as f:
        for i in range(3):
            f['unwrapPhase'][i] += bump * (i + 1)
    with h5py.File(cfg[PREFIX + 'controlMaskFile'], 'r+') as f:
        f['mask'][24:40, 24:40] = False
    cfg[PREFIX + 'gammaBin'] = os.environ['MIAPLPY_GAMMA_BIN']
    cfg[PREFIX + 'maxTemporalRms'] = .05
    out = run_correction(source, geometry, cfg)
    with h5py.File(out) as f:
        for i in range(3):
            np.testing.assert_allclose(f[DATASET][i], bump * (i + 1), atol=.05)


@pytest.mark.parametrize('norm', ['L1', 'L2'])
def test_inversion_wrapper_forwards_dataset_and_solver(norm, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from miaplpy import network_inversion
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(work_dir=str(tmp_path), template_file='template.cfg',
        ifgramStackFile='ifgramStack_atm.h5', residualNorm=norm,
        temp_coh='tempCoh', maskThreshold=.5, L1_alpha=.001,
        minNormVelocity=True, shadow_mask=False, dset=DATASET)
    with mock.patch.object(network_inversion, 'MiaplPyParser') as parser, \
         mock.patch.object(network_inversion.readfile, 'read_template', return_value={'miaplpy.timeseries.tempCohType': 'full'}), \
         mock.patch.object(network_inversion.ut, 'add_attribute'), \
         mock.patch.object(network_inversion.ifgram_inversion_L1L2, 'main') as invert, \
         mock.patch.object(network_inversion.generate_temporal_coherence, 'main'):
        parser.return_value.parse.return_value = args
        network_inversion.main([])
        passed = invert.call_args.args[0]
        assert passed[passed.index('--dset') + 1] == DATASET
        assert passed[passed.index('--norm') + 1] == norm
        assert passed[0] == 'ifgramStack_atm.h5'


def test_rebuild_invalidates_update_timestamp(scene):
    source, geometry, cfg, _ = scene
    first = run_correction(source, geometry, cfg, backend=FakeGamma())
    with h5py.File(first) as f:
        previous = float(f[DATASET].attrs['MODIFICATION_TIME'])
    second = run_correction(source, geometry, cfg, backend=FakeGamma())
    with h5py.File(second) as f:
        assert float(f[DATASET].attrs['MODIFICATION_TIME']) > previous


def test_dimension_mismatch_rejected(scene):
    source, geometry, cfg, _ = scene
    Path(cfg[PREFIX + 'diffParFile']).write_text('range_samp_1: 63\naz_samp_1: 64\n')
    with pytest.raises(ValueError, match='dimensions'):
        run_correction(source, geometry, cfg, backend=FakeGamma())
