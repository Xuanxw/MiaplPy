#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import glob
import re
import warnings
import numpy as np
import argparse
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
from mintpy.utils import ptime

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='Baselines directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='Output text file')
    parser.add_argument('-r', '--baseline_ratio', dest='baseline_ratio', default=1, type=float,
                        help='Ratio between temporal and perpendicular baseline (default = 1)')
    parser.add_argument('-t', '--temporalBaseline', dest='t_threshold', default=120, type=int,
                        help='Temporal baseline threshold')
    parser.add_argument('-p', '--perpBaseline', dest='p_threshold', default=200, type=int,
                        help='Perpendicular baseline threshold')
    parser.add_argument('-d', '--date_list', dest='date_list', default=None, type=str,
                        help='Text file having existing SLC dates')
    #parser.add_argument('--MinSpanTree', dest='min_span_tree', action='store_true',
    #                      help='Keep minimum spanning tree pairs')

    inps = parser.parse_args(args=iargs)
    return inps


def find_baselines(iargs=None, baselines=None, dates0=None):
    inps = cmd_line_parse(iargs)

    if baselines is None:
        baselines, dates0 = get_baselines_dict(inps.baseline_dir)
    elif dates0 is None:
        dates0 = sorted(baselines)
    with open(inps.date_list, 'r') as f:
        date_list = f.readlines()
        date_list = [dd.split('\n')[0] for dd in date_list]

    min_baselines = min(baselines.values())
    max_baselines = max(baselines.values())

    dates = []
    for date in dates0:
        if date in date_list:
            dates.append(date)

    dates = np.sort(dates)

    days = [(datetime.strptime(date, '%Y%m%d') - datetime.strptime(dates[0], '%Y%m%d')).days for date in dates]

    temp2perp_scale = np.abs((max_baselines - min_baselines) / (np.nanmin(np.array(days)) - np.nanmax(np.array(days))))
    days = [tbase * temp2perp_scale for tbase in days]

    inps.t_threshold *= temp2perp_scale
    multplier = np.sqrt(inps.baseline_ratio)
    days = [x / multplier for x in days]

    pairtr = []
    for i, date in enumerate(dates):
        pairtr.append([days[i], baselines[date] * multplier])

    pairtr = np.array(pairtr)
    tri = Delaunay(pairtr, incremental=False)

    qm = np.zeros([len(dates), len(dates)])

    for trp in pairtr[tri.simplices]:
        x1 = trp[0][0]
        x2 = trp[1][0]
        x3 = trp[2][0]
        b1 = trp[0][1]
        b2 = trp[1][1]
        b3 = trp[2][1]
        if np.abs(x1 - x2) <= inps.t_threshold:
            qm[days.index(x1), days.index(x2)] = np.abs(b1 - b2)
            qm[days.index(x2), days.index(x1)] = np.abs(b1 - b2)
        if np.abs(x2 - x3) <= inps.t_threshold:
            qm[days.index(x2), days.index(x3)] = np.abs(b2 - b3)
            qm[days.index(x3), days.index(x2)] = np.abs(b2 - b3)
        if np.abs(x1 - x3) <= inps.t_threshold:
            qm[days.index(x1), days.index(x3)] = np.abs(b1 - b3)
            qm[days.index(x3), days.index(x1)] = np.abs(b1 - b3)

    qm[qm > inps.p_threshold] = 0

    #if inps.min_span_tree:
    #    X = csr_matrix(qm)
    #    Tcsr = minimum_spanning_tree(X)
    #    A = Tcsr.toarray()
    #else:
    #    for i in range(len(dates)):
    #        if len(np.nonzero(qm[i, :])[0]) <= 1:
    #            qm[i, :] = 0
    #    A = np.triu(qm)

    for i in range(len(dates)):
        if len(np.nonzero(qm[i, :])[0]) <= 1:
            qm[i, :] = 0
    A = np.triu(qm)

    ind1, ind2 = np.where(A > 0)
    ifgdates = ['{}_{}\n'.format(dates[g], dates[h]) for g, h in zip(ind1, ind2)]

    with open(inps.out_file, 'w') as f:
        f.writelines(ifgdates)

    plot_baselines(ind1=ind1, ind2=ind2, dates=dates, baselines=baselines,
                   out_dir=os.path.dirname(inps.out_file))

    return


def _get_gamma_bperp_files(baseline_dir=None, bperp_file=None):
    """Find projected GAMMA pair-baseline files from a pattern or directory."""
    special_values = {None, '', 'auto', 'none', 'no', 'false'}
    bperp_value = str(bperp_file).lower() if bperp_file is not None else None
    files = []

    if bperp_value not in special_values:
        patterns = bperp_file if isinstance(bperp_file, (list, tuple)) else [bperp_file]
        for pattern in patterns:
            pattern = os.path.abspath(os.path.expanduser(str(pattern)))
            if os.path.isdir(pattern):
                files.extend(glob.glob(os.path.join(pattern, '**', '*.base_perp'), recursive=True))
            else:
                files.extend(glob.glob(pattern, recursive=True))
    elif baseline_dir and os.path.isdir(baseline_dir):
        files = glob.glob(os.path.join(baseline_dir, '**', '*.base_perp'), recursive=True)

    files = sorted(set(
        os.path.abspath(fname)
        for fname in files
        if os.path.isfile(fname) and str(fname).endswith('.base_perp')
    ))
    if not files:
        source = bperp_file if bperp_value not in special_values else baseline_dir
        raise FileNotFoundError(
            f'No GAMMA .base_perp files found from: {source}\n'
            'Set miaplpy.load.bperpFile to the pairwise .base_perp pattern. '
            'A .baseline file alone is not a projected perpendicular baseline.'
        )
    return files


def _gamma_date_pair_from_path(fname):
    """Extract and normalize the ordered date pair from a GAMMA file path."""
    date_pair_pattern = re.compile(
        r'(?<!\d)(\d{8}|\d{6})[-_](\d{8}|\d{6})(?!\d)'
    )
    for value in reversed(os.path.normpath(fname).split(os.sep)):
        match = date_pair_pattern.search(value)
        if match:
            return tuple(ptime.yyyymmdd(date) for date in match.groups())
    raise ValueError(f'Cannot find a date pair in GAMMA baseline path: {fname}')


def get_gamma_baselines_dict(bperp_files, date_list=None, residual_tolerance=None):
    """Solve acquisition baselines from pairwise GAMMA perpendicular baselines.

    Each pair file supplies the equation ``B(date2) - B(date1) = bperp``.
    The earliest requested acquisition is fixed at zero because only baseline
    differences are physically relevant to the downstream workflow.
    """
    if isinstance(bperp_files, (str, os.PathLike)):
        bperp_files = glob.glob(os.fspath(bperp_files), recursive=True)
    bperp_files = sorted(set(os.path.abspath(fname) for fname in bperp_files))

    requested_dates = None
    if date_list is not None:
        requested_dates = sorted({ptime.yyyymmdd(str(date)) for date in date_list})
    if requested_dates is not None and len(requested_dates) == 1:
        return {requested_dates[0]: 0.0}

    path_pairs = [(*_gamma_date_pair_from_path(fname), fname) for fname in bperp_files]
    all_dates = sorted({date for date1, date2, _ in path_pairs for date in [date1, date2]})
    if not all_dates:
        raise ValueError('No acquisition dates are available for GAMMA baseline solving.')

    neighbors = {date: set() for date in all_dates}
    for date1, date2, _ in path_pairs:
        neighbors[date1].add(date2)
        neighbors[date2].add(date1)

    components = []
    unseen = set(all_dates)
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            date = pending.pop()
            if date in component:
                continue
            component.add(date)
            pending.extend(neighbors[date] - component)
        unseen -= component
        components.append(sorted(component))

    if requested_dates is None:
        relevant_components = components
    else:
        missing_dates = sorted(set(requested_dates) - set(all_dates))
        if missing_dates:
            raise ValueError(
                'No GAMMA .base_perp paths contain requested acquisitions: '
                + ', '.join(missing_dates)
            )
        relevant_components = [
            component for component in components
            if any(date in component for date in requested_dates)
        ]

    if len(relevant_components) > 1:
        raise ValueError(
            'GAMMA .base_perp network is disconnected; components: '
            + '; '.join(','.join(component) for component in relevant_components)
        )

    dates = relevant_components[0]
    date_set = set(dates)
    grouped_edges = {}
    for date1, date2, fname in path_pairs:
        if date1 not in date_set or date2 not in date_set:
            continue
        base_dict = read_gamma_perp_baseline(fname, atr_dict={})
        value = np.mean([
            float(base_dict['P_BASELINE_TOP_HDR']),
            float(base_dict['P_BASELINE_BOTTOM_HDR']),
        ])
        if not np.isfinite(value):
            raise ValueError(f'Non-finite GAMMA baseline value in: {fname}')

        pair = tuple(sorted([date1, date2]))
        oriented_value = float(value) if (date1, date2) == pair else -float(value)
        grouped_edges.setdefault(pair, []).append((oriented_value, fname))

    edges = []
    for (date1, date2), values_and_files in sorted(grouped_edges.items()):
        values = np.array([item[0] for item in values_and_files], dtype=np.float64)
        files = [item[1] for item in values_and_files]
        if np.ptp(values) > 0.1:
            warnings.warn(
                f'Duplicate GAMMA baselines for {date1}_{date2} differ by '
                f'{np.ptp(values):.4f} m; using their mean. Files: {files}'
            )
        edges.append((date1, date2, float(np.mean(values)), ', '.join(files)))

    date_index = {date: index for index, date in enumerate(dates)}
    design = np.zeros((len(edges), len(dates)), dtype=np.float64)
    observations = np.zeros(len(edges), dtype=np.float64)
    for row, (date1, date2, value, _) in enumerate(edges):
        design[row, date_index[date1]] = -1.0
        design[row, date_index[date2]] = 1.0
        observations[row] = value

    solution, _, rank, _ = np.linalg.lstsq(design[:, 1:], observations, rcond=None)
    if rank != len(dates) - 1:
        raise ValueError(
            f'GAMMA baseline network rank is {rank}; expected {len(dates) - 1}.'
        )

    baselines = np.concatenate(([0.0], solution))
    residuals = design @ baselines - observations
    worst_index = int(np.argmax(np.abs(residuals)))
    worst_residual = abs(residuals[worst_index])
    if residual_tolerance is not None and worst_residual > residual_tolerance:
        date1, date2, observed, fname = edges[worst_index]
        predicted = baselines[date_index[date2]] - baselines[date_index[date1]]
        raise ValueError(
            'Inconsistent GAMMA baseline network. '
            f'Worst pair {date1}_{date2} from {fname}: observed={observed:.4f} m, '
            f'predicted={predicted:.4f} m, residual={residuals[worst_index]:.4f} m.'
        )
    if residual_tolerance is None and worst_residual > 0.1:
        warnings.warn(
            'GAMMA pair baselines do not close exactly; using the least-squares '
            f'solution (maximum residual: {worst_residual:.4f} m).'
        )

    solved = {date: float(baselines[index]) for index, date in enumerate(dates)}
    output_dates = requested_dates if requested_dates is not None else dates
    output_reference = solved[output_dates[0]]
    return {date: solved[date] - output_reference for date in output_dates}


def get_baselines_dict(baseline_dir, processor=None, bperp_file=None, date_list=None):

    if processor == 'gamma' and date_list is not None and len(set(date_list)) == 1:
        date = ptime.yyyymmdd(str(next(iter(date_list))))
        return {date: 0.0}, [date]

    gamma_files = []
    if processor == 'gamma' or (bperp_file and str(bperp_file).lower() not in ['auto', 'none', 'no', 'false']):
        gamma_files = _get_gamma_bperp_files(baseline_dir, bperp_file)
    elif baseline_dir and os.path.isdir(baseline_dir):
        gamma_files = glob.glob(os.path.join(baseline_dir, '*.base_perp'))

    if gamma_files:
        baselines = get_gamma_baselines_dict(gamma_files, date_list=date_list)
        return baselines, sorted(baselines)

    bf = os.listdir(baseline_dir)
    if bf[0].endswith('.base_perp'):
        # gamma
        bf2 = bf
        processor = 'gamma'
    elif not bf[0].endswith('txt'):
        bf2 = ['{}/{}.txt'.format(d, d) for d in bf]
        processor = 'isce'
    else:
        bf2 = bf
        processor = 'isce'

    baselines = {}
    reference = bf[0].split('_')[0]
    baselines[reference] = 0
    dates = [x.split('_')[1].split('.')[0] for x in bf]
    dates.append(reference)

    if processor == 'gamma':
        for d in bf2:
            secondary = d.split('.base_perp')[0].split('_')[-1]
            base_dict = read_gamma_perp_baseline(os.path.join(baseline_dir, d))
            baselines[secondary] = np.mean([float(base_dict['P_BASELINE_TOP_HDR']),
                                            float(base_dict['P_BASELINE_BOTTOM_HDR'])])
    else:
        for d in bf2:
            secondary = d.split('.txt')[0].split('_')[-1]
            with open(os.path.join(baseline_dir, d), 'r') as f:
                lines = f.readlines()
                if len(lines) != 0:
                    if 'Bperp (average):' in lines[1]:
                        baseline = float(lines[1].split('Bperp (average):')[1])
                    else:
                        baseline = float(lines[1].split('PERP_BASELINE_TOP')[1])
                    # baselines.append(baseline)
                    baselines[secondary] = baseline
    return baselines, dates

def read_gamma_perp_baseline(base_perp_file, atr_dict=None):
    """
    load gamma baselines
    """
    if atr_dict is None:
        atr_dict = {}
    bperp_list = []

    f = open(base_perp_file)
    lines = f.readlines()
    f.close()

    header_indices = [
        index for index, line in enumerate(lines)
        if all(name in line.split() for name in ['bpara', 'bperp', 'blen'])
    ]
    if not header_indices:
        raise ValueError(f'Invalid GAMMA .base_perp file (missing table header): {base_perp_file}')

    start_line_idx = header_indices[0] + 1
    for i in range(start_line_idx, len(lines)):
        c = lines[i].strip().split()
        if len(c) == 9:
            bperp_list.append(float(c[7]))

    if not bperp_list:
        raise ValueError(f'Invalid GAMMA .base_perp file (no baseline rows): {base_perp_file}')

    atr_dict['P_BASELINE_TOP_HDR'] = str(bperp_list[0])
    atr_dict['P_BASELINE_BOTTOM_HDR'] = str(bperp_list[-1])
    return atr_dict

def plot_baselines(ind1, ind2, dates=None, baselines=None, out_dir=None, baseline_dir=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')

    if not baseline_dir is None and baselines is None:
        baselines = get_baselines_dict(baseline_dir)[0]

    dates = np.sort(dates)

    ifgdates = ['{}_{}, {}, {}, {}\n'.format(dates[g], dates[h], str(baselines[dates[g]]),
                                             str(baselines[dates[h]]), str(baselines[dates[g]] - baselines[dates[h]]))
                    for g, h in zip(ind1, ind2)]

    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(8, 4))

    for d in ifgdates:
        X = d.split(',')[0].split('_')
        x1 = datetime.strptime(X[0], '%Y%m%d')
        x2 = datetime.strptime(X[1], '%Y%m%d')

        Y = d.split('\n')[0].split(',')[1:3]
        y1 = float(Y[0])
        y2 = float(Y[1])
        plt.plot([x1, x2], [y1, y2], 'ko-', markersize=10)

    plt.xlabel('Time [years]')
    plt.ylabel('Perp Baseline [m]')

    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    fig.savefig(out_dir + '/network.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)

    return

def find_short_pbaseline_pair(baselines, date_list, ministack_size, last_index):

    second_index = np.arange(last_index - ministack_size + 1, last_index)
    diff_bselines = [np.abs(baselines[date_list[last_index - ministack_size - 2]] - baselines[date_list[i]]) for i in second_index]
    min_ind = np.min(diff_bselines)
    pair = (date_list[last_index - ministack_size - 2], date_list[second_index[diff_bselines.index(min_ind)]])
    return pair


def find_mini_stacks(date_list, baseline_dir, month=6, baselines=None):
    pairs = []
    dates = [datetime.strptime(date_str, '%Y%m%d') for date_str in date_list]
    bperp = baselines if baselines is not None else get_baselines_dict(baseline_dir)[0]
    years = np.array([x.year for x in dates])
    u, indices_first = np.unique(years, return_index=True)
    f_ind = indices_first
    l_ind = np.zeros(indices_first.shape, dtype=int)
    l_ind[0:-1] = np.array(f_ind[1::]).astype(int)
    l_ind[-1] = len(dates)
    ref_inds = []
    for i in range(len(f_ind)):
        months = np.array([x.month for x in dates[f_ind[i]:l_ind[i]]])
        u, indices = np.unique(months, return_index=True)
        ind = np.where(u == month)[0]
        if len(ind) == 0:
            ind = int(len(u)//2)
        else:
            ind = ind[0]
        ref_ind = indices[ind] + f_ind[i]
        ref_inds.append(ref_ind)

        for k in range(f_ind[i], l_ind[i]):
            if not ref_ind == k:
                pairs.append((date_list[ref_ind], date_list[k]))
        ministack_size = l_ind[i] - f_ind[i]
        if i > 0:
            pairs.append(find_short_pbaseline_pair(bperp, date_list, ministack_size, l_ind[i]))

    for i in range(len(ref_inds)-1):
        pairs.append((date_list[ref_inds[i]], date_list[ref_inds[i+1]]))
        pairs.append((date_list[l_ind[i]-1], date_list[l_ind[i]]))
    return pairs

def find_one_year_interferograms(date_list):
    dates = np.array([datetime.strptime(date, '%Y%m%d') for date in date_list])

    ifg_ind = []
    for i, date in enumerate(dates):
        range_1 = date + timedelta(days=365) - timedelta(days=5)
        range_2 = date + timedelta(days=365) + timedelta(days=5)
        index = np.where((dates >= range_1) * (dates <= range_2))[0]
        if len(index) >= 1:
            date_diff = list(dates[index] - (date + timedelta(days=365)))
            ind = date_diff.index(np.nanmin(date_diff))
            ind_date = index[ind]
            date2 = date_list[ind_date]
            ifg_ind.append((date_list[i], date2))

    return ifg_ind

if __name__ == '__main__':
    find_baselines()
