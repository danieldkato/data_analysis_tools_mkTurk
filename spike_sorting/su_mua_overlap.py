"""
Spike-time overlap between sorted single units and per-channel MUA threshold crossings.

Kilosort and the 4SD MUA detector process the same raw recording independently, on a
shared clock (both are raw imec sample times with t=0 at recording start, with no sync
correction applied to either). For every (unit, channel) pair this module counts how many
of the unit's spikes coincide with a threshold crossing on that channel, which measures
how much of a unit is visible on each site.

All pairs are computed, not just each unit's peak channel: distant channels supply the
empirical baseline, which lands on the analytic Poisson expectation
`1 - exp(-2*tol*rate)` and so needs no shuffle control.

Channels are identified by SpikeGLX id throughout, matching the MUA filenames, and no
depth-rank conversion is applied. `peak_channel` is a glx id too, so it indexes the matrix
columns directly -- but the SU waveform cache stores a DEPTH RANK under that same name, so
convert before comparing against it.

The result is cached beside the session's single-unit HDF5, under processed_h5.

Counts, not fractions, are cached, so both normalizations stay available:

    n_match / n_su[:, None]     fraction of the unit found on the channel
    n_match / n_mua[None, :]    fraction of the channel contributed by the unit

The two are not interchangeable. The MUA detector has a hard dead time (~0.333 ms) and
emits one event per crossing, so it subsamples heavily; a unit firing faster than its
channel crosses threshold cannot match above `n_mua/n_su`. Always read the first ratio
against that ceiling.

Usage::

    from data_analysis_tools_mkTurk.spike_sorting.su_mua_overlap import (
        cache_overlap_matrix, load_result)
    path = cache_overlap_matrix('West', '20230920')
    result = load_result(path)

CLI::

    python -m data_analysis_tools_mkTurk.spike_sorting.su_mua_overlap --monkey West --date 20230920
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH
from ..npix import get_site_coords, h5_2_ch_meta, map_ks_chans_to_depth_idx
from ..utils_meta import init_dirs, resolve_ks_h5_path
from .staging import find_recording_dir

# Coincidence window. True matches land within one 30 kHz sample (33.3 us) of zero lag, so
# this admits the +/-1-sample spread between Kilosort's template time and the MUA detector's
# threshold-crossing peak, while staying far below the accidental-coincidence regime.
TOL_S: float = 5e-5
SAMP_RATE: int = 30000

# Cached beside the session's single-unit HDF5.
OVERLAP_FILENAME: str = 'su_mua_overlap.npz'


def load_mua_times(mua_dir: Path, glx: int) -> npt.NDArray[np.float64]:
    """
    Load one channel's threshold-crossing times, collapsed to a sorted 1-D array.

    Each detected event stores the timestamps of both its negative and positive peak;
    the sign label picks which one is the spike's extremum, read from either the current
    ch*_sign_label.npy or the older ch*_sls.npy. This mirrors utils_ephys.load_data
    but takes the per-spike argmin/argmax -- load_data indexes with a stray [0], applying
    one spike's column choice to every spike on the channel.

    Parameters
    ----------
    mua_dir : pathlib.Path
        The session's MUA_4SD directory.
    glx : int
        SpikeGLX channel id.

    Returns
    -------
    numpy.ndarray
        Sorted spike times in seconds.
    """
    ts = np.load(mua_dir / 'ch{:0>3d}_ts.npy'.format(glx))
    pks = np.load(mua_dir / 'ch{:0>3d}_pks.npy'.format(glx))
    # Older sessions name the sign labels _sls.npy; get_MUA switched to _sign_label.npy
    # partway through, so both spellings are live on the locker (utils_ephys.load_data
    # falls back the same way).
    sl_path = mua_dir / 'ch{:0>3d}_sign_label.npy'.format(glx)
    if not sl_path.exists():
        sl_path = mua_dir / 'ch{:0>3d}_sls.npy'.format(glx)
    sl = np.load(sl_path)

    neg = np.flatnonzero(sl == 1)
    pos = np.flatnonzero(sl == 0)
    t_neg = ts[neg, np.argmin(pks[neg, 0:2], axis=1)] if neg.size else np.empty(0)
    t_pos = ts[pos, np.argmax(pks[pos, 0:2], axis=1)] if pos.size else np.empty(0)

    return np.sort(np.concatenate([t_neg, t_pos]))


def load_unit_times(ks_dir: Path) -> list[npt.NDArray[np.float64]]:
    """
    Load spike times for every Kilosort unit.

    Uses the exported per-unit files when present, and otherwise rebuilds them from the
    raw sorter output, which is lossless and avoids depending on the export step.

    Parameters
    ----------
    ks_dir : pathlib.Path
        Raw kilosort4 output directory.

    Returns
    -------
    list of numpy.ndarray
        Spike times in seconds, indexed by Kilosort template id.
    """
    n_units = np.load(ks_dir / 'templates.npy', mmap_mode='r').shape[0]
    perunit_dir = ks_dir / 'spike_times_perunit'

    if perunit_dir.is_dir():
        return [np.load(perunit_dir / 'clu_{:0>3d}_st.npy'.format(u)) for u in range(n_units)]

    spike_times = np.squeeze(np.load(ks_dir / 'spike_times.npy')) / SAMP_RATE
    spike_clusters = np.squeeze(np.load(ks_dir / 'spike_clusters.npy'))
    return [spike_times[spike_clusters == u] for u in range(n_units)]


def peak_channels(ks_dir: Path, h5path: str | Path) -> npt.NDArray[np.int64]:
    """
    SpikeGLX channel id of each unit's largest-amplitude template channel.

    Kept for comparison against the overlap matrix's own argmax; the sorter's choice and
    the measured best-matching channel do not always agree.

    Parameters
    ----------
    ks_dir : pathlib.Path
        Raw kilosort4 output directory.
    h5path : str or pathlib.Path
        Session HDF5, read for the probe geometry that locates each Kilosort channel.

    Returns
    -------
    numpy.ndarray
        Glx channel id per unit, indexed by Kilosort template id. Note this is the glx id,
        matching the matrix's channel axis -- NOT the depth rank that the SU waveform cache
        stores under the same name. Convert before comparing the two.
    """
    templates = np.load(ks_dir / 'templates.npy')
    channel_positions = np.load(ks_dir / 'channel_positions.npy')

    # channel_map is NOT the glx mapping: on a 2-bank probe it is a plain arange while
    # channel_positions interleaves the banks (y = 0, 3840, 3860, 20, ...), so indexing it
    # puts roughly half the units in the wrong bank, mm away from their real site. Only a
    # channel's physical y locates it, which is what map_ks_chans_to_depth_idx matches on.
    ks_to_depth = map_ks_chans_to_depth_idx(h5path, channel_positions[:, 1])
    glx_by_depth = (get_site_coords(*h5_2_ch_meta(h5path))
                    .sort_values('ch_idx_depth')['ch_idx_glx'].to_numpy())

    peak_ks = np.argmax(np.ptp(templates, axis=1), axis=1)
    return glx_by_depth[ks_to_depth[peak_ks]]


def count_matches(st: npt.NDArray[np.float64], mua_ts: npt.NDArray[np.float64]) -> int:
    """
    Count spikes in `st` whose nearest neighbour in `mua_ts` falls within TOL_S.

    Parameters
    ----------
    st : numpy.ndarray
        Spike times in seconds.
    mua_ts : numpy.ndarray
        Sorted threshold-crossing times in seconds.

    Returns
    -------
    int
        Number of coincident spikes.
    """
    if st.size == 0 or mua_ts.size == 0:
        return 0

    idx = np.searchsorted(mua_ts, st)
    best = np.full(st.shape, np.inf)
    for off in (-1, 0):
        neighbour = mua_ts[np.clip(idx + off, 0, mua_ts.size - 1)]
        best = np.minimum(best, np.abs(neighbour - st))

    return int(np.count_nonzero(best <= TOL_S))


# Set once per worker by _init_worker, so the spike trains are not re-pickled per task.
_UNIT_ST: list[npt.NDArray[np.float64]] = []
_MUA_DIR: Path = Path()


def _init_worker(unit_st: list[npt.NDArray[np.float64]], mua_dir: Path) -> None:
    """Seed each worker once with the unit spike trains, so they are not re-pickled per task."""
    global _UNIT_ST, _MUA_DIR
    _UNIT_ST = unit_st
    _MUA_DIR = mua_dir


def _match_channel(glx: int) -> tuple[int, npt.NDArray[np.int32], int, float]:
    """Match one channel against every unit. Returns (glx, counts, n_crossings, last_time)."""
    mua_ts = load_mua_times(_MUA_DIR, glx)

    counts: npt.NDArray[np.int32] = np.fromiter(
        (count_matches(st, mua_ts) for st in _UNIT_ST),
        dtype=np.int32, count=len(_UNIT_ST))
    return glx, counts, mua_ts.size, float(mua_ts[-1]) if mua_ts.size else 0.0


def overlap_matrix(monkey: str, date: str, n_jobs: int = -1) -> dict[str, npt.NDArray[Any]]:
    """
    Build the unit x channel overlap matrix for one session.

    Channels are streamed one at a time and swept against every unit held in memory, so
    each MUA file is read once.

    Parameters
    ----------
    monkey : str
        Monkey identifier.
    date : str
        Recording date (YYYYMMDD).
    n_jobs : int, optional
        Worker processes; -1 uses every core. The default is -1.

    Returns
    -------
    dict
        Arrays keyed as described in the module docstring: n_match (n_units, n_channels),
        n_su, n_mua, unit_ids, peak_channel, duration_s, tol_s, channel_ids. Channel axes
        are in SpikeGLX channel id order; unit axes are in Kilosort template id.
    """
    data_path_list, _, _ = init_dirs(BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH)
    if len(data_path_list) != 1:
        raise ValueError('Multiple or no recordings found for {}, {}'.format(monkey, date))
    recording_dir = find_recording_dir(Path(data_path_list[0]))
    ks_dir = recording_dir / 'kilosort4'
    mua_dir = recording_dir / 'MUA_4SD'

    unit_st = load_unit_times(ks_dir)
    glx_ids = np.array(sorted(int(f.name[2:5]) for f in mua_dir.glob('ch[0-9][0-9][0-9]_ts.npy')))
    n_units = len(unit_st)
    n_chans = len(glx_ids)

    n_match: npt.NDArray[np.int32] = np.zeros((n_units, n_chans), dtype=np.int32)
    n_mua: npt.NDArray[np.int32] = np.zeros(n_chans, dtype=np.int32)
    t_max = 0.0

    # One task per channel: each worker reads a single MUA file and sweeps it against every
    # unit, so the units are shipped once via the initializer rather than per task.
    col_of_glx = {int(g): i for i, g in enumerate(glx_ids)}
    workers = os.cpu_count() if n_jobs == -1 else n_jobs
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(unit_st, mua_dir)) as pool:
        for glx, counts, n_cross, last_t in pool.map(_match_channel,
                                                     [int(g) for g in glx_ids],
                                                     chunksize=4):
            col = col_of_glx[glx]
            n_match[:, col] = counts
            n_mua[col] = n_cross
            t_max = max(t_max, last_t)

    n_su = np.array([st.size for st in unit_st], dtype=np.int32)
    t_max = max(t_max, max((float(st[-1]) for st in unit_st if st.size), default=0.0))

    return {
        'n_match': n_match,
        'n_su': n_su,
        'n_mua': n_mua,
        'unit_ids': np.arange(n_units, dtype=np.int32),
        'channel_ids': glx_ids.astype(np.int32),
        'peak_channel': peak_channels(ks_dir, resolve_ks_h5_path(monkey, date)).astype(np.int32),
        'duration_s': np.asarray(t_max, dtype=np.float64),
        'tol_s': np.asarray(TOL_S, dtype=np.float64),
    }


def load_result(path: Path) -> dict[str, npt.NDArray[Any]]:
    """
    Read a cached overlap result.

    Parameters
    ----------
    path : pathlib.Path
        The session's su_mua_overlap.npz.

    Returns
    -------
    dict
        The arrays written by overlap_matrix().

    Raises
    ------
    FileNotFoundError
        If the session has not been computed yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            'Missing {}. Run: python -m data_analysis_tools_mkTurk.spike_sorting.su_mua_overlap '
            '--monkey <monkey> --date <date>'.format(path)
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def compute_overlap_matrix(monkey: str, date: str, n_jobs: int = -1,
                         skip_existing: bool = False) -> Path:
    """
    Compute one session's overlap matrix and cache it beside the single-unit HDF5.

    The importable entry point; the CLI block below wraps it.

    Parameters
    ----------
    monkey : str
        Monkey identifier.
    date : str
        Recording date (YYYYMMDD).
    n_jobs : int, optional
        Worker processes; -1 uses every core. The default is -1.
    skip_existing : bool, optional
        Return the existing result's path without recomputing. The default is False.

    Returns
    -------
    pathlib.Path
        Path of the written (or already present) su_mua_overlap.npz.
    """
    path = resolve_ks_h5_path(monkey, date).with_name(OVERLAP_FILENAME)
    if skip_existing and path.exists():
        print('{} exists, skipping'.format(path))
        return path

    start = time.time()
    result = overlap_matrix(monkey, date, n_jobs=n_jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Type checkers cannot prove a **dict splat carries no 'allow_pickle' key, which
    # np.savez_compressed takes as a keyword-only bool; every key here is an array.
    np.savez_compressed(path, **result)  # type: ignore

    print('{} units x {} channels -> {} ({:.1f} MB) in {:.1f} min'.format(
        result['n_match'].shape[0], result['n_match'].shape[1], path,
        path.stat().st_size / 1e6, (time.time() - start) / 60))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monkey', required=True)
    parser.add_argument('--date', required=True, help='YYYYMMDD')
    parser.add_argument('--n-jobs', type=int, default=-1, help='worker processes (-1 = all cores)')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--override', action='store_true',
                        help='recompute even when the npz exists (wins over --skip-existing)')
    args = parser.parse_args()

    compute_overlap_matrix(args.monkey, args.date, n_jobs=args.n_jobs,
                           skip_existing=args.skip_existing and not args.override)
