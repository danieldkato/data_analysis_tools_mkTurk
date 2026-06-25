import os
import sys
import gc
import glob
import time
import argparse
from pathlib import Path

import numpy as np
from scipy import signal
from natsort import os_sorted

from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import (
    readMeta, ChanGainsIM, ChannelCountsIM, ChannelCountsNI, makeMemMapRaw,
)
try:
    from .utils_meta import get_recording_path
    from .make_engram_path import BASE_DATA_PATH
except ImportError:
    from data_analysis_tools_mkTurk.utils_meta import get_recording_path
    from data_analysis_tools_mkTurk.make_engram_path import BASE_DATA_PATH

try:
    import cupy as cp
except ImportError:
    cp = None

# MUA spike detection parameters
_SPIKE_THRESH_SD = 4
_MIN_DIST_SAMPS = 20
_LEFT_SAMPS = 15
_RIGHT_SAMPS = 30
_HIGH_PASS_HZ = 300
_CHUNK_SEC = 20
_DEFAULT_N_CHANS = 384
_DEFAULT_FS = 30000.0


def _read_meta_info(bin_path: Path, meta: dict) -> tuple[float, float, int]:
    """Return (Fs, file_len_sec, n_chans), patching meta if fileTimeSecs is absent."""
    if 'fileTimeSecs' in meta:
        Fs = float(meta['imSampRate'])
        file_len_sec = float(meta['fileTimeSecs'])
        n_chans = int(meta['nSavedChans']) - 1
    else:
        print('meta file time not saved. recovering from OS file size.')
        n_chans = _DEFAULT_N_CHANS
        Fs = _DEFAULT_FS
        filesize_os = os.path.getsize(bin_path)
        file_len_sec = filesize_os / (Fs * 2 * (n_chans + 1))
        print(f'file size: {filesize_os}')
        print(f'file len (sec): {file_len_sec}')
        meta['fileSizeBytes'] = filesize_os
        meta['nSavedChans'] = 385
        meta['imAiRangeMin'] = '-0.6'
        meta['imAiRangeMax'] = '0.6'
        meta['imMaxInt'] = '512'
        meta['imroTbl'] = (
            '(0,384)' + ''.join(f'({i} 0 1 500 250 0)' for i in range(384))
        )
    return Fs, file_len_sec, n_chans


def _get_dig_channel(meta: dict) -> int:
    """Return the digital sync channel index for imec or NI probes."""
    dwReq = 0
    if meta['typeThis'] == 'imec':
        if 'snsApLfSy' not in meta:
            return 384
        AP, LF, SY = ChannelCountsIM(meta)
        if SY == 0:
            raise RuntimeError("No imec sync channel saved.")
        return AP + LF + dwReq
    else:
        MN, MA, XA, DW = ChannelCountsNI(meta)
        if dwReq > DW - 1:
            raise RuntimeError(f"Requested digital word {dwReq} exceeds max {DW-1}.")
        return MN + MA + XA + dwReq


def _process_chunk(raw_data, meta, first_samp: int, last_samp: int,
                   sos, n_chans: int, Fs: float):
    """Load one chunk, apply gain, CAR, and highpass filter. Returns (data_filt, chunk_start_sec)."""
    chunk_start_sec = first_samp / Fs

    raw_chunk = np.array(raw_data[:n_chans, first_samp:last_samp])

    # Gain correction
    APgain, _ = ChanGainsIM(meta) # type: ignore
    fI2V = float(meta['imAiRangeMax']) / int(meta['imMaxInt'])
    conv_all = fI2V / APgain  # shape (n_chans,)

    if cp is not None:
        arr = cp.array(raw_chunk)
        conv = cp.multiply(cp.array(conv_all), arr.T).T
        conv = cp.subtract(conv.T, cp.mean(conv, axis=1)).T
        car = cp.subtract(conv, cp.mean(conv, axis=0))
        data_filt = signal.sosfiltfilt(sos, cp.asnumpy(car), axis=1)
    else:
        conv = np.multiply(conv_all, raw_chunk.T).T
        conv = np.subtract(conv.T, np.mean(conv, axis=1)).T
        car = np.subtract(conv, np.mean(conv, axis=0))
        data_filt = signal.sosfiltfilt(sos, car, axis=1)

    return data_filt, chunk_start_sec


def _extract_spikes_channel(data_chunk, stds_bych, ch_idx, Fs, chunk_start_sec,
                             spike_thresh_sd, min_dist, left_samps, right_samps):
    """Detect positive and negative threshold crossings and extract waveforms for one channel."""
    sw_len = left_samps + right_samps
    chunk_len = len(data_chunk)
    ch_thresh = stds_bych[ch_idx] * spike_thresh_sd

    pks_neg, _ = signal.find_peaks(-data_chunk, height=ch_thresh, distance=min_dist)
    pks_pos, _ = signal.find_peaks(data_chunk, height=ch_thresh, distance=min_dist)
    pks = np.sort(np.concatenate((pks_neg, pks_pos)))

    if len(pks) > 0:
        pks = pks[np.concatenate(([True], np.diff(pks) > sw_len))]

    spike_wfs = np.empty([len(pks), sw_len])
    spike_pks = np.empty([len(pks), 3])
    spike_ts = np.empty([len(pks), 2])
    spike_sl = np.empty(len(pks))

    for spkidx, pk in enumerate(pks):
        sw_start = max(1, pk - left_samps)
        sw_end = min(pk + right_samps, chunk_len)
        wf_tmp = data_chunk[sw_start:sw_end]

        if pk - left_samps < 1:
            wf_tmp = np.append(np.zeros(-(pk - left_samps) + 1), wf_tmp)
        if pk + right_samps > chunk_len:
            wf_tmp = np.append(wf_tmp, np.zeros(pk + right_samps - chunk_len))

        spike_wfs[spkidx, :] = wf_tmp

        pks_amp = data_chunk[pk]
        pks_ts = pk / Fs + chunk_start_sec

        if pks_amp < 0:
            ind = np.argmax(wf_tmp)
            spike_sl[spkidx] = 1
        else:
            ind = np.argmin(wf_tmp)
            spike_sl[spkidx] = 0

        pks_opps_amp = wf_tmp[ind]
        pks_opps_ts = (ind - left_samps) / Fs + pks_ts

        spike_pks[spkidx, :] = (pks_amp, pks_opps_amp, stds_bych[ch_idx])
        spike_ts[spkidx, :] = (pks_ts, pks_opps_ts)

    return spike_wfs, spike_pks, spike_ts, spike_sl


def _consolidate_tmp_files(MUA_dir: Path, n_chans: int, n_chunks: int,
                            data_path: Path, spike_thresh: float,
                            t_win: float, high_pass_Hz: float) -> None:
    """Read per-chunk tmp .npy files, concatenate, save final files, and delete tmps."""
    for ch in range(n_chans):
        print(f'ch{ch} of {n_chans}')
        results = {}
        for tag, shape_extra in [('wfs', None), ('ts', None), ('pks', None),
                                  ('sds', 'scalar'), ('sls', None)]:
            fname = list(MUA_dir.glob(f'spike_{tag}_ch{ch:03d}_tmp.npy'))
            with open(fname[0], 'rb') as f:
                chunks = []
                for _ in range(n_chunks):
                    try:
                        arr = np.load(f)
                        chunks.append(arr[np.newaxis] if shape_extra == 'scalar' else arr)
                    except Exception:
                        pass
            results[tag] = np.concatenate(chunks)

        np.save(MUA_dir / f'ch{ch:03d}_pks', results['pks'])
        np.save(MUA_dir / f'ch{ch:03d}_ts', results['ts'])
        np.save(MUA_dir / f'ch{ch:03d}_wfs', results['wfs'])
        np.save(MUA_dir / f'ch{ch:03d}_sds', results['sds'])
        np.save(MUA_dir / f'ch{ch:03d}_sls', results['sls'])
        np.savez(MUA_dir / f'ch{ch:03d}_meta',
                 spike_thresh=spike_thresh,
                 recording=data_path, # type: ignore
                 n_chunks_for_processing=n_chunks,
                 chunk_len_sec=t_win,
                 high_pass_Hz=high_pass_Hz)

    for pattern in ['spike_wfs_ch*_tmp.npy', 'spike_ts_ch*_tmp.npy',
                    'spike_pks_ch*_tmp.npy', 'spike_sds_ch*_tmp.npy',
                    'spike_sls_ch*_tmp.npy']:
        for f in MUA_dir.glob(pattern):
            os.remove(f)


def _consolidate_trig_files(trig_dir: Path, n_chunks: int) -> None:
    """Merge per-chunk trigger index files into a single trig_ind.npy and delete chunk files."""
    trig_files = os_sorted(glob.glob(str(trig_dir / 'trig_ind_*')))
    assert len(trig_files) == n_chunks

    trig_is_on = []
    for file in trig_files:
        trig_is_on.extend(np.load(file))

    trig_is_on = np.array(trig_is_on)
    trig_goff = np.where(np.diff(trig_is_on) != 1)[0]
    trig_gon = np.concatenate(([0], trig_goff + 1))
    trig_goff = np.concatenate((trig_goff, [trig_is_on.shape[0] - 1]))
    on_ind = trig_is_on[trig_gon]
    off_ind = trig_is_on[trig_goff]

    np.save(trig_dir / 'trig_ind', (on_ind, off_ind))

    for file in trig_files:
        os.remove(file)


def get_MUA(monkey: str, date: str) -> None:
    """
    Extract multi-unit activity (MUA) from SpikeGLX Neuropixels recordings.

    For each recording found for the given monkey/date:
      1. Reads the .meta and .bin files.
      2. Extracts the digital trigger line and saves per-chunk trigger indices.
      3. Applies gain correction, channel-mean subtraction, CAR, and highpass filtering
         in 20-second chunks (GPU-accelerated via cupy when available).
      4. Detects positive and negative threshold crossings (4 SD) per channel and saves
         spike waveforms, timestamps, amplitudes, SDs, and sign labels as tmp .npy files.
      5. Consolidates tmp files into per-channel final .npy files and removes tmps.
      6. Merges chunked trigger indices into a single trig_ind.npy.

    Output directories created under data_path:
      - MUA_4SD/   : per-channel spike data (wfs, ts, pks, sds, sls, meta)
      - imec_trig/ : merged trigger on/off sample indices

    Args:
        monkey: Monkey identifier (e.g. 'Butter').
        date:   Recording date string (e.g. '20231113').
    """
    data_path_list = get_recording_path(BASE_DATA_PATH, monkey, date)

    for data_path in data_path_list:
        data_path = Path(data_path)
        print(f'\nData path: {data_path}  exists={data_path.exists()}')

        bin_path = next(data_path.glob('*ap.bin'))

        meta = readMeta(bin_path)
        Fs, file_len_sec, n_chans = _read_meta_info(bin_path, meta)
        if n_chans != _DEFAULT_N_CHANS:
            raise ValueError(f"Expected {_DEFAULT_N_CHANS} channels, got {n_chans} in {bin_path}")
        file_len_samps = int(file_len_sec * Fs)

        # Trigger setup
        dig_ch = _get_dig_channel(meta)
        print(f'dig channel: {dig_ch}')
        trig_dir = data_path / 'imec_trig'
        trig_dir.mkdir(exist_ok=False)

        # Filter
        sos = signal.butter(3, _HIGH_PASS_HZ, btype='high', fs=Fs, output='sos')

        win_len_samps = int(_CHUNK_SEC * Fs)
        n_chunks = int(np.ceil(file_len_sec / _CHUNK_SEC))
        print(f'n_chunks: {n_chunks}')

        # Output directory and file handles
        MUA_dir = data_path / 'MUA_4SD'
        MUA_dir.mkdir(exist_ok=False)

        wf_fhand = [open(MUA_dir / f'spike_wfs_ch{ch:03d}_tmp.npy', 'wb') for ch in range(n_chans)]
        pk_fhand = [open(MUA_dir / f'spike_pks_ch{ch:03d}_tmp.npy', 'wb') for ch in range(n_chans)]
        st_fhand = [open(MUA_dir / f'spike_ts_ch{ch:03d}_tmp.npy', 'wb') for ch in range(n_chans)]
        sd_fhand = [open(MUA_dir / f'spike_sds_ch{ch:03d}_tmp.npy', 'wb') for ch in range(n_chans)]
        sl_fhand = [open(MUA_dir / f'spike_sls_ch{ch:03d}_tmp.npy', 'wb') for ch in range(n_chans)]

        raw_data = makeMemMapRaw(bin_path, meta)
        total_t = 0

        for chunk in range(n_chunks):
            print(f'iter {chunk+1} of {n_chunks}')
            t0 = time.perf_counter()

            first_samp = chunk * win_len_samps
            last_samp = min(first_samp + win_len_samps, file_len_samps)

            # Save trigger indices for this chunk
            dig_array = raw_data[dig_ch, first_samp:last_samp].squeeze()
            ind = np.where(dig_array != 0)[0] + first_samp
            np.save(trig_dir / f'trig_ind_{first_samp}_{last_samp}', ind)

            data_filt, chunk_start_sec = _process_chunk(
                raw_data, meta, first_samp, last_samp, sos, n_chans, Fs
            )

            stds_bych = np.std(data_filt, axis=1)

            for ch in range(n_chans):
                wfs, pks, ts, sl = _extract_spikes_channel(
                    data_filt[ch, :], stds_bych, ch, Fs, chunk_start_sec,
                    _SPIKE_THRESH_SD, _MIN_DIST_SAMPS, _LEFT_SAMPS, _RIGHT_SAMPS
                )
                np.save(wf_fhand[ch], wfs)
                np.save(st_fhand[ch], ts)
                np.save(pk_fhand[ch], pks)
                np.save(sd_fhand[ch], stds_bych[ch])
                np.save(sl_fhand[ch], sl)

            gc.collect(generation=2)
            elapsed = time.perf_counter() - t0
            print(f'chunk took {elapsed:.1f} sec')
            total_t += elapsed

        print(f'total took {total_t:.1f} sec')

        for ch in range(n_chans):
            wf_fhand[ch].close()
            st_fhand[ch].close()
            pk_fhand[ch].close()
            sd_fhand[ch].close()
            sl_fhand[ch].close()

        _consolidate_tmp_files(MUA_dir, n_chans, n_chunks, data_path,
                               _SPIKE_THRESH_SD, _CHUNK_SEC, _HIGH_PASS_HZ)
        _consolidate_trig_files(trig_dir, n_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MUA from SpikeGLX recordings')
    parser.add_argument('--monkey', required=True, help='Monkey identifier (e.g. Butter)')
    parser.add_argument('--date', required=True, help='Recording date (e.g. 20231113)')
    args = parser.parse_args()
    get_MUA(args.monkey, args.date)
