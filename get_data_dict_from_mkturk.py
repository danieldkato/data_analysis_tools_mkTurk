import json
import os
import pickle
import sys
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
from natsort import os_sorted

try:
    from .utils_mkturk import create_data_mat, gen_short_scene_info
    from .utils_meta import init_dirs
    from .make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH
except ImportError:
    from data_analysis_tools_mkTurk.utils_mkturk import create_data_mat, gen_short_scene_info
    from data_analysis_tools_mkTurk.utils_meta import init_dirs
    from data_analysis_tools_mkTurk.make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

_LEN_FILECODE = 6
_MAX_TRIG_DUR_MS = 300
_FS = 30000.0


def _get_filecode(filecode_ind: list, len_filecode: int, out_list: list) -> list:
    """Append filecode index group(s) to out_list, chunking or trimming as needed."""
    if len(filecode_ind) > len_filecode and len(filecode_ind) % len_filecode == 0:
        for i in range(0, len(filecode_ind), len_filecode):
            out_list.append(filecode_ind[i:i + len_filecode])
    elif len(filecode_ind) == len_filecode:
        out_list.append(filecode_ind)
    elif len(filecode_ind) > len_filecode:
        out_list.append(filecode_ind[len(filecode_ind) - len_filecode:])
    return out_list


def _decode_filecodes(trig_on, trig_off, Fs: float) -> tuple:
    """
    Parse imec trigger stream to identify filecodes and per-file sample command indices.

    Returns (filecodes_imec, scs_ind_imec, n_scs_imec, filecodes_ind_imec) as numpy arrays.
    """
    trig_dur = (trig_off - trig_on) / Fs * 1000  # ms

    ind = np.where(trig_dur <= _MAX_TRIG_DUR_MS)[0]
    filecodes_ind_imec_possible = []
    for _, g in groupby(enumerate(ind), lambda ix: ix[0] - ix[1]):
        filecode_ind = list(map(itemgetter(1), g))
        print(filecode_ind, len(filecode_ind))
        filecodes_ind_imec_possible = _get_filecode(
            filecode_ind, _LEN_FILECODE, filecodes_ind_imec_possible
        )
    print(f'{len(filecodes_ind_imec_possible)} possible filecodes found')

    filecodes_imec = []
    scs_ind_imec = []
    n_scs_imec = []
    filecodes_ind_imec = []

    for i, filecode_ind in enumerate(filecodes_ind_imec_possible):
        assert len(filecode_ind) == _LEN_FILECODE, f'filecode length is not {_LEN_FILECODE}'

        filecode_dur = trig_dur[filecode_ind]
        if i == len(filecodes_ind_imec_possible) - 1:
            sc_ind = np.arange(filecode_ind[_LEN_FILECODE - 1] + 1, len(trig_dur))
        else:
            sc_ind = np.arange(filecode_ind[_LEN_FILECODE - 1] + 1,
                               filecodes_ind_imec_possible[i + 1][0])

        n_scs = len(sc_ind)
        f_convert = [round(f / 10 - 1) for f in filecode_dur]
        f_convert = [0 if x < 0 else x for x in f_convert]
        filecode = (f'{f_convert[0]}{f_convert[1]}_'
                    f'{f_convert[2]}{f_convert[3]}_'
                    f'{f_convert[4]}{f_convert[5]}')

        print(i, 'start ind:', filecode_ind[0], 'filecode:', filecode,
              '# of sample commands:', n_scs)
        filecodes_imec.append(filecode)
        scs_ind_imec.append(sc_ind)
        n_scs_imec.append(n_scs)
        filecodes_ind_imec.append(filecode_ind)

    filecodes_ind_imec = np.array(filecodes_ind_imec)
    scs_ind_imec = np.array(scs_ind_imec, dtype='object')
    n_scs_imec = np.array(n_scs_imec)
    filecodes_imec = np.array(filecodes_imec)

    # Fix edge case: first sample command may fire before filecode and get grouped
    # into the previous file as its last sample command
    for idx, (filecode, filecode_ind, scs_ind) in enumerate(
            zip(filecodes_imec, filecodes_ind_imec, scs_ind_imec)):
        if idx > 0 and n_scs_imec[idx - 1] > 0:
            t_diff = (trig_on[filecode_ind[0]]
                      - trig_off[scs_ind_imec[idx - 1][-1]]) / Fs * 1000
            print(idx, t_diff)
            if t_diff < 200:
                print(filecode)
                prev_list = scs_ind_imec[idx - 1]
                scs_ind = np.insert(scs_ind, 0, prev_list[-1])
                scs_ind_imec[idx] = scs_ind
                n_scs_imec[idx] = len(scs_ind)
                scs_ind_imec[idx - 1] = prev_list[:-1]
                n_scs_imec[idx - 1] -= 1

    for i, (filecode, filecode_ind, scs_ind, n_scs) in enumerate(
            zip(filecodes_imec, filecodes_ind_imec, scs_ind_imec, n_scs_imec)):
        print(i, 'filecode:', filecode, '# of sample commands:', n_scs, len(scs_ind))

    return filecodes_imec, scs_ind_imec, n_scs_imec, filecodes_ind_imec


def _match_behav_files(filecodes_imec, scs_ind_imec, n_scs_imec, filecodes_ind_imec,
                       behav_file_list_orig, base_data_path, monkey, penetration) -> tuple:
    """
    Match imec filecodes to mkTurk behavior JSON files.

    Returns (behav_file_list, filecodes_to_analyze, scs_ind_to_analyze,
             n_scs_to_analyze, filecodes_ind_to_analyze).
    """
    behav_file_list = []
    behav_file_list_idx = [-1]
    filecodes_to_analyze = []
    scs_ind_to_analyze = []
    n_scs_to_analyze = []
    filecodes_ind_to_analyze = []

    for i, (f, n_scs, scs_ind, filecodes_ind) in enumerate(
            zip(filecodes_imec, n_scs_imec, scs_ind_imec, filecodes_ind_imec)):
        print(f'\n{i} {f}')
        print('# of scs:', n_scs)

        behav_file = os_sorted(
            Path(base_data_path, monkey, penetration).glob('*' + f + '*.json')
        )
        if len(behav_file) > 0:
            m = json.load(open(behav_file[0].as_posix(), 'rb'))
            n_trials_mk = (len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
                           if len(m['TRIALEVENTS']['TSequenceActualClip']) > 0 else 0)
            print('matching behavior file found')
            print('matched with', behav_file[0].as_posix())
            print('# of mkturk trials:', n_trials_mk)
            print('RewardStage:', m['TASK']['RewardStage'])

            behav_file_list.append(behav_file[0].as_posix())
            behav_file_list_idx.append(
                np.where(np.array(behav_file_list_orig) == behav_file[0].as_posix())[0]
            )
            filecodes_to_analyze.append(f)
            scs_ind_to_analyze.append(scs_ind)
            n_scs_to_analyze.append(n_scs)
            filecodes_ind_to_analyze.append(filecodes_ind)
        else:
            print(f'No corresponding behavior file that matches with filecode {f} from imec')

            for b_f_idx, b_f in enumerate(behav_file_list_orig):
                if b_f in behav_file_list or b_f_idx <= max(behav_file_list_idx):
                    continue

                m = json.load(open(b_f, 'rb'))
                n_trials_mk = (len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
                               if len(m['TRIALEVENTS']['TSequenceActualClip']) > 0 else 0)

                file_time = Path(b_f).stem.split('T')[1].split('_' + monkey)[0]
                file_time_txt = file_time.split('_')
                f_txt = f.split('_')
                diff_all = np.array([
                    abs(int(file_time_txt[j]) - int(f_txt[j])) for j in range(3)
                ])
                str_match = sum(file_time[k] == f[k] for k in range(8))

                if sum(diff_all) <= 3 or len(np.where(diff_all == 0)[0]) == 2 or str_match >= 6:
                    print('filecode is defective but matches most of the datetime string in the behavior file')
                    print('matched with', b_f)
                    print('# of mkturk trials:', n_trials_mk)
                    print('RewardStage:', m['TASK']['RewardStage'])
                    if n_scs == 0 and m['TASK']['RewardStage'] == 0:
                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_to_analyze.append(f)
                        scs_ind_to_analyze.append(scs_ind)
                        n_scs_to_analyze.append(n_scs)
                        filecodes_ind_to_analyze.append(filecodes_ind)
                        break
                    elif abs(n_scs - n_trials_mk) < 7:
                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_to_analyze.append(f)
                        scs_ind_to_analyze.append(scs_ind)
                        n_scs_to_analyze.append(n_scs)
                        filecodes_ind_to_analyze.append(filecodes_ind)
                        break
                    else:
                        print("# of sc and # of mk trials don't match up")
                        break
                else:
                    if n_trials_mk >= n_scs - 2 and n_trials_mk <= n_scs + 2 and n_scs != 0:
                        print('no string matched but the number of sample command triggers seem to match the number of mkturk trials')
                        print('# of mkturk trials:', n_trials_mk)
                        print('matched with', b_f)
                        print('RewardStage:', m['TASK']['RewardStage'])
                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_to_analyze.append(f)
                        scs_ind_to_analyze.append(scs_ind)
                        n_scs_to_analyze.append(n_scs)
                        filecodes_ind_to_analyze.append(filecodes_ind)
                        break

    return (behav_file_list, filecodes_to_analyze, scs_ind_to_analyze,
            n_scs_to_analyze, filecodes_ind_to_analyze)


def _align_triggers(trig_on, trig_off, scs_ind, n_scs, n_scs_imec, idx, Fs):
    """
    Align imec trigger times to mkturk trial count, padding or trimming as needed.

    Returns (trig_on_time, trig_off_time, scs_ind, n_scs) after alignment.
    """
    trig_on_time = trig_on[scs_ind] / Fs
    trig_off_time = trig_off[scs_ind] / Fs
    trig_dur = trig_off_time - trig_on_time
    return trig_on_time, trig_off_time, trig_dur, scs_ind


def _reconcile_trial_counts(trig_on_time, trig_off_time, trig_dur, scs_ind,
                             n_scs, n_trials_mk, sc_dur_mk, n_scs_imec, idx):
    """Pad or trim imec triggers to match mkturk trial count. Returns updated arrays."""
    if n_trials_mk == n_scs:
        print('# of mkturk trials matches # of imec trigs')
        print(np.nansum(np.abs(sc_dur_mk / 1000 - trig_dur)))

    elif n_trials_mk > n_scs:
        print('more mkturk trials than imec trigs')
        n_diff = n_trials_mk - n_scs
        t_diff_first = np.nansum(np.abs(sc_dur_mk[0:n_scs] / 1000 - trig_dur))
        t_diff_last = np.nansum(np.abs(sc_dur_mk[n_diff:n_trials_mk] / 1000 - trig_dur))
        print(f'first {n_diff} mkturk trials:', sc_dur_mk[0:n_diff],
              f'last {n_diff} mkturk trials:', sc_dur_mk[n_scs:n_trials_mk])
        print(f'aligned to first {n_scs}:', t_diff_first,
              f'aligned to last {n_scs}:', t_diff_last)
        if t_diff_first > 5 and t_diff_last > 5:
            print('difference is too big on both ends. Check trig_dur and sc_dur_mk.')
            print(len(np.where(sc_dur_mk < 0)[0]), 'negative sc durations in mkturk file')

        if t_diff_first < t_diff_last:
            print('adds nan values at the end of imec trigger')
            trig_on_time = np.concatenate((trig_on_time, np.nan * np.ones(n_diff)))
            trig_off_time = np.concatenate((trig_off_time, np.nan * np.ones(n_diff)))
        else:
            print('adds nan values at the beginning of imec trigger')
            trig_on_time = np.concatenate((np.nan * np.ones(n_diff), trig_on_time))
            trig_off_time = np.concatenate((np.nan * np.ones(n_diff), trig_off_time))
        n_scs = len(trig_on_time)
        n_scs_imec[idx] = n_scs

    else:  # n_trials_mk < n_scs
        print('more imec trigs than mkturk trials')
        n_diff = n_scs - n_trials_mk
        t_diff_first = np.nansum(np.abs(sc_dur_mk / 1000 - trig_dur[0:n_trials_mk]))
        t_diff_last = np.nansum(np.abs(sc_dur_mk / 1000 - trig_dur[n_diff:n_scs]))
        print(f'first {n_diff} trig:', trig_dur[0:n_diff],
              f'last {n_diff} trig:', trig_dur[n_trials_mk:n_scs])
        print(f'aligned to first {n_trials_mk}:', t_diff_first,
              f'aligned to last {n_trials_mk}:', t_diff_last)
        if t_diff_first > 5 and t_diff_last > 5:
            print('difference is too big on both ends. Check trig_dur and sc_dur_mk.')
            print(len(np.where(sc_dur_mk < 0)[0]), 'negative sc durations in mkturk file')
            print(len(np.where(trig_dur == 0)[0]), '0s found in imec trigger')
            if len(np.where(trig_dur == 0)[0]) == n_diff:
                print(np.nansum(np.abs(sc_dur_mk / 1000 - trig_dur[trig_dur != 0])),
                      'diff after removing 0s from imec trigger')

        if len(np.where(trig_dur == 0)[0]) == n_diff:
            print('removing 0s from imec trigger')
            scs_ind = scs_ind[trig_dur != 0]
            trig_on_time = trig_on_time[trig_dur != 0]
            trig_off_time = trig_off_time[trig_dur != 0]
            trig_dur = trig_dur[trig_dur != 0]
        else:
            if t_diff_last > t_diff_first:
                scs_ind = scs_ind[0:n_trials_mk]
                trig_on_time = trig_on_time[0:n_trials_mk]
                trig_off_time = trig_off_time[0:n_trials_mk]
                print(f'removing {n_scs - n_trials_mk} sample commands from the end')
            else:
                scs_ind = scs_ind[n_diff:n_scs]
                trig_on_time = trig_on_time[n_diff:n_scs]
                trig_off_time = trig_off_time[n_diff:n_scs]
                print(f'removing {n_scs - n_trials_mk} sample commands from the beginning')
        n_scs = len(scs_ind)
        n_scs_imec[idx] = n_scs

    return trig_on_time, trig_off_time, scs_ind, n_scs


def _build_stim_info_sess(data_dict: dict) -> dict:
    """Aggregate per-stimulus timing and metadata across all presentations into stim_info_sess."""
    unique_stim = []
    stim_all, stim_t_all, stim_t_mk_all = [], [], []
    stim_present_bool, stim_rsvp_num, stim_trial_num = [], [], []
    reward_bool, stim_scenefile, stim_dur_all, stim_iti_dur_all = [], [], [], []

    for n_stim in data_dict:
        stim_all.append(data_dict[n_stim]['stim_info_short'])
        stim_rsvp_num.append(data_dict[n_stim]['rsvp_num'])
        stim_trial_num.append(data_dict[n_stim]['trial_num'])
        reward_bool.append(data_dict[n_stim]['reward'])

        t_on_mk = data_dict[n_stim]['imec_trig_on'] + data_dict[n_stim]['t_mk'] / 1000

        if type(data_dict[n_stim]['ph_t_rise']) == float:
            stim_dur_s = np.unique(np.array(data_dict[n_stim]['stim_info'].loc[:, 'dur'].tolist()))[0] / 1000
            t_on_ph = (data_dict[n_stim]['imec_trig_on']
                       + data_dict[n_stim]['ph_t_rise'] / 1000
                       + stim_dur_s * data_dict[n_stim]['rsvp_num']
                       + data_dict[n_stim]['iti_dur'] / 1000 * data_dict[n_stim]['rsvp_num'])
        else:
            t_on_ph = np.nan

        stim_present_bool.append(0 if data_dict[n_stim]['t_mk'] == -1 else 1)
        stim_t_mk_all.append(t_on_mk)
        stim_t_all.append(t_on_ph)
        stim_scenefile.append(data_dict[n_stim]['scenefile'])

        if data_dict[n_stim]['stim_info_short'] not in unique_stim:
            unique_stim.append(data_dict[n_stim]['stim_info_short'])

        stim_dur_all.append(
            np.unique(np.array(data_dict[n_stim]['stim_info'].loc[:, 'dur'].tolist()))[0] / 1000
        )
        stim_iti_dur_all.append(data_dict[n_stim]['iti_dur'])

    stim_all = np.array(stim_all)
    stim_t_all = np.array(stim_t_all)
    stim_t_mk_all = np.array(stim_t_mk_all)
    stim_present_bool = np.array(stim_present_bool)
    stim_rsvp_num = np.array(stim_rsvp_num)
    reward_bool = np.array(reward_bool)
    stim_scenefile = np.array(stim_scenefile)
    stim_dur_all = np.array(stim_dur_all)
    stim_trial_num = np.array(stim_trial_num)
    stim_iti_dur_all = np.array(stim_iti_dur_all)

    stim_info_sess = {}
    for stim in unique_stim:
        ind = np.where(stim_all == stim)[0]
        stim_info_sess[stim] = {
            'stim_ind': ind,
            't_on': stim_t_all[ind],
            't_on_mk': stim_t_mk_all[ind],
            'dur': stim_dur_all[ind],
            'iti_dur': stim_iti_dur_all[ind] / 1000,
            'present_bool': stim_present_bool[ind],
            'rsvp_num': stim_rsvp_num[ind],
            'trial_num': stim_trial_num[ind],
            'reward_bool': reward_bool[ind],
            'scenefile': stim_scenefile[ind],
        }

    print('# of unique stimulus prepared during the session:', len(unique_stim))
    return stim_info_sess


def get_data_dict_from_mkturk(monkey: str, date: str) -> None:
    """
    Build and save data dictionaries aligning mkTurk behavior files with imec trigger data.

    For each recording found for the given monkey/date:
      1. Loads imec trigger on/off sample indices from imec_trig/trig_ind.npy.
      2. Decodes filecodes from the trigger stream to identify behavior files.
      3. Matches each filecode to a mkTurk JSON behavior file.
      4. Aligns imec trigger times to mkturk trial counts (padding/trimming mismatches).
      5. Builds a per-stimulus data_dict and saves it as data_dict_<session_name>.pkl.
      6. Aggregates stimulus timing into stim_info_sess and saves it.
      7. Writes the unique stimulus list to stim_list.json.

    Args:
        monkey: Monkey identifier (e.g. 'Butter').
        date:   Recording date string (e.g. '20231113').
    """
    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(
        BASE_DATA_PATH, monkey, date, BASE_SAVE_OUT_PATH
    )

    for n, (data_path, save_out_path, plot_save_out_path) in enumerate(
            zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        if not isinstance(save_out_path, Path):
            save_out_path = Path(save_out_path)
        print(save_out_path)
        print('\nsave out path found:', save_out_path.exists())

        if not save_out_path.exists():
            os.makedirs(save_out_path, exist_ok=True)

        penetration = data_path.relative_to(BASE_DATA_PATH / monkey).as_posix().split('/')[0]
        trig_path = data_path / 'imec_trig'

        if not data_path.exists():
            if n == len(data_path_list) - 1:
                sys.exit("data path doesn't exist")
            continue
        elif not trig_path.exists():
            if n == len(data_path_list) - 1:
                sys.exit("trig path doesn't exist")
            continue

        try:
            trig_on, trig_off = np.load(trig_path / 'trig_ind.npy')
        except Exception:
            continue

        if len(trig_on) == 0 and len(trig_off) == 0:
            sys.exit('trig file is empty')

        # Behavior files in the penetration directory
        behav_file_list_orig = os_sorted(
            Path(BASE_DATA_PATH, monkey, penetration).glob('*.json')
        )
        print('Number of behavior files in the data path:', len(behav_file_list_orig))
        for i, b_f in enumerate(behav_file_list_orig):
            m = json.load(open(b_f, 'rb'))
            n_trials_mk_prepared = len(m['TRIALEVENTS']['Sample']['0'])
            n_trials_mk_shown = (len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
                                 if len(m['TRIALEVENTS']['TSequenceActualClip']) > 0 else 0)
            behav_file_list_orig[i] = b_f.as_posix()
            print(b_f.stem, n_trials_mk_prepared, n_trials_mk_shown)

        filecodes_imec, scs_ind_imec, n_scs_imec, filecodes_ind_imec = _decode_filecodes(
            trig_on, trig_off, _FS
        )

        behav_file_list, filecodes_to_analyze, scs_ind_to_analyze, n_scs_to_analyze, \
            filecodes_ind_to_analyze = _match_behav_files(
                filecodes_imec, scs_ind_imec, n_scs_imec, filecodes_ind_imec,
                behav_file_list_orig, BASE_DATA_PATH, monkey, penetration
            )

        assert len(behav_file_list) == len(n_scs_to_analyze)
        print(len(behav_file_list_orig) - len(behav_file_list), 'behavior files are unmatched\n')
        print('unmatched:', list(set(behav_file_list_orig) - set(behav_file_list)))
        print('trigs unaccounted for:',
              len(trig_on) - (np.sum(n_scs_to_analyze)
                              + _LEN_FILECODE * len(filecodes_to_analyze)))

        n_stims_sess = 0
        n_trials_sess = 0
        data_dict_all = {}

        for idx, (behav_file, scs_ind, n_scs, filecode_ind) in enumerate(
                zip(behav_file_list, scs_ind_to_analyze,
                    n_scs_to_analyze, filecodes_ind_to_analyze)):
            print(behav_file)
            m = json.load(open(behav_file, 'rb'))

            if m['TASK']['RewardStage'] == 0:
                print('calibration file. removing it from further analysis')
                continue

            if len(m['TASK']['ImageBagsSample']) != len(m['SCENES']['SampleScenes']):
                print("ImageBagSample and number of scenefiles don't match. skip.")
                continue

            data_dict = create_data_mat(behav_file)
            print(m['TASK']['ImageBagsSample'])

            try:
                sc_off_mk = np.array(m['TRIALEVENTS']['SampleCommandOffReturnTime'], dtype=float)
            except Exception:
                print('no SampleCommandOffReturnTime in the behavior file')
                sc_off_mk = [np.nan]

            try:
                sc_on_mk = np.array(m['TRIALEVENTS']['SampleCommandReturnTime'], dtype=float)
            except Exception:
                print('no SampleCommandReturnTime in the behavior file')
                sc_on_mk = [np.nan]

            n_rsvp = (max(m['TASK']['NRSVP'], m['TASK']['NRSVPMax'])
                      if 'NRSVPMax' in m['TASK'] else m['TASK']['NRSVP'])

            if len(sc_on_mk) != len(sc_off_mk):
                print('mkturk behavior file has mismatched sample command on and off')
                if sc_on_mk[0] < sc_off_mk[0] and len(sc_on_mk) > len(sc_off_mk):
                    sc_off_mk = np.concatenate(
                        (sc_off_mk, np.nan * np.ones(len(sc_on_mk) - len(sc_off_mk)))
                    )

            sc_dur_mk = sc_off_mk - sc_on_mk
            n_trials_mk = len(sc_dur_mk)

            print(f'# of trials from mkturk file: {n_trials_mk}\n'
                  f'# of imec triggers: {n_scs}')

            if n_trials_mk <= 10 and n_scs <= 10:
                print('too few trials. Removing it from further analysis')
                continue

            trig_on_time, trig_off_time, trig_dur, scs_ind = _align_triggers(
                trig_on, trig_off, scs_ind, n_scs, n_scs_imec, idx, _FS
            )
            mean = np.mean(trig_dur)
            sd = np.std(trig_dur)
            print('mean trig dur:', mean, 'sd:', sd)

            trig_on_time, trig_off_time, scs_ind, n_scs = _reconcile_trial_counts(
                trig_on_time, trig_off_time, trig_dur, scs_ind,
                n_scs, n_trials_mk, sc_dur_mk, n_scs_imec, idx
            )

            assert n_trials_mk == n_scs == len(trig_on_time) == len(trig_off_time)
            trig_on_time = np.repeat(trig_on_time, n_rsvp)
            trig_off_time = np.repeat(trig_off_time, n_rsvp)

            for n_stim in data_dict:
                data_dict[n_stim]['imec_trig_on'] = trig_on_time[n_stim]
                data_dict[n_stim]['imec_trig_off'] = trig_off_time[n_stim]
                data_dict[n_stim]['stim_info_short'] = gen_short_scene_info(
                    data_dict[n_stim]['stim_info']
                )

            data_dict_new = {
                n_stim + n_stims_sess: {
                    **data_dict[n_stim],
                    'trial_num': data_dict[n_stim]['trial_num'] + n_trials_sess,
                }
                for n_stim in data_dict
            }
            data_dict_all.update(data_dict_new)
            n_stims_sess += len(data_dict)
            n_trials_sess += n_trials_mk
            print(n_stims_sess)

        print('total # of stimulus presentations prepared in this session:', n_stims_sess)

        save_out_file_name = 'data_dict_' + data_path.name
        pickle.dump(data_dict_all, open(save_out_path / save_out_file_name, 'wb'), protocol=2)

        data_dict = pickle.load(open(save_out_path / save_out_file_name, 'rb'))
        stim_info_sess = _build_stim_info_sess(data_dict)

        print(save_out_path)
        pickle.dump(stim_info_sess, open(save_out_path / 'stim_info_sess', 'wb'), protocol=2)

        with open(save_out_path / 'stim_list.json', 'w') as f:
            json.dump(list(stim_info_sess.keys()), f)


if __name__ == '__main__':
    monkey = sys.argv[2]
    date = sys.argv[3]
    get_data_dict_from_mkturk(monkey, date)
