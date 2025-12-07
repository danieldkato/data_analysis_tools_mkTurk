from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from utils_meta import init_dirs
import matplotlib.pyplot as plt 
import math
from make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

def allchan_meanpsth(monkey: str, date: str):
    """
    Aggregate mean PSTH data from individual channels into a single file.

    This function loads mean PSTH (Peri-Stimulus Time Histogram) data from individual 
    channel files and combines them into a single multi-dimensional array containing 
    data for all channels.

    Args:
        monkey (str): Identifier for the monkey subject.
        date (str): Date of the recording session in string format.

    Returns:
        None: The function saves the aggregated data to disk but does not return a value.

    Side Effects:
        - Prints the date and save output path to console
        - Creates a combined numpy array file 'mean_psth_allchan.npy' in the save output directory
        - Exits the program if required paths don't exist after checking all path variations

    Raises:
        SystemExit: If data_path or save_out_path don't exist for the last item in the path lists.

    Notes:
        - Expects exactly 384 channel files with naming pattern '*_mean_psth.npy'
        - Only processes data if all 384 channel files are present
        - Uses ENGRAM_PATH global variable for base path construction
        - Output shape: (n_chans, n_stims, time_bins) where n_chans=384
    """
    base_data_path = BASE_DATA_PATH
    base_save_out_path = BASE_SAVE_OUT_PATH

    print(date)

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

    for n, (data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
        print(save_out_path)
        if not data_path.exists():
            if n == len(data_path_list)-1:
                sys.exit('data path doesn''t exist')
            else:
                continue
        elif not save_out_path.exists():
            if n == len(data_path_list)-1:
                sys.exit('save out path doesn''t exist')
            else:
                continue

        n_chans = 384 
        file_lists= os_sorted(save_out_path.glob('*_mean_psth.npy'))
        if len(file_lists) == n_chans:
            psth = np.load(file_lists[0])
            n_stims = psth.shape[0]
            mean_psth_stim = np.nan * np.ones((n_chans, n_stims, psth.shape[1]))
            for ch in range(n_chans):
                mean_psth_stim[ch,:] = np.load(file_lists[ch])

            np.save(save_out_path / 'mean_psth_allchan.npy', mean_psth_stim)

if __name__ == "__main__":
    monkey = sys.argv[2]
    date = str(sys.argv[3])
    allchan_meanpsth(monkey, date)