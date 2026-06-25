from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from .utils_meta import init_dirs
import matplotlib.pyplot as plt 
import math
from sys import platform
from .make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

def allchan_wf(monkey: str, date: str):
    """
    Load and aggregate waveform features from individual channel files into a single output file.

    This function processes electrophysiology data for a specific monkey and date, loading
    spike waveform features (spike duration, peak-to-trough ratio, and spike height) from
    individual channel files and consolidating them into a single NPZ file.

    Args:
        monkey (str): Identifier for the monkey subject (e.g., 'monkey1', 'monkey2').
        date (str): Recording date string used to locate the data files.

    Returns:
        None: The function saves output to disk but does not return a value.

    Raises:
        SystemExit: If the data path or save output path does not exist after checking
                    all possible paths in data_path_list.

    Notes:
        - Expects 384 channels of data with up to 1000 action potentials per channel.
        - Loads three feature files per channel: spike_dur, pt_ratio, and spike_height.
        - Output is saved as 'wf_features.npz' containing three arrays:
            * spike_dur: Spike duration for each channel/action potential
            * pt_ratio: Peak-to-trough ratio for each channel/action potential
            * spike_height: Spike height for each channel/action potential
        - Missing data points are filled with NaN values.
        - Uses paths defined by ENGRAM_PATH and assumes specific directory structure
          initialized by init_dirs().
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

        n_chans  = 384
        n_aps = 1000
        spike_dur_all = np.nan * np.ones((n_chans,n_aps))
        pt_ratio_all =  np.nan * np.ones((n_chans,n_aps))
        spike_height_all = np.nan * np.ones((n_chans,n_aps)) 
        for n_chan in range(n_chans):
            spike_dur = np.load(save_out_path / 'ch{:0>3d}_spike_dur.npy'.format(n_chan))
            pt_ratio = np.load(save_out_path / 'ch{:0>3d}_pt_ratio.npy'.format(n_chan))
            spike_height = np.load(save_out_path / 'ch{:0>3d}_spike_height.npy'.format(n_chan))

            spike_dur_all[n_chan,0:len(spike_dur)] = spike_dur
            pt_ratio_all[n_chan,0:len(pt_ratio)] = pt_ratio
            spike_height_all[n_chan,0:len(spike_height)] = spike_height

        np.savez(save_out_path / 'wf_features.npz', spike_dur = spike_dur_all,pt_ratio = pt_ratio_all, spike_height = spike_height_all)
        
if __name__ == "__main__":
    monkey = sys.argv[2]
    date = str(sys.argv[3])
    allchan_wf(monkey, date)