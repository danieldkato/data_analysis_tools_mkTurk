from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from utils_meta import init_dirs
import matplotlib
import matplotlib.pyplot as plt 
import math
from sys import platform
from make_engram_path import ENGRAM_PATH


def allchan_bl(monkey: str, date: str):
    """
    Calculate and save baseline activity statistics across all channels for neural recording data.
    
    This function processes neural recording data for a specific monkey and date, computing
    baseline firing rate statistics (mean and standard deviation) across all channels and
    stimuli. It saves the results and generates a visualization of mean baseline activity.
    
    Args:
        monkey: Identifier for the monkey subject (e.g., 'monkey1', 'monkey2').
        date: Date of the recording session in string format.
    
    Returns:
        None. The function saves outputs to disk:
            - 'bl_allchan.npy': NumPy array of shape (n_chans, n_stims, 2) containing
              mean and standard deviation of baseline activity for each channel and stimulus.
            - 'baseline activity across channels.png': Plot showing mean baseline firing
              rate across all channels.
    
    Notes:
        - The function expects baseline data files named 'ch{:0>3d}_psth_bl_stim' to exist
          in the save_out_path directory for each channel.
        - Baseline period is defined as -200ms to trial onset.
        - The function processes 384 channels by default.
        - Firing rates in the plot are converted to spikes/sec (multiplied by 100).
        - The function will exit if required data or save paths don't exist.
    
    Raises:
        SystemExit: If data_path or save_out_path doesn't exist.
    """
    base_data_path = ENGRAM_PATH / 'Data'
    base_save_out_path = ENGRAM_PATH / 'users/Younah/ephys'

    print(date)

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)


    for data_path, save_out_path, plot_save_out_path in zip(data_path_list, save_out_path_list, plot_save_out_path_list):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
        print(save_out_path)
        if not data_path.exists():
            sys.exit('data path doesn''t exist')
        elif not save_out_path.exists():
            sys.exit('save out path doesn''t exist')

        n_chans = 384
        bl = pickle.load(open(save_out_path / 'ch{:0>3d}_psth_bl_stim'.format(0),'rb'))
        stim_list = list(bl.keys())
        n_stims = len(stim_list)

        bl_dict = dict.fromkeys(stim_list) #store mean and sd baseline activity per stimulus
        bl_allchan = np.nan * np.ones((n_chans,n_stims,2))
        for n,stim in enumerate(stim_list):
            bl_dict[stim] = np.nan * np.ones((n_chans,2))
            for ch in range(n_chans):
                bl = pickle.load(open(save_out_path / 'ch{:0>3d}_psth_bl_stim'.format(ch),'rb'))
                
                assert set(stim_list) == set(list(bl.keys()))
                
                bl_across_trials = np.nanmean(bl[stim],axis = 0)
                sd_bl = np.nanstd(bl_across_trials)
                mean_bl = np.nanmean(bl_across_trials)
                bl_allchan[ch,n]= (mean_bl, sd_bl)

        np.save(save_out_path / 'bl_allchan.npy', bl_allchan)


        fig, ax = plt.subplots(figsize = (20,5))

        matplotlib.rcParams.update({'font.size': 15})
        matplotlib.rcParams.update({'figure.facecolor': (1,1,1)})
        plt.plot(np.nanmean(bl_allchan[:,:,0],axis = 1) * 100)
        plt.xlabel('ch #')
        plt.title('mean baseline activity per channel \n(-200ms to the onset of trial)')
        plt.ylabel ('FR (spikes / sec)')
        fig.savefig(plot_save_out_path / 'baseline activity across channels.png', bbox_inches = 'tight')
