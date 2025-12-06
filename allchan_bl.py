from natsort import os_sorted
import pickle
import pathlib as Path
import os
import sys
import pandas as pd
import numpy as np
from utils_ephys import * 
from utils_meta import * 
import matplotlib
import matplotlib.pyplot as plt 
import math
from sys import platform

engram_path = Path.Path('/mnt/smb/locker/issa-locker')

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

monkey = sys.argv[2]
date = str(sys.argv[3])
print(date)

data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)


for data_path, save_out_path, plot_save_out_path in zip(data_path_list, save_out_path_list, plot_save_out_path_list):
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