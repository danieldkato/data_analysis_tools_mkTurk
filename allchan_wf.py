from natsort import os_sorted
import pickle
import pathlib as Path
import os
import sys
import pandas as pd
import numpy as np
from utils_ephys import * 
from utils_meta import * 
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

for n, (data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
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