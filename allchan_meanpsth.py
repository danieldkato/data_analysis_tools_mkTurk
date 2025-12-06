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
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta

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

    n_chans = 384 
    file_lists= os_sorted(save_out_path.glob('*_mean_psth.npy'))
    if len(file_lists) == n_chans:
        psth = np.load(file_lists[0])
        n_stims = psth.shape[0]
        mean_psth_stim = np.nan * np.ones((n_chans, n_stims, psth.shape[1]))
        for ch in range(n_chans):
            mean_psth_stim[ch,:] = np.load(file_lists[ch])

        np.save(save_out_path / 'mean_psth_allchan.npy', mean_psth_stim)
