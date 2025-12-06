# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:58:42 2024

@author: danie
"""

import os
import socket
import numpy as np

# Define base data path:
hostname = socket.gethostname()
if hostname == 'DESKTOP-1PVCRAF':
    drive = 'V:'
    folder_level_offset = 0
    local_base = os.path.join('F:\\')
    local_data_path = os.path.join(local_base, 'h5s_test')
    h5dir = local_data_path
elif hostname == 'laptop':
    drive = 'X:'
    local_base = os.path.join('C:\\', 'Users', 'danie', 'Documents')
    folder_level_offset = 0
    local_data_path = os.path.join(local_base, 'h5s_test')
    h5dir = local_data_path
elif 'rc.zi.columbia.edu' in hostname:
    drive = '/mnt/smb/locker/issa-locker'
    folder_level_offset = 4
mnt = drive + os.path.sep
base_data_path = mnt +'Data'

from data_analysis_tools_mkTurk.utils_meta import get_recording_path
from data_analysis_tools_mkTurk.IO import h5_2_trial_df, find_im_full_paths


#date = '20231011'
date = '20240410'
h5_path = os.path.join(h5dir, '{}.h5'.format(date))
trial_params_df = h5_2_trial_df(h5_path)
if 'stim_idx' not in trial_params_df.columns:
    stim_idx = np.floor(np.random.uniform(0,2,trial_params_df.shape[0]))
    trial_params_df['stim_idx'] = stim_idx
im_paths = find_im_full_paths(trial_params_df, local_data_path=base_data_path)