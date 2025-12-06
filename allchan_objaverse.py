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

import socket 
host = socket.gethostname()
if 'rc.zi.columbia.edu' in host: 
    engram_path = Path.Path('/mnt/smb/locker/issa-locker')
elif 'DESKTOP' in host:
    engram_path = Path.Path('Z:/')
elif 'Younah' in host:
    engram_path = Path.Path('/Volumes/issa-locker/')

base_data_path = engram_path  / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

monkey = sys.argv[2]
date = str(sys.argv[3])

data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)
for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
    print(save_out_path)
    print('\nsave out path found: '+ str(save_out_path.exists()))

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

    psth_ex = pickle.load(open( save_out_path / 'ch{:0>3d}_objaverse_psth'.format(0),'rb'))
    stim_list = list(psth_ex.keys())

    n_stims = len(stim_list)
    psth_len= 0
    n_trials_max = 0
    for stim in psth_ex:
        psth_len = np.nanmax((psth_len,psth_ex[stim].shape[1]))
        n_trials_max = np.nanmax((n_trials_max,psth_ex[stim].shape[0]))

    print(n_stims,n_trials_max,psth_len)
    
    psth_allchan = np.nan * np.ones((n_chans,n_stims,n_trials_max,psth_len))
    for ch in range(n_chans):
        n_trials_stim = []
        psth = pickle.load(open( save_out_path / 'ch{:0>3d}_objaverse_psth'.format(ch),'rb'))
        for n, stim in enumerate(stim_list):
            psth_len = psth[stim].shape[1]
            n_trials = psth[stim].shape[0]
            n_trials_stim.append(n_trials)
            psth_allchan[ch,n,0:n_trials,0:psth_len] = psth[stim]
    psth_objaverse_meta = [(s,n) for s, n in zip(stim_list,n_trials_stim)]
    np.save(save_out_path / 'psth_objaverse_allchan.npy', psth_allchan)
    pickle.dump(psth_objaverse_meta, open(save_out_path / 'psth_objaverse_meta','wb'), protocol = 2)
    
    FSI_concat, FSI = get_FSI_allchans(save_out_path)
    
    try: 
        chanmap = np.load(save_out_path / 'chanmap.npy')
    except:
        chanmap, imroTbl = get_chanmap(data_path)


    ap_coord, dv_coord, ml_coord, ang, hang, dep = get_coords_sess(base_data_path, monkey, str(date))
    print(ap_coord, dv_coord, ml_coord, ang, hang, dep)
    max_depth = dep
    
    if 'HAng' in data_path.name:
        if hang  !=0:
            ap_coord =round(ap_coord -  1/np.cos(hang*np.pi/180),1)

    fig = get_FSI_heatmap(FSI_concat, FSI,plot_save_out_path, 'all', chanmap)



    if 'HAng' in data_path.name:
        fig.savefig(base_save_out_path / (monkey + '_face') / ('FSI_all_' +  date + '_ap_' + str(ap_coord) + '_dv_' + str(dv_coord) + '_ml_' + str(ml_coord) + \
                                                                                        '_ang_' + str(round(90-ang)) + '_hang_' + str(round(hang)) + '_dep_' + str(max_depth) + '.png'), bbox_inches = 'tight')
    else:
        fig.savefig(base_save_out_path / (monkey + '_face') / ('FSI_all_' + date + '_ap_' + str(ap_coord) + '_dv_' + str(dv_coord) + '_ml_' + str(ml_coord) + \
                                                                                        '_ang_' + str(round(90-ang)) + '_dep_' + str(max_depth) + '.png'), bbox_inches = 'tight')

    fig = get_FSI_heatmap(FSI_concat, FSI,plot_save_out_path, 'oldman', chanmap)
    fig = get_FSI_heatmap(FSI_concat, FSI, plot_save_out_path, 'neptune', chanmap)
    

    if not np.isnan(FSI_concat['elias']).all():
        fig = get_FSI_heatmap(FSI_concat, FSI,plot_save_out_path, 'elias', chanmap)
    
    if not np.isnan(FSI_concat['neptune_big']).all():
        fig = get_FSI_heatmap(FSI_concat, FSI, plot_save_out_path, 'neptune_big', chanmap)
    
    if not np.isnan(FSI_concat['sophie']).all():
        fig = get_FSI_heatmap(FSI_concat, FSI, plot_save_out_path, 'sophie', chanmap)

