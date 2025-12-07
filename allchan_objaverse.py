from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from utils_ephys import get_FSI_allchans, get_FSI_heatmap
from utils_meta import init_dirs, get_chanmap, get_coords_sess
import matplotlib.pyplot as plt 
import math
from make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

def allchan_objaverse(monkey: str, date: str):
    """
    Process and analyze objaverse PSTH data across all channels for a given monkey and date.
    This function loads per-channel PSTH (Peri-Stimulus Time Histogram) data, consolidates it into
    a single multi-dimensional array, computes Face Selectivity Index (FSI) metrics, and generates
    heatmap visualizations with probe coordinate information.
    
    Args:
        monkey (str): The identifier for the monkey subject (e.g., 'monkey1', 'monkey2').
        date (str): The date of the recording session in string format.
        
    Returns:
        None: The function saves outputs to disk rather than returning values.
        
    Side Effects:
        - Creates and saves the following files in the save_out_path directory:
            - 'psth_objaverse_allchan.npy': Combined PSTH data array with shape 
              (n_chans, n_stims, n_trials_max, psth_len)
            - 'psth_objaverse_meta': Pickled metadata containing stimulus list and trial counts
        - Generates and saves FSI heatmap figures to plot_save_out_path and base_save_out_path
        - Prints progress information including save paths, stimulus counts, and probe coordinates
        
    Raises:
        SystemExit: If data_path or save_out_path doesn't exist for the last path in the list.
        
    Notes:
        - Assumes 384 channels of recording data
        - Uses NaN padding for variable trial counts and PSTH lengths across stimuli
        - Generates heatmaps for different identity categories: 'all', 'oldman', 'neptune', 
          'elias', 'neptune_big', and 'sophie' (conditional on data availability)
        - Adjusts AP coordinates for hanging angle (HAng) recordings
        - Requires previously generated per-channel PSTH pickle files
    """

    base_data_path = BASE_DATA_PATH
    base_save_out_path = BASE_SAVE_OUT_PATH

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)
    for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
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

if __name__ == "__main__":
    monkey = sys.argv[2]
    date = str(sys.argv[3])
    allchan_objaverse(monkey, date)