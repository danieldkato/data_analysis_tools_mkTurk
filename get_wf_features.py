from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
from sys import platform
from utils_meta import init_dirs
import matplotlib
from make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH

def get_wf_features(n_chan: int, monkey: str, date: str):
    """
    Extract waveform features from multi-unit activity (MUA) data for a specific channel.
    This function loads spike waveform data, timestamps, peaks, and sign labels from processed
    neural recordings, then calculates key waveform features including spike duration, 
    peak-to-trough ratio, and spike height. The features are computed from a random sample
    of action potentials and saved to disk.
    
    Args:
        monkey (str): Identifier for the monkey subject (e.g., 'monkey1', 'monkey2').
        date (str): Recording date in string format (e.g., 'YYYY-MM-DD').
    
    Returns:
        None: The function saves the following files to the save_out_path directory:
            - ch{n_chan:03d}_wf_feaures_ind.npy: Indices of sampled waveforms
            - ch{n_chan:03d}_spike_dur.npy: Spike duration (peak-to-trough time)
            - ch{n_chan:03d}_pt_ratio.npy: Peak-to-trough amplitude ratio
            - ch{n_chan:03d}_spike_height.npy: Total spike height (|peak| + |trough|)
    
    Raises:
        SystemExit: If data_path or save_out_path doesn't exist for all path combinations.
    
    Notes:
        - Requires command line argument sys.argv[1] specifying the channel number to process.
        - Uses a sampling frequency (Fs) of 30000 Hz.
        - Samples up to 1000 action potentials randomly if more are available.
        - Expects MUA data in the 'MUA_4SD' subdirectory with specific file naming conventions.
        - Peak is detected at index 15 of the waveform array.
        - Distinguishes between positive and negative going events for peak/trough identification.
    """

    base_data_path = BASE_DATA_PATH
    base_save_out_path = BASE_SAVE_OUT_PATH

    # load meta files
    print(date)

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

    for n, (data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
        print(save_out_path)
        if not data_path.exists():
            if n == len(data_path_list) -1:
                sys.exit('data path doesn''t exist')
            else:
                continue
        elif not save_out_path.exists():
            if n == len(data_path_list) -1:
                sys.exit('save out path doesn''t exist')
            else:
                continue
        ##############################################################################################################################################
        Fs = 30000

        MUA_dir = data_path / 'MUA_4SD'
        print('MUA_dir = {}'.format(MUA_dir))
        ts_file = next(MUA_dir.glob('ch{:0>3d}_ts.npy'.format(n_chan)))
        pk_file =next(MUA_dir.glob('ch{:0>3d}_pks.npy'.format(n_chan)))
        wf_file =next(MUA_dir.glob('ch{:0>3d}_wfs.npy'.format(n_chan)))
        try:
            sl_file = next(MUA_dir.glob('ch{:0>3d}_sign_label.npy'.format(n_chan)))
        except:
            sl_file = next(MUA_dir.glob('ch{:0>3d}_sls.npy'.format(n_chan)))

        st = np.load(ts_file)
        wf  = np.load(wf_file)
        pk = np.load(pk_file)
        sl = np.load(sl_file)

        print(len(sl))

        n_aps = 1000
        if len(sl) > n_aps:
            ind = np.random.choice(
                        range(len(sl)), size=n_aps, replace=False)
        else:
            ind = range(len(sl))

        spike_dur = []
        pt_ratio = []
        spike_height = []
        for i in range(len(ind)):
            event = pk[ind[i],0]
            event_ind = 15
            if event >0:
                peak = event
                peak_ind = event_ind
                trough = np.min(wf[ind[i],0:30])
                trough_ind = np.argmin(wf[ind[i],0:30])
            elif event <0:
                trough = event
                trough_ind = event_ind
                peak = np.max(wf[ind[i],0:30])
                peak_ind = np.argmax(wf[ind[i],0:30])

            spike_dur.append((peak_ind - trough_ind)/ Fs)
            pt_ratio.append(peak/trough)
            spike_height.append(np.abs(peak) + np.abs(trough))

        spike_dur  = np.array(spike_dur)
        pt_ratio = np.array(pt_ratio)
        spike_height = np.array(spike_height)
        #spike_dur = st[ind,np.argmax(pk[ind,0:2],axis =1)] - st[ind,np.argmin(pk[ind,0:2],axis =1)] # peak - trough
        #pt_ratio = np.max(pk[ind,0:2],axis = 1) / np.min(pk[ind,0:2],axis =1)
        #spike_height = np.abs(np.max(pk[ind,0:2],axis = 1)) + np.abs(np.min(pk[ind,0:2],axis =1))
        
        np.save(save_out_path / 'ch{:0>3d}_wf_feaures_ind.npy'.format(n_chan), ind)
        np.save(save_out_path /'ch{:0>3d}_spike_dur.npy'.format(n_chan), spike_dur)
        np.save(save_out_path /'ch{:0>3d}_pt_ratio.npy'.format(n_chan), pt_ratio)
        np.save(save_out_path /'ch{:0>3d}_spike_height.npy'.format(n_chan), spike_height)



        # pick 10 and save waveform plots
        # n_spikes_to_plot = 20

        # matplotlib.rcParams.update({'font.size': 15})
        # matplotlib.rcParams.update({'figure.facecolor': (1,1,1)})
        # fig, axes = plt.subplots(10,int(n_spikes_to_plot/10),figsize = (5*int(n_spikes_to_plot/10),20))
        # ax = axes.flatten()
        # for i  in range(n_spikes_to_plot):
        #     ax[i].plot(wf[ind[i],:]) # -15 to 40
        #     # peak or trough
        #     event = pk[ind[i],0]
        #     event_ind = 15
        #     if event >0:
        #         peak = event
        #         peak_ind = event_ind
        #         trough = np.min(wf[ind[i],0:30])
        #         trough_ind = np.argmin(wf[ind[i],0:30])
        #     elif event <0:
        #         trough = event
        #         trough_ind = event_ind
        #         peak = np.max(wf[ind[i],0:30])
        #         peak_ind = np.argmax(wf[ind[i],0:30])

        #     # peak = np.max(pk[ind[i],0:2])
        #     # trough = np.min(pk[ind[i],0:2])
        #     # peak_ind = np.where(wf[ind[i],:] == peak)
        #     # trough_ind = np.where(wf[ind[i],:] == trough)
        #     ax[i].scatter(peak_ind, wf[ind[i],peak_ind],c = 'r')
        #     ax[i].scatter(trough_ind, wf[ind[i],trough_ind],c = 'b')
        #     ax[i].set_title('{:2.2f} ms'.format(spike_dur[i]*1000))
        #     ax[i].set_xticklabels([-15,40])

        # plt.tight_layout()
        # plt.suptitle('ch{:0>3d}'.format(n_chan), y = 1.005, fontweight = 'bold')

        # plt.savefig(plot_save_out_path / 'waveform' / 'ch{:0>3d}.png'.format(n_chan), bbox_inches = 'tight')

if __name__ == "__main__":
    n_chan = int(sys.argv[1])
    monkey = sys.argv[2]
    date = str(sys.argv[3])
    get_wf_features(n_chan, monkey, date)