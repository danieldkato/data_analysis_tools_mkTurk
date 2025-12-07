from pathlib import Path
import sys
import numpy as np
from natsort import os_sorted
from itertools import chain
import pickle
import math
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from sys import platform
from utils_meta import init_dirs, read_snsChanMap, read_imroTbl
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta
from make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH


def get_psth_objaverse(n_chan: int, monkey: str, date: str):
    """
    Calculate and visualize peri-stimulus time histograms (PSTH) for Objaverse stimuli.
    This function processes neural recording data for a specific monkey and date, computing
    PSTHs for various Objaverse 3D object stimuli. It generates visualizations comparing
    neural responses to face vs. non-face stimuli and calculates Face Selectivity Index (FSI).
    
    Args:
        monkey (str): Identifier for the monkey subject (e.g., 'monkey1', 'monkey2')
        date (str): Recording date in string format
    
    Returns:
        None: Function saves outputs to disk including:
            - Pickled PSTH data per channel
            - PNG plots of individual stimulus responses
            - PNG plots comparing face vs. object responses
            - FSI (Face Selectivity Index) calculations
    
    Side Effects:
        - Reads neural recording data from BASE_DATA_PATH
        - Reads stimulus metadata from objaverse.json
        - Creates output directories if they don't exist
        - Saves pickled data files to save_out_path
        - Saves plot images to plot_save_out_path/objaverse/
        - Prints status messages and warnings to console
        - May exit program if critical paths don't exist
    
    Notes:
        - Processes multiple recording sessions for the given date
        - Uses 10ms time bins for PSTH calculation
        - Includes 100ms before and after stimulus presentation
        - Calculates FSI for different face stimuli variants (oldman, neptune, elias, sophie, neptune_big)
        - FSI = (face_response - object_response) / (face_response + object_response)
        - Requires channel number to be passed via sys.argv[1]
    """

    base_data_path = BASE_DATA_PATH
    base_save_out_path = BASE_SAVE_OUT_PATH

    # Objaverse data
    objaverse_data =json.load(open(base_data_path / monkey/'stim_info/objaverse.json','rb'))

    objaverse_id = objaverse_data['stim_id']
    objaverse_color = objaverse_data['stim_color']

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)
    for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
        print(save_out_path)
        if not data_path.exists():
            if n!=0:
                sys.exit('data path doesn''t exist')
            else:
                continue

        if not save_out_path.exists():
            print('save out path doesn''t exist')
            if n!=0:
                sys.exit()
            else:
                continue
        if not (plot_save_out_path / 'objaverse').exists():
            os.makedirs(plot_save_out_path / 'objaverse',exist_ok= True)

        try:
            meta_iter = data_path.glob('*ap.meta')
            bin_iter = data_path.glob('*ap.bin')   
            meta_path = next(meta_iter)
            bin_path = next(bin_iter) 

            meta = readMeta(bin_path)
            chanmap = read_snsChanMap(meta)
            imroTbl = read_imroTbl(meta)
        except:
            chanmap = np.load(save_out_path / 'chanmap.npy')

        # modify the channel map by channel mapping
        # this only affects plots not actual data
        n_chan_new = int(chanmap[n_chan,1])

        psth = pickle.load(open(save_out_path / 'ch{:0>3d}_psth_stim'.format(n_chan),'rb'))
        psth_bl = pickle.load(open(save_out_path / 'ch{:0>3d}_psth_bl_stim'.format(n_chan),'rb'))
        psth_meta = pickle.load(open(save_out_path / 'psth_stim_meta'.format(n_chan),'rb'))

        stim_info_sess = pickle.load(open(save_out_path / 'stim_info_sess','rb'))
        stim_ids_all = list(stim_info_sess.keys())

        # ignore dur part
        stim_ids_all = np.array([s.split('_dur')[0] for s in stim_ids_all])

        stim_dur_all = []
        for id in objaverse_id:
            if id in stim_ids_all:
                ind = np.where(stim_ids_all== id)[0][0]
                id = list(psth_meta.keys())[ind]
                stim_dur_all.append(psth_meta[id]['stim_dur'])

        if len(stim_dur_all) == 0:
            print('no objaverse stimuli in this recording')
        else:
            binwidth_psth = 0.01 # 10 ms 
            t_before = 0.1 # 100ms before
            t_after = 0.1 # 100ms after 
            stim_dur = max(stim_dur_all)
            print(stim_dur)

            n_bins = int(np.ceil((stim_dur + t_before + t_after)/binwidth_psth))
            psth_bins = np.linspace(-t_before, stim_dur + t_after, n_bins+1)
            bincents = psth_bins - binwidth_psth/2
            bincents = bincents[1:]

            objaverse_stim = []
            psth_objaverse = []
            sd_objaverse = []
            psth_face = []
            psth_oldman = []
            psth_neptune = []
            psth_nonface = []

            psth_bl_face = []
            psth_bl_oldman = []
            psth_bl_neptune = []
            psth_bl_nonface = []

            n_trials_objaverse = dict()
            n_trials_face = []
            n_trials_nonface = []
            objaverse_psth = dict()
            for id in objaverse_id:
                if id in stim_ids_all:
                    ind = np.where(stim_ids_all== id)[0][0]
                    id = list(psth_meta.keys())[ind]
                    stim_dur_all.append(psth_meta[id]['stim_dur'])
                    print(psth_meta[id].keys())
                    n_trials = psth_meta[id]['n_trials']
                    print(n_trials)

                    objaverse_psth[id] =psth[id][np.array([~np.isnan(p).all() for p in psth[id]])]
                    psth_mean = np.nanmean(objaverse_psth[id] * 100, axis = 0)

                    psth_objaverse.append(psth_mean)
                    sd_objaverse.append(np.nanstd(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])] * 100, axis = 0)/np.sqrt(n_trials))
                    #sd_objaverse.append(np.nanstd(psth[id]['psth'], axis = 0))
                    n_trials_objaverse[id] = n_trials
                    
                    if 'neptune' in id:
                        psth_neptune.append(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])])
                        psth_face.append(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])])
                        psth_bl_neptune.append(psth_bl[id])
                        psth_bl_face.append(psth_bl[id])
                        n_trials_face.append(n_trials)
                        objaverse_stim.append(id.split('/')[1].split('.')[0])
                    elif 'oldman' in id:
                        psth_oldman.append(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])])
                        psth_face.append(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])])
                        psth_bl_oldman.append(psth_bl[id])
                        psth_bl_face.append(psth_bl[id])
                        n_trials_face.append(n_trials)
                        objaverse_stim.append(id.split('/')[1].split('.')[0])
                    elif 'face' not in id:
                        psth_nonface.append(psth[id][np.array([~np.isnan(p).all() for p in psth[id]])])
                        psth_bl_nonface.append(psth_bl[id])
                        n_trials_nonface.append(n_trials)
                        objaverse_stim.append(id.split('/')[0])

            filename= 'ch{:0>3d}_objaverse_psth'.format(n_chan)

            pickle.dump(objaverse_psth, open(save_out_path / filename,'wb'),protocol = 2)

            psth_face = np.vstack(psth_face)
            psth_nonface = np.vstack(psth_nonface)
            if len(psth_oldman) > 0:
                psth_oldman = np.vstack(psth_oldman)
            if len(psth_neptune) > 0:
                psth_neptune = np.vstack(psth_neptune)

            if len(psth_bl_face) > 0:
                psth_bl_face = np.vstack(psth_bl_face)  
            if len(psth_bl_nonface) > 0:
                psth_bl_nonface = np.vstack(psth_bl_nonface)
            if len(psth_bl_oldman) > 0:
                psth_bl_oldman = np.vstack(psth_bl_oldman)
            if len(psth_bl_neptune) > 0:
                psth_bl_neptune = np.vstack(psth_bl_neptune)

            # plot
            fig, ax = plt.subplots()
            for p,sd,stim,id in zip(psth_objaverse, sd_objaverse,objaverse_stim, n_trials_objaverse):
                ax.plot(bincents,p,label = stim + '_' + str(n_trials_objaverse[id]))
                ax.fill_between(bincents, p-sd, p+sd,
                    alpha=0.2)

            ax.set_xlabel('Time (s)', size= 15)
            ax.set_ylabel('spikes/s', size = 15)

            ax.legend(bbox_to_anchor = (1,1))

            ax.set_title('ch{:0>3d}'.format(n_chan_new))
            ymin, ymax  = ax.get_ylim()
            ax.fill_between(np.linspace(0,stim_dur),ymin,ymax,color = 'y', alpha = 0.2)
            filename = 'ch{:0>3d}_objaverse_psth'.format(n_chan_new) + '.png'

            plt.savefig(plot_save_out_path / 'objaverse' / filename, bbox_inches = 'tight')

            plt.close()


            # face vs. object 

            cmap= plt.cm.get_cmap('tab10')

            face_color = cmap(0)
            nonface_color = cmap(1)

            fig, ax = plt.subplots()
            n_face_trials = np.nansum(n_trials_face)

            face_mean_psth = np.nanmean(psth_face*100,axis = 0)
            sd_psth = np.nanstd(psth_face*100,axis = 0) / np.sqrt(n_face_trials)
            plt.plot(bincents,face_mean_psth,label = 'face',color =face_color)
            ax.fill_between(bincents, face_mean_psth-sd_psth, 
                            face_mean_psth+sd_psth,alpha=0.2)


            n_nonface_trials = np.nansum(n_trials_nonface)
            nonface_mean_psth = np.nanmean(psth_nonface*100,axis = 0)
            sd_psth = np.nanstd(psth_nonface*100,axis = 0) / np.sqrt(n_nonface_trials)


            plt.plot(bincents,nonface_mean_psth,label = 'objects',color =nonface_color)
            ax.fill_between(bincents, nonface_mean_psth-sd_psth, 
                            nonface_mean_psth+sd_psth,alpha=0.2)

            ymin, ymax  = ax.get_ylim()
            ax.fill_between(np.linspace(0,stim_dur),ymin,ymax,color = 'y', alpha = 0.5)


            ax.set_xlabel('Time (s)', size= 15)
            ax.set_ylabel('spikes/s', size = 15)

            ax.legend(bbox_to_anchor = (1,1))
            ax.set_title('ch{:0>3d}'.format(n_chan_new) + f'\n # face trials = {n_face_trials}  # non-face trials ={n_nonface_trials}')

            filename = 'ch{:0>3d}_objaverse_face_psth'.format(n_chan_new)
    

            plt.savefig(plot_save_out_path / 'objaverse'/ filename, bbox_inches= 'tight')

            plt.close()

            FSI = dict()
            psth_sum = face_mean_psth + nonface_mean_psth
            psth_diff = face_mean_psth - nonface_mean_psth 

            FSI_all = psth_diff/psth_sum
            FSI['all'] = FSI_all

            # objaverse face psth using just oldman
            if len(psth_oldman) > 0:

                bl_trial = np.nanmean(psth_bl_oldman)
                face_mean_psth = np.nanmean(psth_oldman,axis = 0)
                
                bl_trial = np.nanmean(psth_bl_nonface)
                nonface_mean_psth = np.nanmean(psth_nonface,axis = 0)

                psth_sum = face_mean_psth + nonface_mean_psth
                psth_diff = face_mean_psth - nonface_mean_psth 

                print(psth_diff, psth_sum)

                FSI_oldman = psth_diff/psth_sum
                
            else:
                FSI_oldman =  np.nan

            FSI['oldman'] = FSI_oldman

            # objaverse face psth using just neptune

            if len(psth_neptune) > 0:
                bl_trial = np.nanmean(psth_bl_neptune)
                face_mean_psth = np.nanmean(psth_neptune,axis = 0)

                bl_trial = np.nanmean(psth_bl_nonface)
                nonface_mean_psth = np.nanmean(psth_nonface,axis = 0)
                psth_sum = face_mean_psth + nonface_mean_psth
                psth_diff = face_mean_psth - nonface_mean_psth 

                #psth_sum = np.nanmean(psth_neptune,axis = 0) + np.nanmean(max_nonface,axis = 0)
                #psth_diff = np.nanmean(psth_neptune,axis = 0) - np.nanmean(max_nonface,axis = 0)


                FSI_neptune = psth_diff/psth_sum
                #FSI_neptune[psth_sum < np.nanmean(np.vstack((psth_bl_neptune, psth_bl_nonface)))] = 0 
            else:
                FSI_neptune = np.nan

            FSI['neptune'] = FSI_neptune

            # if eliasneptune stimuli were run

            id_elias = 'face/eliasapplefacepotatoheadnormalhires.glb_sz_0.7652740736666667_posX_0_posY_0_posZ_0_rotX_0_rotY_0_rotZ_0_light00_posX_0_posY_1_posZ_1_camera00_posX_0_posY_0_posZ_10_targetX_0_targetY_0_targetZ_0'
            if len(np.where(stim_ids_all == id_elias )[0]) ==1:
                print('elias found')
                psth_elias = psth[id_elias]
                n_trials_elias = psth_meta[id_elias]['n_trials']
                psth_bl_elias = psth_bl[id_elias]
                bl_trial = np.nanmean(psth_bl_elias)

                face_mean_psth = np.nanmean(psth_elias,axis = 0)# - bl_trial
                
                bl_trial = np.nanmean(psth_bl_nonface)
                nonface_mean_psth = np.nanmean(psth_nonface,axis = 0) #- bl_trial

                psth_sum = face_mean_psth + nonface_mean_psth
                psth_diff = face_mean_psth - nonface_mean_psth 

                #psth_sum = np.nanmean(psth_oldman,axis = 0) + np.nanmean(max_nonface,axis = 0)
                #psth_diff = np.nanmean(psth_oldman,axis = 0) - np.nanmean(max_nonface,axis = 0)

                print(psth_diff, psth_sum)

                FSI_elias = psth_diff/psth_sum
                FSI_elias[psth_sum < np.nanmean(np.vstack((psth_bl_elias, psth_bl_nonface)))] = 0 
            else:
                FSI_elias = np.nan

            FSI['elias'] = FSI_elias
            
            id_neptune_big = 'face/neptuneapplefacepotatoheadnormalhires.glb_sz_0.7652740736666667_posX_0_posY_0_posZ_0_rotX_0_rotY_0_rotZ_0_light00_posX_0_posY_1_posZ_1_camera00_posX_0_posY_0_posZ_10_targetX_0_targetY_0_targetZ_0'
            if len(np.where(stim_ids_all == id_neptune_big )[0]) ==1:
                print('neptune_big found')
                psth_neptune_big = psth[id_neptune_big]
                n_trials_neptune_big = psth_meta[id_neptune_big]['n_trials']
                psth_bl_neptune_big = psth_bl[id_neptune_big]
                bl_trial = np.nanmean(psth_bl_neptune_big)

                face_mean_psth = np.nanmean(psth_neptune_big,axis = 0)#- bl_trial

                bl_trial = np.nanmean(psth_bl_nonface)
                nonface_mean_psth = np.nanmean(psth_nonface,axis = 0)#- bl_trial
                
                psth_sum = face_mean_psth + nonface_mean_psth
                psth_diff = face_mean_psth - nonface_mean_psth

                #psth_sum = np.nanmean(psth_neptune,axis = 0) + np.nanmean(max_nonface,axis = 0)
                #psth_diff = np.nanmean(psth_neptune,axis = 0) - np.nanmean(max_nonface,axis = 0)

                FSI_neptune_big = psth_diff/psth_sum
                FSI_neptune_big[psth_sum < np.nanmean(np.vstack((psth_bl_neptune_big, psth_bl_nonface)))] = 0 

            else:
                FSI_neptune_big = np.nan

            FSI['neptune_big'] = FSI_neptune_big


            # if sophie stimuli were run 
            id_sophie = 'finished/sophie_Nose_0_Mouth_0_EyeL_22_5_EyeR_-22_5.glb_sz_0.6_posX_0_posY_0_posZ_0_rotX_0_rotY_0_rotZ_0_light00_posX_0_posY_1_posZ_1_camera00_posX_0_posY_0_posZ_10_targetX_0_targetY_0_targetZ_0'
            if len(np.where(stim_ids_all == id_sophie )[0]) ==1:
                print('sophie found')

                stim_keys = list(psth.keys())
                ind = np.where([id_sophie in k for k in stim_keys])[0][0]
                
                psth_sophie = psth[stim_keys[ind]]
                n_trials_sophie = psth_meta[stim_keys[ind]]['n_trials']
                psth_bl_sophie = psth_bl[stim_keys[ind]]

                bl_trial = np.nanmean(psth_bl_sophie)

                face_mean_psth = np.nanmean(psth_sophie,axis = 0) #- bl_trial

                bl_trial = np.nanmean(psth_bl_nonface)
                nonface_mean_psth = np.nanmean(psth_nonface,axis = 0) #- bl_trial
                
                psth_sum = face_mean_psth + nonface_mean_psth
                psth_diff = face_mean_psth - nonface_mean_psth

                #psth_sum = np.nanmean(psth_neptune,axis = 0) + np.nanmean(max_nonface,axis = 0)
                #psth_diff = np.nanmean(psth_neptune,axis = 0) - np.nanmean(max_nonface,axis = 0)

                FSI_sophie = psth_diff/psth_sum
                FSI_sophie[psth_sum < np.nanmean(np.vstack((psth_bl_sophie, psth_bl_nonface)))] = 0 

            else:
                FSI_sophie = np.nan

            FSI['sophie'] = FSI_sophie

            filename = 'ch{:0>3d}_FSI'.format(n_chan)
            pickle.dump(FSI, open(save_out_path /filename,'wb'), protocol = 2)

            print('done')
            
if __name__ == '__main__':
    n_chan = int(sys.argv[1])
    monkey = sys.argv[2]
    date = sys.argv[3]
    get_psth_objaverse(n_chan, monkey, date)