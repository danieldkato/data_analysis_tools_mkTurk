from natsort import os_sorted
import pickle
import pathlib as Path
import os
import sys
import numpy as np
from utils_ephys import * 
from utils_meta import * 

engram_path = Path.Path('/mnt/smb/locker/issa-locker')

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

n_chan = int(sys.argv[1])
monkey = sys.argv[2]
date = sys.argv[3]

data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

for n, (data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
    MUA_dir = data_path / Path.Path('MUA_4SD')
    print(data_path)
    if not data_path.exists():
        print('data path doesn''t exist')
        if n == len(data_path_list)-1:
            sys.exit()
        else:
            continue

    if not save_out_path.exists():
        print('save out path doesn''t exist')
        if n == len(data_path_list) - 1:
            sys.exit()
        else:
            continue

    try: 
        stim_info_path= os_sorted(save_out_path.glob('stim_info_sess'))[0]
        print(stim_info_path)
        print('\nstim info sess found: ' + str(stim_info_path.exists()))
    except:
        print('no stim_info_sess found')
        if n == len(data_path_list) -1 :
            sys.exit()
        else:
            continue

    try:
        data_dict_path = os_sorted(save_out_path.glob('data_dict_' + monkey + '*'))[0]
    except:
        print('no data dict found')
        if n == len(data_path_list) -1 :
            sys.exit()
        else:
            continue

    if not save_out_path.exists():
        os.makedirs(save_out_path,exist_ok= True)
    
    if not plot_save_out_path.exists():
        os.makedirs(plot_save_out_path,exist_ok= True)
    ##############################################################################################################################################


    # get baseline
    ch_psth_bl, ch_psth_bl_meta,ch_psth_bl_stim = get_data_bl(n_chan, MUA_dir, data_dict_path,stim_info_path)

    pickle.dump(ch_psth_bl, open(save_out_path /'ch{:0>3d}_psth_bl'.format(n_chan) ,'wb'), protocol = 2)
    pickle.dump(ch_psth_bl_stim, open(save_out_path /'ch{:0>3d}_psth_bl_stim'.format(n_chan) ,'wb'), protocol = 2)
    # get spike times, peak amplitude, waveform, and psth per stimulus 

    ch_pk_stim, ch_wf_stim, ch_st_stim, ch_sl_stim, ch_ind_stim,ch_psth_stim,ch_psth_stim_meta = \
        get_data_bystim(n_chan, MUA_dir, stim_info_path, binwidth_psth= 0.01)

    pickle.dump(ch_pk_stim, open(save_out_path / 'ch{:0>3d}_pk_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_wf_stim, open(save_out_path / 'ch{:0>3d}_wf_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_st_stim, open(save_out_path / 'ch{:0>3d}_st_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_sl_stim, open(save_out_path / 'ch{:0>3d}_sl_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_ind_stim, open(save_out_path / 'ch{:0>3d}_ind_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_psth_stim, open(save_out_path /'ch{:0>3d}_psth_stim'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(ch_psth_stim_meta, open(save_out_path /'ch{:0>3d}_psth_stim_meta'.format(n_chan),'wb'), protocol = 2)

    # get mean psth per stimulus
    # get the longest length 
    psth_len= 0
    for stim in ch_psth_stim:
        psth_len = np.nanmax((psth_len,ch_psth_stim[stim].shape[1]))

    mean_psth = np.nan * np.ones((len(ch_psth_stim), psth_len))
    for n,stim in enumerate(ch_psth_stim):
        mean_psth[n,0:ch_psth_stim[stim].shape[1]] = np.nanmean(ch_psth_stim[stim],axis = 0)

    np.save(save_out_path / 'ch{:0>3d}_mean_psth.npy'.format(n_chan),mean_psth)

    # get psth per scenefile

    stim_list_sess = list(ch_psth_stim_meta.keys())

    all_scenefile = []
    for stim in stim_list_sess:
        all_scenefile.extend(ch_psth_stim_meta[stim]['scenefile'])

    unique_scenefile = np.unique(all_scenefile)

    psth_byscenefile = dict.fromkeys(unique_scenefile) 
    psth_byscenefile_meta = dict.fromkeys(unique_scenefile)

    for s in unique_scenefile:
        psth_new = []
        psth_byscenefile_meta[s] = dict()
        psth_byscenefile_meta[s]['stim_ids'] = []
        psth_byscenefile_meta[s]['ind'] = []
        psth_byscenefile_meta[s]['n_trials_stim'] = []
        n_trials = 0
        for stim_ind,stim in enumerate(stim_list_sess):
            if s in ch_psth_stim_meta[stim]['scenefile']:
                psth_new.append(ch_psth_stim[stim])
                psth_byscenefile_meta[s]['ind'].append(stim_ind)
                psth_byscenefile_meta[s]['stim_ids'].append(stim)
                t_before = ch_psth_stim_meta[stim]['t_before']
                t_after = ch_psth_stim_meta[stim]['t_after']
                psth_bins = ch_psth_stim_meta[stim]['psth_bins']
                binwidth = ch_psth_stim_meta[stim]['binwidth']
                n_trials += ch_psth_stim_meta[stim]['n_trials']
                psth_byscenefile_meta[s]['n_trials_stim'].append(ch_psth_stim_meta[stim]['n_trials'])
                stim_dur = ch_psth_stim_meta[stim]['stim_dur']
        psth_new = np.vstack(psth_new)
        psth_byscenefile[s]= psth_new
        psth_byscenefile_meta[s]['stim_dur'] = stim_dur
        psth_byscenefile_meta[s]['t_before'] = t_before
        psth_byscenefile_meta[s]['t_after'] = t_after
        psth_byscenefile_meta[s]['psth_bins'] = psth_bins
        psth_byscenefile_meta[s]['binwidth'] = binwidth
        psth_byscenefile_meta[s]['n_trials'] = n_trials

    pickle.dump(psth_byscenefile, open(save_out_path /'ch{:0>3d}_psth_scenefile'.format(n_chan),'wb'), protocol = 2)
    pickle.dump(psth_byscenefile_meta, open(save_out_path /'ch{:0>3d}_psth_scenefile_meta'.format(n_chan),'wb'), protocol = 2)

    # plots      
    # psth_scenefile =pickle.load(open(save_out_path /'ch{:0>3d}_psth_scenefile'.format(n_chan),'rb'))
    # psth_scenefile_meta= pickle.load(open(save_out_path /'ch{:0>3d}_psth_scenefile_meta'.format(n_chan),'rb'))

    gen_psth_byscenefile(n_chan,plot_save_out_path, psth_byscenefile, psth_byscenefile_meta)
