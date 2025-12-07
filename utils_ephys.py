import numpy as np
import math
from pathlib import Path
import pickle
import os
import sys
from natsort import os_sorted
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils_code import Config

def load_data(n_chan,MUA_dir):
    # loads spike times, peak values of detected spikes, waveform 
    ts_file = next(MUA_dir.glob('ch{:0>3d}_ts.npy'.format(n_chan)))
    pk_file =next(MUA_dir.glob('ch{:0>3d}_pks.npy'.format(n_chan)))

    if len(list(MUA_dir.glob('ch{:0>3d}_wfs.npy'.format(n_chan)))) >0:
        wf_file =next(MUA_dir.glob('ch{:0>3d}_wfs.npy'.format(n_chan)))
        wf  = np.load(wf_file)
    else: 
        wf_file = []
        wf = []
    try:
        sl_file = next(MUA_dir.glob('ch{:0>3d}_sign_label.npy'.format(n_chan)))
    except:
        sl_file = next(MUA_dir.glob('ch{:0>3d}_sls.npy'.format(n_chan)))

    st = np.load(ts_file)
        
    pk = np.load(pk_file)
    sl = np.load(sl_file)

    if st.shape[0] > 1: # spike times stores timestamps both negative and positive peaks in a spike event
        if len(np.where(sl==1)[0]) != 0:
            st_neg = st[np.where(sl==1)[0],np.argmin(pk[np.where(sl==1)[0],0:2], axis = 1)[0]]
        else:
            st_neg = np.array([])
        
        if len(np.where(sl==0)[0]) !=0:
            st_pos = st[np.where(sl==0)[0],np.argmax(pk[np.where(sl==0)[0],0:2], axis = 1)[0]]
        else:
            st_pos = np.array([])

        st = np.sort(np.hstack((st_neg,st_pos)))

    #assert len(st) == len(wf) == len(pk) == len(sl), 'length of files does not match'
    assert len(st) == len(pk) == len(sl), 'length of files does not match'
    return st, wf, pk, sl

def get_data_bystim(n_chan, MUA_dir, stim_info_path, t_before = 0.1, t_after = 0.1, binwidth_psth = 0.01):
    
    # all data are organized in the order of stimuli in stim_info_sess

    st, wf, pk, sl = load_data(n_chan, MUA_dir)

    stim_info_sess = pickle.load(open(stim_info_path,'rb'))

    # organize spike data by stimulus 

    ch_pk_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_wf_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_st_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_sl_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_ind_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_psth_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_psth_stim_meta = dict.fromkeys(list(stim_info_sess.keys()))

    # psth meta

    for stim in stim_info_sess:
        ch_psth_stim_meta[stim] = dict()
        iti_dur = np.nanmax(stim_info_sess[stim]['iti_dur'])
        stim_dur = np.nanmax(stim_info_sess[stim]['dur'])

        n_bins = int(np.ceil((stim_dur + t_before + t_after)/binwidth_psth))
        psth_bins = np.linspace(-t_before, stim_dur + t_after, n_bins+1)

        ch_psth_stim_meta[stim]['t_before'] = t_before
        ch_psth_stim_meta[stim]['t_after'] = t_after
        ch_psth_stim_meta[stim]['stim_dur'] = stim_dur
        ch_psth_stim_meta[stim]['iti_dur'] = iti_dur
        ch_psth_stim_meta[stim]['binwidth'] = binwidth_psth
        ch_psth_stim_meta[stim]['n_bins'] = n_bins
        ch_psth_stim_meta[stim]['psth_bins'] = psth_bins
        ch_psth_stim_meta[stim]['scenefile'] = stim_info_sess[stim]['scenefile']

        ch_pk_stim[stim] = []
        ch_wf_stim[stim] = []
        ch_st_stim[stim] = []
        ch_sl_stim[stim] = []
        ch_ind_stim[stim] = []
        ch_psth_stim[stim] = np.empty((len(stim_info_sess[stim]['t_on']),len(psth_bins)-1))
        n_trials = 0
        for i, t_on in enumerate(stim_info_sess[stim]['t_on']):
            # 100 ms before the onset of stimulus
            t_start = t_on - t_before
            t_end = t_on + stim_dur +  t_after
     
            if st.shape[0] > 1: 
                ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
            else:
                ind = np.where((st >= t_start) & (st <= t_end))[0]
            
            ch_pk_stim[stim].append(pk[ind])
            if len(wf) >0:
                ch_wf_stim[stim].append(wf[ind])
            ch_st_stim[stim].append(st[ind])
            ch_sl_stim[stim].append(sl[ind])
            ch_ind_stim[stim].append(ind)
            
            ch_psth_stim[stim][i,:], _ =  np.histogram(st[ind]-t_on, psth_bins)
            n_trials +=1

        ch_psth_stim_meta[stim]['n_trials'] = n_trials

    return ch_pk_stim, ch_wf_stim, ch_st_stim, ch_sl_stim, ch_ind_stim,ch_psth_stim,ch_psth_stim_meta

def get_data_bysc(n_chan, MUA_dir, data_dict_path,save_out_path, t_before = 0.1, t_after = 0.1, binwidth_psth = 0.01):
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

    assert len(st) == len(wf) == len(pk) == len(sl)
    if st.shape[0] > 1: # spike times stores timestamps both negative and positive peaks in a spike event
        st_neg = st[np.where(sl==1)[0],np.argmin(pk[np.where(sl==1)[0],0:2], axis = 1)[0]]
        st_pos = st[np.where(sl==0)[0],np.argmax(pk[np.where(sl==0)[0],0:2], axis = 1)[0]]

        st = np.sort(np.hstack((st_neg,st_pos)))

    # 1) organize spike data by sample command by block (behavior file)
    
    for d_path in data_dict_path:
        data_dict = pickle.load(open(d_path,'rb'))
        # get behavior file name
        behav_file = Path(d_path).stem.split('data_dict_')[1]
        scenefile = data_dict[0]['scenefile']
        print(behav_file)

        trig_on= []
        trig_off = []
        for n_stim in data_dict:
            if n_stim == 0:
                trig_on.append(data_dict[n_stim]['imec_trig_on'])
                trig_off.append(data_dict[n_stim]['imec_trig_off'])
            if data_dict[n_stim]['imec_trig_on'] not in trig_on and not math.isnan(data_dict[n_stim]['imec_trig_on']):
                trig_on.append(data_dict[n_stim]['imec_trig_on'])
            if data_dict[n_stim]['imec_trig_off'] not in trig_off and not math.isnan(data_dict[n_stim]['imec_trig_off']):
                trig_off.append(data_dict[n_stim]['imec_trig_off'])

        assert len(trig_on) == len(trig_off)
        trig_dur = np.array(trig_off) - np.array(trig_on)

        max_sc_dur = round(np.nanmax(trig_dur))
        n_bins = int(np.ceil((max_sc_dur+ t_before + t_after)/binwidth_psth))
        psth_bins = np.linspace(-t_before, max_sc_dur + t_after, n_bins+1)
        
        n_scs = len(trig_on)

        n_spikes = []
        ch_st_sc = dict.fromkeys(range(n_scs))
        ch_ind_sc = dict.fromkeys(range(n_scs))
        ch_psth_sc_meta = dict.fromkeys(range(n_scs))
        ch_psth_sc = np.empty((n_scs,len(psth_bins)-1))
        n_trials = 0 
        for n, (t_on, t_off) in enumerate(zip(trig_on,trig_off)):
            ch_psth_sc_meta[n] = dict()
            sc_dur = t_off - t_on

            # get spike times from 100ms before the sample command onset and 100ms after
            t_start = t_on - t_before
            t_end = t_off+ t_after
            if st.shape[0] > 1: 
                ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
            else:
                ind = np.where((st >= t_start) & (st <= t_end))[0]

            ch_st_sc[n] = st[ind]
            ch_ind_sc[n] = ind
            ch_psth_sc_meta[n]['t_on'] = t_on
            ch_psth_sc_meta[n]['t_off'] = t_off
            ch_psth_sc_meta[n]['sc_dur'] = sc_dur

            ch_psth_sc[n], _ = np.histogram(st[ind]-t_on, psth_bins)
            
            n_spikes.append(len(ind))
        
        ch_psth_sc_meta['n_spikes_tot'] = len(st) # number of total detected events
        ch_psth_sc_meta['n_spikes_sc'] = np.nansum(n_spikes)
        ch_psth_sc_meta['n_spikes_mean'] = np.nanmean(n_spikes)
        ch_psth_sc_meta['max_sc_dur'] = max_sc_dur
        ch_psth_sc_meta['t_before'] = t_before
        ch_psth_sc_meta['t_after'] = t_after
        ch_psth_sc_meta['binwidth'] = binwidth_psth
        ch_psth_sc_meta['n_bins'] = n_bins
        ch_psth_sc_meta['n_trials'] = n_scs
        ch_psth_sc_meta['psth_bins'] = psth_bins
        ch_psth_sc_meta['scenefile'] = scenefile
        ch_psth_sc_meta['behav_file'] = behav_file

        filename ='ch{:0>3d}_st_sc'.format(n_chan) + f'_{behav_file}'
        print(filename)
        pickle.dump(ch_st_sc, open(save_out_path / filename,'wb'), protocol = 2)
        
        filename ='ch{:0>3d}_ind_sc'.format(n_chan) + f'_{behav_file}'
        pickle.dump(ch_ind_sc, open(save_out_path / filename,'wb'), protocol = 2)

        filename ='ch{:0>3d}_psth_sc'.format(n_chan) + f'_{behav_file}'
        pickle.dump(ch_psth_sc, open(save_out_path /filename,'wb'), protocol = 2)
        filename ='ch{:0>3d}_psth_sc_meta'.format(n_chan) + f'_{behav_file}'
        pickle.dump(ch_psth_sc_meta, open(save_out_path /filename,'wb'), protocol = 2)

def get_data_bl(n_chan, MUA_dir, data_dict_path,stim_info_path, t_before = 0.2, t_after = 0, binwidth_psth = 0.01):

    st, wf, pk, sl = load_data(n_chan, MUA_dir)
    data_dict = pickle.load(open(data_dict_path,'rb'))

    n_stims = len(data_dict)
    # get baseline for each stimulus 
    # -200 to 0 from onset of a trial
    n_bins = int(np.ceil(t_before/binwidth_psth))
    psth_bins = np.linspace(-t_before, t_after, n_bins+1)

    ch_st_bl = dict.fromkeys(range(n_stims))
    ch_ind_bl = dict.fromkeys(range(n_stims))
    ch_psth_bl_meta = dict.fromkeys(range(n_stims))
    ch_psth_bl = np.nan * np.ones((n_stims,len(psth_bins)-1))

    for n in data_dict:
        t_on = data_dict[n]['imec_trig_on']
        t_off = data_dict[n]['imec_trig_off']
        sc_dur = t_off - t_on
        t_start = t_on - t_before
        t_end = t_on + t_after
        if st.shape[0] > 1: 
            ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
        else:
            ind = np.where((st >= t_start) & (st <= t_end))[0]

        ch_st_bl[n] = st[ind]
        ch_ind_bl[n] = ind

        if data_dict[n]['t_mk'] != -1:
            ch_psth_bl[n], _ = np.histogram(st[ind]-t_on, psth_bins)
    
    ch_psth_bl_meta['t_before'] = t_before
    ch_psth_bl_meta['t_end'] = t_after
    ch_psth_bl_meta['binwidth'] = binwidth_psth

    # organize by stimulus
    stim_info_sess = pickle.load(open(stim_info_path,'rb'))
    
    ch_psth_bl_stim = dict.fromkeys(list(stim_info_sess.keys()))
    for stim in stim_info_sess:
        ch_psth_bl_stim[stim] = ch_psth_bl[stim_info_sess[stim]['stim_ind']]

    return ch_psth_bl, ch_psth_bl_meta,ch_psth_bl_stim

def get_data_bysc_all(n_chan, MUA_dir, data_dict_path,save_out_path, t_before = 0.1, t_after = 0.1, binwidth_psth = 0.01):
    ts_file = next(MUA_dir.glob('ch{:0>3d}_ts.npy'.format(n_chan)))
    pk_file =next(MUA_dir.glob('ch{:0>3d}_pks.npy'.format(n_chan)))
    wf_file =next(MUA_dir.glob('ch{:0>3d}_wfs.npy'.format(n_chan)))
    sl_file = next(MUA_dir.glob('ch{:0>3d}_sign_label.npy'.format(n_chan)))

    st = np.load(ts_file)
    wf  = np.load(wf_file)
    pk = np.load(pk_file)
    sl = np.load(sl_file)

    assert len(st) == len(wf) == len(pk) == len(sl)
    if st.shape[0] > 1: # spike times stores timestamps both negative and positive peaks in a spike event
        st_neg = st[np.where(sl==1)[0],np.argmin(pk[np.where(sl==1)[0],0:2], axis = 1)[0]]
        st_pos = st[np.where(sl==0)[0],np.argmax(pk[np.where(sl==0)[0],0:2], axis = 1)[0]]

        st = np.sort(np.hstack((st_neg,st_pos)))

    # 1) organize spike data by sample command by block (behavior file)


    data_dict = pickle.load(open(data_dict_path,'rb'))
    # get behavior file name
    #behav_file = Path(d_path).stem.split('data_dict_')[1]
   #scenefile = data_dict[0]['scenefile']
    #print(behav_file)

    trig_on= []
    trig_off = []
    for n_stim in data_dict:
        if n_stim == 0:
            trig_on.append(data_dict[n_stim]['imec_trig_on'])
            trig_off.append(data_dict[n_stim]['imec_trig_off'])
        if data_dict[n_stim]['imec_trig_on'] not in trig_on and not math.isnan(data_dict[n_stim]['imec_trig_on']):
            trig_on.append(data_dict[n_stim]['imec_trig_on'])
        if data_dict[n_stim]['imec_trig_off'] not in trig_off and not math.isnan(data_dict[n_stim]['imec_trig_off']):
            trig_off.append(data_dict[n_stim]['imec_trig_off'])

    assert len(trig_on) == len(trig_off)
    trig_dur = np.array(trig_off) - np.array(trig_on)

    max_sc_dur = round(np.nanmax(trig_dur))
    print(max_sc_dur)
    n_bins = int(np.ceil((max_sc_dur+ t_before + t_after)/binwidth_psth))
    psth_bins = np.linspace(-t_before, max_sc_dur + t_after, n_bins+1)
    
    n_scs = len(trig_on)

    n_spikes = []
    ch_st_sc = dict.fromkeys(range(n_scs))
    ch_ind_sc = dict.fromkeys(range(n_scs))
    ch_psth_sc_meta = dict.fromkeys(range(n_scs))
    ch_psth_sc = np.empty((n_scs,len(psth_bins)-1))

    for n, (t_on, t_off) in enumerate(zip(trig_on,trig_off)):
        ch_psth_sc_meta[n] = dict()
        sc_dur = t_off - t_on

        # get spike times from 100ms before the sample command onset and 100ms after
        t_start = t_on - t_before
        t_end = t_off+ t_after
        if st.shape[0] > 1: 
            ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
        else:
            ind = np.where((st >= t_start) & (st <= t_end))[0]

        ch_st_sc[n] = st[ind]
        ch_ind_sc[n] = ind
        ch_psth_sc_meta[n]['t_on'] = t_on
        ch_psth_sc_meta[n]['t_off'] = t_off
        ch_psth_sc_meta[n]['sc_dur'] = sc_dur

        ch_psth_sc[n], _ = np.histogram(st[ind]-t_on, psth_bins)
        
        n_spikes.append(len(ind))
    
    ch_psth_sc_meta['n_spikes_tot'] = len(st) # number of total detected events
    ch_psth_sc_meta['n_spikes_sc'] = np.nansum(n_spikes)
    ch_psth_sc_meta['n_spikes_mean'] = np.nanmean(n_spikes)
    ch_psth_sc_meta['max_sc_dur'] = max_sc_dur
    ch_psth_sc_meta['t_before'] = t_before
    ch_psth_sc_meta['t_after'] = t_after
    ch_psth_sc_meta['binwidth'] = binwidth_psth
    ch_psth_sc_meta['n_bins'] = n_bins
    ch_psth_sc_meta['n_trials'] = n_scs
    ch_psth_sc_meta['psth_bins'] = psth_bins

    #filename ='ch{:0>3d}_st_sc'.format(n_chan)
    #print(filename)
    #pickle.dump(ch_st_sc, open(save_out_path / filename,'wb'), protocol = 2)
    #filename ='ch{:0>3d}_ind_sc'.format(n_chan) 
    #pickle.dump(ch_ind_sc, open(save_out_path / filename,'wb'), protocol = 2)
    filename ='ch{:0>3d}_psth_sc'.format(n_chan) 
    pickle.dump(ch_psth_sc, open(save_out_path /filename,'wb'), protocol = 2)
    filename ='ch{:0>3d}_psth_sc_meta'.format(n_chan) 
    pickle.dump(ch_psth_sc_meta, open(save_out_path /filename,'wb'), protocol = 2)


# concatenate across channels
def get_psth_sc_allchan(save_out_path):
    psth_meta = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_meta'.format(0)))[0],'rb'))
    n_bins = psth_meta['n_bins']
    n_chans = 384
    # Concatenate all channels per scenefile 
    psth_all = np.empty((n_chans,n_bins))
    
    for n in range(n_chans):
        psth = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc'.format(n)))[0],'rb'))
        psth_mean = np.nanmean(psth, axis = 0)
        psth_all[n,:] = psth_mean

    pickle.dump(psth_all, open(save_out_path / 'psth_sc','wb'),protocol = 2)


def get_psth_sc_bybehav_allchan(save_out_path):
    psth_meta_files = os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_meta_[0-9]*'.format(0)))

    behav_files = []
    for m in psth_meta_files:
        behav_files.append(m.as_posix().split('meta_')[1])

    for b in behav_files:
        n_chans = 384
        psth_all = []
        for n in range(n_chans):
            psth = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_'.format(n) + b))[0],'rb'))
            psth_all.append(np.nanmean(psth, axis =0))

        psth_all = np.vstack(psth_all)

        filename ='psth_sc'+f'_{b}'
        pickle.dump(psth_all, open(save_out_path /filename,'wb'), protocol = 2)

def get_psth_byscenefile_allchans(save_out_path):
    psth = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_scenefile'.format(0)))[0],'rb'))
    scenefile_unique = list(psth.keys())
    print(scenefile_unique)

    n_chans = 384
    # Concatenate all channels per scenefile 
    for s in scenefile_unique:
        psth_all = []
        for n in range(n_chans):
            print('save_out_path = {}'.format(save_out_path))
            psth = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_scenefile'.format(n)))[0],'rb'))
            if type(psth[s]) != dict:
                psth_mean = np.nanmean(psth[s], axis = 0)
                psth_all.append(psth_mean)

        psth_new = np.vstack(psth_all)
        pickle.dump(psth_new, open(save_out_path / ('psth_' + Path(s).stem),'wb'),protocol = 2)
        
def load_kilosort(n_chan, kilosort_dir):

    st = np.load(kilosort_dir / 'good_unit'/ 'clu_{:0>3d}_st.npy'.format(n_chan))

    return st

def get_data_bystim_kilosort(n_chan, kilosort_dir, stim_info_path, t_before = 0.1, t_after = 0.1, binwidth_psth = 0.01):
    
    # all data are organized in the order of stimuli in stim_info_sess

    st = load_kilosort(n_chan,kilosort_dir)

    stim_info_sess = pickle.load(open(stim_info_path,'rb'))

    # organize spike data by stimulus 
    ch_psth_stim = dict.fromkeys(list(stim_info_sess.keys()))
    ch_psth_stim_meta = dict.fromkeys(list(stim_info_sess.keys()))

    # psth meta

    for stim in stim_info_sess:
        ch_psth_stim_meta[stim] = dict()
        iti_dur = np.nanmax(stim_info_sess[stim]['iti_dur'])
        stim_dur = np.nanmax(stim_info_sess[stim]['dur'])

        n_bins = int(np.ceil((stim_dur + t_before + t_after)/binwidth_psth))
        psth_bins = np.linspace(-t_before, stim_dur + t_after, n_bins+1)

        ch_psth_stim_meta[stim]['t_before'] = t_before
        ch_psth_stim_meta[stim]['t_after'] = t_after
        ch_psth_stim_meta[stim]['stim_dur'] = stim_dur
        ch_psth_stim_meta[stim]['iti_dur'] = iti_dur
        ch_psth_stim_meta[stim]['binwidth'] = binwidth_psth
        ch_psth_stim_meta[stim]['n_bins'] = n_bins
        ch_psth_stim_meta[stim]['psth_bins'] = psth_bins
        ch_psth_stim_meta[stim]['scenefile'] = stim_info_sess[stim]['scenefile']

        ch_psth_stim[stim] = np.empty((len(stim_info_sess[stim]['t_on']),len(psth_bins)-1))
        n_trials = 0
        for i, t_on in enumerate(stim_info_sess[stim]['t_on']):
            # 100 ms before the onset of stimulus
            t_start = t_on - t_before
            t_end = t_on + stim_dur +  t_after
     
            if st.shape[0] > 1: 
                ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
            else:
                ind = np.where((st >= t_start) & (st <= t_end))[0]
            
            ch_psth_stim[stim][i,:], _ =  np.histogram(st[ind]-t_on, psth_bins)
            n_trials +=1

        ch_psth_stim_meta[stim]['n_trials'] = n_trials

    return ch_psth_stim,ch_psth_stim_meta

def get_data_bl_kilosort(n_chan, kilosort_dir,data_dict_path,stim_info_path, t_before = 0.2, t_after = 0, binwidth_psth = 0.01):

    st = load_kilosort(n_chan,kilosort_dir)

    stim_info_sess = pickle.load(open(stim_info_path,'rb'))

    data_dict = pickle.load(open(data_dict_path,'rb'))

    n_stims = len(data_dict)
    # get baseline for each stimulus 
    # -200 to 0 from onset of a trial
    n_bins = int(np.ceil(t_before/binwidth_psth))
    psth_bins = np.linspace(-t_before, t_after, n_bins+1)

    ch_st_bl = dict.fromkeys(range(n_stims))
    ch_ind_bl = dict.fromkeys(range(n_stims))
    ch_psth_bl_meta = dict.fromkeys(range(n_stims))
    ch_psth_bl = np.nan * np.ones((n_stims,len(psth_bins)-1))

    for n in data_dict:
        t_on = data_dict[n]['imec_trig_on']
        t_off = data_dict[n]['imec_trig_off']
        sc_dur = t_off - t_on
        t_start = t_on - t_before
        t_end = t_on + t_after
        if st.shape[0] > 1: 
            ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
        else:
            ind = np.where((st >= t_start) & (st <= t_end))[0]

        ch_st_bl[n] = st[ind]
        ch_ind_bl[n] = ind

        if data_dict[n]['t_mk'] != -1:
            ch_psth_bl[n], _ = np.histogram(st[ind]-t_on, psth_bins)
    
    ch_psth_bl_meta['t_before'] = t_before
    ch_psth_bl_meta['t_end'] = t_after
    ch_psth_bl_meta['binwidth'] = binwidth_psth

    # organize by stimulus
    stim_info_sess = pickle.load(open(stim_info_path,'rb'))
    
    ch_psth_bl_stim = dict.fromkeys(list(stim_info_sess.keys()))
    for stim in stim_info_sess:
        ch_psth_bl_stim[stim] = ch_psth_bl[stim_info_sess[stim]['stim_ind']]

    return ch_psth_bl, ch_psth_bl_meta,ch_psth_bl_stim


################################ plotting fuctions 
        
##### heatmaps
        
def gen_heatmap(psth,t_start,t_end, binwidth_psth, vmin = [], vmax =[]):
    fig, ax = plt.subplots(figsize= (15,20))
    if vmin == [] and vmax == []:
        ax = sns.heatmap(psth)
    else:
        ax = sns.heatmap(psth, vmin = vmin, vmax = vmax)
    plt.gca().invert_yaxis()

    ax2 = ax.twiny()

    xmin, xmax = ax.get_xlim()
    n_bins = int(xmax - xmin)
    
    psth_bins =np.arange(t_start, t_end, binwidth_psth)
    psth_binedge = np.hstack((psth_bins,t_end))

    xticks =np.array(list(range(n_bins+1)))
    ax.set_xticks(xticks)
    ax2.set_xticks(xticks)

    xticklabels = [''] * (n_bins+1)

    for n in range(int((n_bins+1)/5)):
        xticklabels[int(n*5)] = round(psth_binedge[int(n*5)],2)

    xticklabels[len(xticklabels)-1] = round(psth_binedge[len(psth_binedge)-1],2)
    ax.set_xticklabels(xticklabels)
    ax2.set_xticklabels(xticklabels)

    ax.set_xlabel('time(seconds) from stimulus onset',  size= 20, labelpad = 20)
    ax.set_ylabel('ch num', size = 20, labelpad = 15)

    # onset dashed line
    ymin, ymax = ax.get_ylim()

    ax.plot([0 - t_start/binwidth_psth, 0 - t_start/binwidth_psth],[ymin,ymax], 'k--')
    plt.rcParams['savefig.facecolor']='white'

    return fig, ax

def gen_heatmap_sc_bybehav(save_out_path, plot_save_out_path):
    psth_meta_files = os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_meta_[0-9]*'.format(0)))

    behav_files = []
    for m in psth_meta_files:
        behav_files.append(m.as_posix().split('meta_')[1])
    
    for b in behav_files:
        psth_meta = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_meta_'.format(0)+b))[0],'rb'))
        t_before = psth_meta['t_before']
        stim_dur = psth_meta['max_sc_dur']
        t_after = psth_meta['t_after']
        binwidth_psth = psth_meta['binwidth']

        psth = pickle.load(open(os_sorted(save_out_path.glob('psth_sc'+f'_{b}'))[0],'rb'))
        fig,ax = gen_heatmap(psth,-t_before,stim_dur + t_after, binwidth_psth)
        ax.set_title(b)

        filename = 'psth_sc_' + f'{b}'  + '.png'
        fig.savefig(plot_save_out_path/ filename,bbox_inches = 'tight')

def gen_heatmap_bysc(save_out_path, plot_save_out_path):
    psth_meta = pickle.load(open(os_sorted(save_out_path.glob('ch{:0>3d}_psth_sc_meta'.format(0)))[0],'rb'))
    t_before = psth_meta['t_before']
    stim_dur = psth_meta['max_sc_dur']
    t_after = psth_meta['t_after']
    binwidth_psth = psth_meta['binwidth']

    psth = pickle.load(open(os_sorted(save_out_path.glob('psth_sc'))[0],'rb'))
    fig, ax = gen_heatmap(psth,-t_before,stim_dur + t_after, binwidth_psth)

    filename = 'psth_sc' + '.png'

    fig.savefig(plot_save_out_path/ filename,bbox_inches = 'tight')

def gen_heatmap_byscenefile(save_out_path, plot_save_out_path,chanmap):
    # bin to plot dictates xlabels 
    psth_meta = pickle.load(open(os_sorted(save_out_path.glob('psth_scenefile_meta'))[0],'rb'))
    scenefile_unique = list(psth_meta.keys())
    print(scenefile_unique)
    
    for s in scenefile_unique:
        t_before = psth_meta[s]['t_before']
        stim_dur = psth_meta[s]['stim_dur']
        t_after = psth_meta[s]['t_after']
        binwidth_psth = psth_meta[s]['binwidth']
    
        psth = pickle.load(open(os_sorted(save_out_path.glob('psth_' + Path(s).stem))[0],'rb'))

        n_trials = psth_meta[s]['n_trials']
        assert psth.shape[0] == chanmap.shape[0]

        psth = psth[np.argsort(chanmap[:,1]),:]
        fig, ax = gen_heatmap(psth,-t_before,stim_dur + t_after, binwidth_psth)

        ax.set_title(Path(s).stem + '\n' + str(n_trials) + ' trials', size = 20, pad = 20)

        filename = Path(s).stem + '.png'
        fig.savefig(plot_save_out_path/ filename,bbox_inches = 'tight')
        plt.close()

        # z scored
        # remove the last 50ms. pump noise might be there
        # psth = psth[:,0:round((t_before + stim_dur-0.05)/binwidth_psth)]
        # mean_perchan = np.nanmean(psth,axis = 1)
        # std_perchan = np.nanstd(psth,axis = 1)
        # zscored = (psth- np.tile(mean_perchan[:,np.newaxis],[1,psth.shape[1]])) / np.tile(std_perchan[:,np.newaxis], [1,psth.shape[1]])

        # fig, ax = gen_heatmap(zscored,- t_before,stim_dur-0.05, binwidth_psth, -5, 5)

        # ax.set_title(Path(s).stem + '\n' + str(n_trials) + ' trials', size = 20, pad = 20)

        # filename = Path(s).stem + '_zscored.png'
        # fig.savefig(plot_save_out_path/ filename,bbox_inches = 'tight')

        # plt.close()

########### psth plots
        
def gen_psth_plots_bysc(n_chan,save_out_path, ch_psth_path_list, ch_psth_meta_path_list):

    if not (save_out_path / 'plots_psth_bysc').exists():
        os.makedirs(save_out_path / 'plots_psth_bysc',exist_ok= True)
    #get_rastor_bysc_day(n_chan, ch_st_sc_path_list)

    n_runs = len(ch_psth_path_list)
    if n_runs == 0:
        sys.exit('no spike time data was provided')

    max_fr_all = []
    for n,p in enumerate(ch_psth_path_list):
        psth = pickle.load(open(p,'rb'))
        psth_mean = np.mean(psth, axis=0)
        # error bars 
        psth_sem = np.std(psth, axis=0)/np.sqrt(len(psth))
        max_fr_all.append(np.nanmax(psth_mean + psth_sem))

    max_fr = max(max_fr_all)

    n_cols = 4
    n_rows = math.ceil(n_runs/n_cols)
    plot_width = 15
    plot_height = 5 * n_rows

    fig, axs = plt.subplots(nrows= n_rows,ncols = n_cols, figsize = (plot_width,plot_height))

    n_spikes_tot = 0
    for n,(p,m) in enumerate(zip(ch_psth_path_list,ch_psth_meta_path_list)):
        psth = pickle.load(open(p,'rb'))
        psth_meta = pickle.load(open(m,'rb'))

        psth_mean = np.mean(psth, axis=0)
        # error bars 
        psth_sem = np.std(psth, axis=0)/np.sqrt(len(psth))

        bincents = psth_meta['psth_bins'] - psth_meta['binwidth']/2
        bincents = bincents[1:]

        ax = axs.ravel()[n]
        ax.plot(bincents, psth_mean)
        ax.fill_between(bincents, psth_mean-psth_sem, psth_mean+psth_sem,
            alpha=0.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FR (spks/sec)')
        ax.set_xlim([-psth_meta['t_before'], psth_meta['max_sc_dur']+ psth_meta['t_after']])
        ax.set_ylim([0,max_fr])
        ax.set_title(Path(psth_meta['scenefile']).stem + '\n' + psth_meta['behav_file'] + ' \n' + str(psth_meta['n_trs']) + ' trials ' + 
                    str(psth_meta['n_spikes_sc']) + ' spikes' )
        
        n_spikes_tot += psth_meta['n_spikes_sc']

    fig.suptitle('ch{:0>3d} \n '.format(n_chan) + '\n' + str(n_spikes_tot) + ' spikes out of ' + str(psth_meta['n_spikes_tot']))
    fig.tight_layout()
    plt.subplots_adjust(wspace = 1)
    filename = 'ch{:0>3d}.png'.format(n_chan)
    plt.savefig(save_out_path / 'plots_psth_bysc'/ filename, bbox_inches = 'tight')
    plt.close()


def gen_psth_byscenefile(n_chan,save_out_path, psth, psth_meta,chanmap):
    if not (save_out_path / 'plots_psth_byscenefile').exists():
        print('creating plot directory')
        os.makedirs(save_out_path / 'plots_psth_byscenefile',exist_ok= True)

    max_fr_all = []
    for s in psth:
        n_trials = np.sum([1 for p in psth[s] if not np.isnan(p).all() ])
        psth_mean = np.nanmean(psth[s], axis=0)
        # error bars 
        psth_sem = np.nanstd(psth[s], axis=0)/np.sqrt(n_trials)

        max_fr_all.append(np.nanmax(psth_mean + psth_sem))

    max_fr = np.nanmax(max_fr_all)

    n_cols = math.ceil(len(psth)/2)
    n_rows = 2

    plot_width = 15/4 * n_cols
    plot_height = 15/4 * n_rows
    fig, axs = plt.subplots(nrows= n_rows,ncols = n_cols, figsize = (plot_width,plot_height))


    for n,s in enumerate(psth):
        n_trials = np.sum([1 for p in psth[s] if not np.isnan(p).all() ])
        psth_mean = np.nanmean(psth[s], axis=0)
        # error bars 
        psth_sem = np.nanstd(psth[s], axis=0)/np.sqrt(n_trials)

        bincents = psth_meta[s]['psth_bins'] - psth_meta[s]['binwidth']/2
        bincents = bincents[1:]
        
        ax = axs.ravel()[n]
        ax.plot(bincents, psth_mean)
        ax.fill_between(bincents, psth_mean-psth_sem, psth_mean+psth_sem,
            alpha=0.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FR (spks/sec)')
        ax.set_xlim([-psth_meta[s]['t_before'], psth_meta[s]['stim_dur']+ psth_meta[s]['t_after']])
        ax.set_ylim([0,max_fr])
        ax.set_title(Path(s).stem + '\n' + str(n_trials) + ' trials ')

    print(n_chan,int(np.where(np.argsort(chanmap[:,1]) == n_chan)[0]))
    fig.suptitle('ch{:0>3d} \n '.format(int(np.where(np.argsort(chanmap[:,1]) == n_chan)[0])))
    fig.tight_layout()
    plt.subplots_adjust(wspace = 1)
    filename = 'ch{:0>3d}.png'.format(int(np.where(np.argsort(chanmap[:,1]) == n_chan)[0]))

    plt.savefig(save_out_path / 'plots_psth_byscenefile'/ filename, bbox_inches = 'tight')  
    plt.close()
  

def get_FSI_allchans(save_out_path):
    # add elias and neptune token if available # 6/5/2024
    config = Config()
    face_type = ['all','oldman','neptune','elias','neptune_big','sophie']

    FSI_chan = pickle.load(open(save_out_path / 'ch000_FSI','rb'))

    n_bins = FSI_chan['all'].shape[0]
    n_chans= 384

    FSI_concat = dict.fromkeys(face_type, np.nan * np.ones((n_chans,n_bins)))
    FSI_early = dict.fromkeys(face_type, np.nan* np.ones(n_chans))
    FSI_late = dict.fromkeys(face_type, np.nan * np.ones(n_chans))

    for ch in range(n_chans):
        FSI_chan = pickle.load(open(save_out_path / 'ch{:0>3d}_FSI'.format(ch),'rb'))

        for ft in face_type:
            if ft in FSI_chan:
                FSI_concat[ft][ch,:] = FSI_chan[ft]
                if not np.isnan(FSI_chan[ft]).all():
                    FSI_early[ft][ch] = np.nanmean(FSI_chan[ft][config.t_early_bin])
                    FSI_late[ft][ch] = np.nanmean(FSI_chan[ft][config.t_late_bin])

    si_threshold = 0.33
    FSI = dict.fromkeys(face_type, (np.nan, np.nan))
    for ft in face_type:
        if np.isnan(FSI_chan[ft]).all():
           continue
        ch_FSI_early =  np.where(FSI_early[ft] >= si_threshold)[0]
        ch_FSI_late = np.where(FSI_late[ft] >= si_threshold)[0]
        FSI[ft] = (ch_FSI_early, ch_FSI_late)
    
    pickle.dump(FSI_concat, open(save_out_path / 'FSI_allchans','wb'), protocol = 2)
    pickle.dump(FSI, open(save_out_path / 'FSI','wb'), protocol = 2)

    return FSI_concat, FSI 

def get_FSI_allchans_bytype(save_out_path, FSI_type):
    n_chans = 384
    FSI_chan = pickle.load(open(save_out_path / ('ch000_FSI_' + FSI_type),'rb'))

    n_bins = FSI_chan.shape[0]

    FSI_concat_all = np.nan  * np.ones((n_chans,n_bins))

    t_early_bins = list(range(15,20)) # 50-100ms
    t_late_bins = list(range(20,25)) # 100-150ms

    FSI_all_early = np.nan * np.ones(n_chans)
    FSI_all_late = np.nan * np.ones(n_chans)

    for ch in range(n_chans):
        
        FSI_chan = pickle.load(open(save_out_path / ('ch{:0>3d}_FSI_'.format(ch) + FSI_type),'rb'))
        FSI_concat_all[ch,:]= FSI_chan
        FSI_all_early[ch] = np.nanmean(FSI_chan[t_early_bins])
        FSI_all_late[ch] = np.nanmean(FSI_chan[t_late_bins])

    FSI_concat = dict()
    FSI_concat[FSI_type] = FSI_concat_all

    FSI_early_all = np.where(FSI_all_early >= 0.33)[0]
    print(FSI_early_all)
    FSI_late_all = np.where(FSI_all_late >= 0.33)[0]

    FSI = dict()
    FSI[FSI_type] = (FSI_early_all,FSI_late_all)

    pickle.dump(FSI_concat, open(save_out_path / ('FSI_allchans_' + FSI_type),'wb'), protocol = 2)

    pickle.dump(FSI, open(save_out_path / ('FSI_' + FSI_type),'wb'), protocol = 2)

    return FSI_concat, FSI 

def get_CSI_rust_allchans(save_out_path):
    n_chans = 384
    FSI_chan = pickle.load(open(save_out_path / 'ch000_CSI_rust','rb'))

    n_bins = FSI_chan['scene'].shape[0]

    FSI_concat_all = np.nan  * np.ones((n_chans,n_bins))

    t_early_bins = list(range(15,20)) # 50-100ms
    t_late_bins = list(range(20,25)) # 100-150ms

    FSI_all_early = np.nan * np.ones(n_chans)
    FSI_all_late = np.nan * np.ones(n_chans)

    for ch in range(n_chans):
        
        FSI_chan = pickle.load(open(save_out_path / 'ch{:0>3d}_CSI_rust'.format(ch),'rb'))
        FSI_concat_all[ch,:]= FSI_chan['scene']
        FSI_all_early[ch] = np.nanmean(FSI_chan['scene'][t_early_bins])
        FSI_all_late[ch] = np.nanmean(FSI_chan['scene'][t_late_bins])

    FSI_concat = dict()
    FSI_concat['rust_scene'] = FSI_concat_all

    FSI_early_all = np.where(FSI_all_early >= 0.33)[0]
    print(FSI_early_all)
    FSI_late_all = np.where(FSI_all_late >= 0.33)[0]

    FSI = dict()
    FSI['rust_scene'] = (FSI_early_all,FSI_late_all)

    pickle.dump(FSI_concat, open(save_out_path / 'SSI_allchans_rust','wb'), protocol = 2)

    pickle.dump(FSI, open(save_out_path / 'SSI_rust','wb'), protocol = 2)

    return FSI_concat, FSI 

def get_CSI_FBOP_allchans(save_out_path, type):
    n_chans = 384
    CSI_chan = pickle.load(open(save_out_path / 'ch000_CSI_FBOP','rb'))

    n_bins = CSI_chan['face'].shape[0]

    CSI_concat_all = np.nan  * np.ones((n_chans,n_bins))

    t_early_bins = list(range(15,20)) # 50-100ms
    t_late_bins = list(range(20,25)) # 100-150ms

    CSI_all_early = np.nan * np.ones(n_chans)
    CSI_all_late = np.nan * np.ones(n_chans)

    for ch in range(n_chans):
        
        CSI_chan = pickle.load(open(save_out_path / 'ch{:0>3d}_CSI_FBOP'.format(ch),'rb'))
        CSI_concat_all[ch,:]= CSI_chan[type]
        CSI_all_early[ch] = np.nanmean(CSI_chan[type][t_early_bins])
        CSI_all_late[ch] = np.nanmean(CSI_chan[type][t_late_bins])

    CSI_concat = dict()
    CSI_concat['FBOP_' + type] = CSI_concat_all

    CSI_early_all = np.where(CSI_all_early >= 0.33)[0]
    print(CSI_early_all)
    CSI_late_all = np.where(CSI_all_late >= 0.33)[0]

    CSI = dict()
    CSI['FBOP_' + type] = (CSI_early_all,CSI_late_all)

    if type == 'scene':
        pickle.dump(CSI_concat, open(save_out_path / 'SSI_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'SSI','wb'), protocol = 2)
    elif type == 'object':
        pickle.dump(CSI_concat, open(save_out_path / 'OSI_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'OSI','wb'), protocol = 2)
    elif type == 'body':
        pickle.dump(CSI_concat, open(save_out_path / 'BSI_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'BSI','wb'), protocol = 2)

    return CSI_concat, CSI 


def get_CSI_SearchStimuli_allchans(save_out_path, type):
    n_chans = 384
    CSI_chan = pickle.load(open(save_out_path / 'ch000_CSI_SearchStimuli','rb'))

    n_bins = CSI_chan['face'].shape[0]

    CSI_concat_all = np.nan  * np.ones((n_chans,n_bins))

    t_early_bins = list(range(15,20)) # 50-100ms
    t_late_bins = list(range(20,25)) # 100-150ms

    CSI_all_early = np.nan * np.ones(n_chans)
    CSI_all_late = np.nan * np.ones(n_chans)

    for ch in range(n_chans):
        
        CSI_chan = pickle.load(open(save_out_path / 'ch{:0>3d}_CSI_SearchStimuli'.format(ch),'rb'))
        CSI_concat_all[ch,:]= CSI_chan[type]
        CSI_all_early[ch] = np.nanmean(CSI_chan[type][t_early_bins])
        CSI_all_late[ch] = np.nanmean(CSI_chan[type][t_late_bins])

    CSI_concat = dict()
    CSI_concat['SearchStimuli_' + type] = CSI_concat_all

    CSI_early_all = np.where(CSI_all_early >= 0.33)[0]
    print(CSI_early_all)
    CSI_late_all = np.where(CSI_all_late >= 0.33)[0]

    CSI = dict()
    CSI['SearchStimuli_' + type] = (CSI_early_all,CSI_late_all)

    if type == 'scene':
        pickle.dump(CSI_concat, open(save_out_path / 'SSI_SearchStimuli_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'SSI_SearchStimuli','wb'), protocol = 2)
    elif type == 'object':
        pickle.dump(CSI_concat, open(save_out_path / 'OSI_SearchStimuli_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'OSI_SearchStimuli','wb'), protocol = 2)
    elif type == 'body':
        pickle.dump(CSI_concat, open(save_out_path / 'BSI_SearchStimuli_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'BSI_SearchStimuli','wb'), protocol = 2)
    elif type == 'face':
        pickle.dump(CSI_concat, open(save_out_path / 'FSI_SearchStimuli_allchans','wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / 'FSI_SearchStimuli','wb'), protocol = 2)
    else:
        pickle.dump(CSI_concat, open(save_out_path /( 'FSI_' + type + 'SearchStimuli_allchans'),'wb'), protocol = 2)
        pickle.dump(CSI, open(save_out_path / ( 'FSI_' + type + 'SearchStimuli'),'wb'), protocol = 2)

    return CSI_concat, CSI 

def get_FSI_heatmap(FSI_concat, FSI, plot_save_out_path,  FSI_type,chanmap):

    n_chans_early = len(FSI[FSI_type][0])
    n_chans_late =  len(FSI[FSI_type][1])
    
    fig, ax = plt.subplots(figsize= (15,20))
    cmap = sns.color_palette("vlag", as_cmap=True)
    FSI_to_plot = FSI_concat[FSI_type][np.argsort(chanmap[:,1]),0:35]
    ax = sns.heatmap(FSI_to_plot, cmap=cmap, vmin = -1, vmax = 1)
    plt.gca().invert_yaxis()

    ax2 = ax.twiny()
    xmin, xmax = ax.get_xlim()
    n_bins = int(xmax - xmin)
    
    t_start = -0.1
    t_end = 0.3 -0.05
    binwidth_psth = 0.01
    psth_bins =np.arange(t_start, t_end, binwidth_psth)
    psth_binedge = np.hstack((psth_bins,t_end))

    xticks =np.array(list(range(n_bins+1)))
    ax.set_xticks(xticks)
    ax2.set_xticks(xticks)

    xticklabels = [''] * (n_bins+1)

    for n in range(int((n_bins+1)/5)):
        xticklabels[int(n*5)] = round(psth_binedge[int(n*5)],2)

    xticklabels[len(xticklabels)-1] = round(psth_binedge[len(psth_binedge)-1],2)
    ax.set_xticklabels(xticklabels)
    ax2.set_xticklabels(xticklabels)

    ax.set_xlabel('time(seconds) from stimulus onset',  size= 20, labelpad = 20)
    ax.set_ylabel('ch num', size = 20, labelpad = 15)
    ax.set_title('FSI_' + FSI_type + '\nAbove 0.33 early= ' + str(n_chans_early) + ' late = ' + str(n_chans_late), size = 20, pad = 20)

    # onset dashed line
    ymin, ymax = ax.get_ylim()

    ax.plot([0 - t_start/binwidth_psth, 0 - t_start/binwidth_psth],[ymin,ymax], 'k--')
    

    filename =  'FSI_' + FSI_type + '.png'

    plt.rcParams['savefig.facecolor']='white'

    plt.savefig(plot_save_out_path/ filename,bbox_inches = 'tight')

    return fig 

def change_point_estimation_fast(psth,pre,pk_loc, pre_stim,plot=False):
    # original documentation by Elias Issa
     # psth = response histogram in 1 ms bins & baseline subtracted
# sc = resolution of coarse search (i.e. 4 ms)
# pre = lower bound on latency (i.e. 50 ms)
# pre_stim = pre-stimulus period
# peak_location = upper bound on latency

# Fast change point estimation done using two steps (a coarse step,
# followed by a fine step).  Searches for the best piecewise linear
# fit to the knee of the PSTH (based on Friedman & Priebe J.
# Neurosci. Methods 1998).
    
    pre = pre + pre_stim
    pk_loc = pk_loc + pre_stim
   
    # cumualtive 
    # make everythig positive 
    cpsth = np.cumsum(psth + np.abs(min(psth)))
    #cpsth= psth
	
    s = []
    c = []
    e = []
    #plt.plot(x,cpsth, 'k')
    xp = np.arange(0,len(cpsth))
    
    sc = 4

    for q in list(range(pre+5, pk_loc-4, sc)):
        x1 = np.vstack((np.arange(pre,q),np.ones(len(range(pre,q))))).T
        x2 = np.vstack((np.arange(q,pk_loc), np.ones(len(range(q,pk_loc))))).T
        y1 = cpsth[pre:q]
        y2 = cpsth[q:pk_loc]

        # slope, intercept of the first line
        m1,c1 = np.linalg.lstsq(x1,y1)[0]
        e1 = np.sqrt(np.nansum(((m1*xp[pre:q]+c1)-y1)**2))

        # slope, intercept of the second line
        m2,c2 = np.linalg.lstsq(x2,y2)[0]
        e2 = np.sqrt(np.nansum(((m2*xp[q:pk_loc]+c2)-y2)**2))

        s.append((m1, m2))
        e.append((e1, e2))
        c.append((c1,c2))

    s = np.vstack(s)
    e = np.vstack(e)
    c = np.vstack(c)

    ind = np.argmin(np.nanmean(e,axis = 1))
    cpe = sc* (ind-1) + pre+5


    # run again at higher resolution
    if cpe >5:
        s = []
        e = []
        c = []
        for q in range(cpe-5, cpe+5):
            x1 = np.vstack((np.arange(pre,q),np.ones(len(range(pre,q))))).T
            x2 = np.vstack((np.arange(q,pk_loc), np.ones(len(range(q,pk_loc))))).T
            y1 = cpsth[pre:q]
            y2 = cpsth[q:pk_loc]

            # slope, intercept of the first line
            m1,c1 = np.linalg.lstsq(x1,y1)[0]
            e1 = np.sqrt(np.nansum(((m1*xp[pre:q]+c1)-y1)**2))

            # slope, intercept of the second line
            m2,c2 = np.linalg.lstsq(x2,y2)[0]
            e2 = np.sqrt(np.nansum(((m2*xp[q:pk_loc]+c2)-y2)**2))

            s.append((m1, m2))
            e.append((e1, e2))
            c.append((c1,c2))

        s = np.vstack(s)
        e = np.vstack(e)
        c = np.vstack(c)

        ind = np.argmin(np.nanmean(e,axis = 1))
        cpe = (ind-1) + cpe -5
        
    if plot:
        fig, ax =plt.subplots()
        plt.plot(xp,cpsth, 'b')
        plt.plot(xp[pre:cpe], xp[pre:cpe] * s[ind,0] + c[ind,0], label = str(q), color = 'g')
        plt.plot(xp[cpe:pk_loc], xp[cpe:pk_loc] * s[ind,1] + c[ind,1], color = 'y')

    if xp[pk_loc] * s[ind,1] + c[ind,1] < xp[cpe] * s[ind,0] + c[ind,0]: # happens when it's noisy 
        quality = -1
    else:
        quality = 1 
    max_fr = np.nanmax(psth[cpe:pk_loc])
    if np.isnan(max_fr):
        max_fr_ind = np.nan
    else:

        max_fr_ind = np.where(psth[cpe:pk_loc] == max_fr)[0][0] + cpe

    return cpe, psth[cpe], max_fr_ind - pre_stim, max_fr, quality


def get_psth_stim_chan(stim_list,ch,save_out_path):
    # returns psth for a single channel
    # if a list of stimuli is given, returns a concatenated psth
    # all individual stimulus psth should have the same length 
    psth = pickle.load(open(save_out_path / 'ch{:0>3d}_psth_stim'.format(ch),'rb'))

    psth_stacked = []
    if isinstance(stim_list, (list, tuple, np.ndarray)):
        for stim in stim_list:
            psth_stacked.append(psth[stim])
    else:
        psth_stacked = psth[stim_list]
    
    psth_stacked = np.vstack(psth_stacked)   
    return psth_stacked


def gen_st_padded(st,t_on, present_bool):
    max_n_spikes = 0
    for st_ in st:
        max_n_spikes = np.max((max_n_spikes,len(st_)))

    print(max_n_spikes)
        
    st_to_plot = np.nan *  np.ones((len(st), max_n_spikes))
    for n,(st_,t, p) in enumerate(zip(st_stim, t_on,present_bool)):
        if ~np.isnan(t) and p == 1:
            if len(st_) < max_n_spikes:
                pad = 1000 * np.ones(max_n_spikes - len(st_))
                if len(st_) == 0:
                    st_to_plot[n,:] =pad
                else:
                    st_to_plot[n,:] = np.hstack((st_-t, pad))
            else: 
                st_to_plot[n,:] = st_-t
    
    return st_to_plot

def smooth_data_np_average(arr, span):  # my original, naive approach
    smoothed = np.array([np.average(arr[val - span:val + span + 1]) for val in range(len(arr))])

    # replacing nan values with the closest one
    for ind, val in enumerate(smoothed):
        non_nan_ind= np.argwhere(~np.isnan(smoothed))
        if math.isnan(val):
            smoothed[ind] = smoothed[non_nan_ind[np.argmin(non_nan_ind - ind)]]
    
    return smoothed





from scipy.interpolate import RectBivariateSpline

def get_CSD(lfp_allchs, t, Fs, sitespace, stim_info_sess):

    n_chans = lfp_allchs.shape[0]

    t_trs = np.sort(np.hstack([stim_info_sess[s]['t_on'] for s in stim_info_sess]))
    t_trs = t_trs[~np.isnan(t_trs)]
    n_trs_max = np.argmin(np.abs(t_trs - t[len(t)-1]))

    t_before = 0.4 
    t_after = 0.1 
    stim_dur = 0.3 

    t_win = int(t_before * Fs  + t_after * Fs + stim_dur * Fs)
    data_bystim = np.nan * np.ones((n_chans,t_win,n_trs_max + 1))
    for i, on_t in enumerate(t_trs[0:n_trs_max+1]):
        tidx = np.argmin(np.abs(t-on_t))
        samp_win = [tidx - t_before * Fs, tidx + stim_dur* Fs + t_after * Fs]

        if tidx - t_before*Fs > 0 and (samp_win[1] - samp_win[0] == t_win) and tidx + stim_dur* Fs + t_after * Fs < lfp_allchs.shape[1]: 
            data_bystim[:,:,i] = lfp_allchs[:,int(samp_win[0]):int(samp_win[1])]

    avg_lfp = np.nanmean(data_bystim,axis = 2)
    CAR = 1
    if CAR: 
        avg_lfp -= np.nanmean(avg_lfp, axis = 0)

       
    # perform spatial smoothing
    smoothby3 = 11
    downsampfact = 5
    n_chs_csd = int(np.round(n_chans/downsampfact))
    idxrange = np.floor(smoothby3 / 2) 
    chs_for_csd = np.round(np.linspace(idxrange, n_chans-idxrange,n_chs_csd)); 
    ss_lfp = np.squeeze(avg_lfp.T.reshape([1,avg_lfp.size]))
    ss_lfp = np.convolve(ss_lfp, np.ones(smoothby3)/smoothby3,mode='same') 
    ss_lfp = ss_lfp.reshape([avg_lfp.shape[1], avg_lfp.shape[0]]).T

    # calculate csd 
    csd_lfp = ss_lfp[chs_for_csd.astype(int),:]
    x = csd_lfp.shape[0]
    csd = (csd_lfp[0:(x-2*sitespace),:] + csd_lfp[(2*sitespace):x,:] \
        - 2*csd_lfp[(sitespace):(x-sitespace),:])/(sitespace^2)*-1
    interpfactor = 10; 
    #newc = imresize(CSD,[interpfactor*size(CSD,1) interpfactor*size(CSD,2)],'bilinear'); 
    #CSDtimebase = linspace(tightwin(1), tightwin(2), size(newc, 2));
    bottomch = chs_for_csd[sitespace]
    topch = chs_for_csd[-sitespace]

    xvals = np.arange(csd.shape[0])
    yvals = np.arange(csd.shape[1])

    interp_csd = RectBivariateSpline(xvals,yvals,csd)

    x2 = np.arange(xvals[0], xvals[-1], 1/5)
    y2 = np.arange(yvals[0], yvals[-1], 1/5)

    newcsd = interp_csd(x2, y2)
    t_new = np.linspace(-t_before,stim_dur + t_after, newcsd.shape[1])
    chs_new = np.linspace(bottomch, topch, newcsd.shape[0])

    return newcsd, ss_lfp, chs_for_csd
