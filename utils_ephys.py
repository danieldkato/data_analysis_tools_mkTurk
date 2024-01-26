import pickle
import numpy as np
import math

def load_data(n_chan,MUA_dir):
    # loads spike times, peak values of detected spikes, waveform 
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

    assert len(st) == len(wf) == len(pk) == len(sl), 'length of files does not match'

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
        iti_dur = stim_info_sess[stim]['iti_dur']
        stim_dur = np.nanmax(stim_info_sess[stim]['dur'])

        n_bins = int(np.ceil((stim_dur + iti_dur + t_before + t_after)/binwidth_psth))
        psth_bins = np.linspace(-t_before, stim_dur + iti_dur + t_after, n_bins+1)

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
            t_end = t_on + stim_dur + iti_dur + t_after
     
            if st.shape[0] > 1: 
                ind = np.unique(np.where((st >= t_start) & (st <= t_end))[0])
            else:
                ind = np.where((st >= t_start) & (st <= t_end))[0]
            
            ch_pk_stim[stim].append(pk[ind])
            ch_wf_stim[stim].append(wf[ind])
            ch_st_stim[stim].append(st[ind])
            ch_sl_stim[stim].append(sl[ind])
            ch_ind_stim[stim].append(ind)
            
            if stim_info_sess[stim]['present_bool'][i] == 0 or math.isnan(t_on):
                ch_psth_stim[stim][i,:] =np.nan * np.ones(len(psth_bins)-1)
            else:
                ch_psth_stim[stim][i,:], _ =  np.histogram(st[ind]-t_on, psth_bins)
                n_trials +=1

        ch_psth_stim_meta[stim]['n_trials'] = n_trials

    return ch_pk_stim, ch_wf_stim, ch_st_stim, ch_sl_stim, ch_ind_stim,ch_psth_stim,ch_psth_stim_meta

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
