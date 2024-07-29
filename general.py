import sys
import os
import warnings
import re
import time
import pickle
import json
import h5py
import pathlib as Path
from copy import deepcopy
import numpy as np
import pandas as pd
from natsort import os_sorted
from operator import itemgetter
import matplotlib.pyplot as plt
from data_analysis_tools_mkTurk.utils_meta import find_channels, get_recording_path, get_coords_sess, get_all_metadata_sess
from data_analysis_tools_mkTurk.utils_mkturk import expand_sess_scenefile2stim
from data_analysis_tools_mkTurk.npix import get_site_coords, partition_adjacent_channels
import gc
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError:
    warnings.warn('Failed to import analysis_metadata module.')


def selectivity_v_coords(base_data_path, monkey, date, class0, class1, inpt_dir=None, 
    response_metric='max', selectivity_metric='dprime', match_strs=None, config='short',
    spacing=10, tip_length=175, plot=False, save_output=False, output_directory=None):
    """
    

    Parameters
    ----------
    class0 : TYPE
        DESCRIPTION.
    
    class1 : TYPE
        DESCRIPTION.
    
    response_metric : 'max' | 'abs' | 'AUC', optional
        Method for quantifying responses on each trial. The default is 'max'.
        # TODO: Add support for 'abs', 'AUC'.
        
    selectivity_metric : TYPE, optional
        DESCRIPTION. The default is 'dprime'.
    
    psth_files : TYPE, optional
        DESCRIPTION. The default is None.
    
    save_output : TYPE, optional
        DESCRIPTION. The default is False.
    
    output_directory : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # Get coordinates of recording sites in current session:
    print('Getting recording site coordinates...')
    Coords = get_site_coords(base_data_path, monkey, date) # c-by-3, where c is number of channels

    # Define directory where to search for preprocessed data:
    if inpt_dir == None:
         inpt_dir = get_recording_path(base_data_path, monkey, date, depth=4)[0]        
    
    # Get unique channel indices for current session:
    print('Finding channels...')
    channels = find_channels(inpt_dir)
    
    # Iterate over channels:
    results_df = pd.DataFrame(columns=[selectivity_metric, 'ap', 'dv', 'ml'])
    fnames = os.listdir(inpt_dir)
    for cx, ch in enumerate(channels):
    
        # Define regular expression specifying files to load data from:
        if match_strs is not None:
            filename_filter = '(' + '|'.join(match_strs) + ')$'
            
            # ^ DDK 2024-01-22: Note that this regex looks for exact matches (cf. postpended '$'),
            # i.e., strings that don't contain any additional characters outside
            # the specified regular expression. So e.g., if match_strs is 
            # 'psth_scenefile', then this regex will exclude any files named 
            # 'ch<ccc>_psth_scenefile_meta'. Not sure if this always desirable but 
            # should work for now. 
            
        else:            
            filename_filter = None
        regex = 'ch'+str(ch).zfill(3) + filename_filter
    
    	# Get names of files matching search criteria:
        curr_channel_files = [x for x in fnames if re.search(regex,x) is not None]

        # Load, stich together contents of matching files:
        curr_channel_data = dict()
        for fx, f in enumerate(curr_channel_files):
            print('Loading file {} of {} for channel {}...'.format(fx, len(curr_channel_files), cx))
            fullpath = os.path.join(inpt_dir, f)
            curr_file_data = pickle.load(open(fullpath,'rb'))
            curr_channel_data.update(curr_file_data)
            
            # ^ TODO: Should probably add verification that there are no overlapping keys
            # between loaded dicts; otherwise not sure how .update() behaves
            # (does it overwrite old keys? Add duplicate keys?)
            
        # Compute response metric for every trial for current channel:
        for key in curr_channel_data:
            if response_metric == 'max':
                curr_channel_data[key] = np.max(curr_channel_data[key], axis=1)
        
        # ^ TODO: Add support for other ways of quantifying single-trial responses 
        # (e.g. AUC)
        
        # Compute selectivity for current channel:
        selectivity = compute_selectivity(curr_channel_data, class0, class1, selectivity_metric)
            
        # Compile results for current channel into dict and append to results table:    
        curr_ch_results=dict()
        curr_ch_results[selectivity_metric] = selectivity 
        curr_ch_results['ap'] = Coords[cx, 0]
        curr_ch_results['dv'] = Coords[cx, 1]
        curr_ch_results['ml'] = Coords[cx, 2]
        results_df.loc[len(results_df.index)] = curr_ch_results
    
    # Plot results if requested:
    if plot:
        fig = plot_selectivity_v_coords(results_df)
    
    # Save output:
    if save_output:
        
        # Set default output directory if necessary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            Path.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_directory, 'selectivity_analysis.pickle')
        with open(output_path, 'wb') as p:
            pickle.dump(results_df, p)
            
        if plot:
            fig_path = os.path.join(output_directory, 'selectivity_v_coords.png')
            plt.savefig(fig_path)
            
    

    return results_df



def plot_selectivity_v_coords(df):
    
    cols = df.columns
    if 'dprime' in cols:
        selectivity_metric = 'dprime'
    elif 'SI' in cols:
        selectivity_metric = 'SI'

    # TODO: Deal with exception handling    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    sc = ax.scatter3D(df['ap'], df['ml'], df['dv'], c=df[selectivity_metric])
    ax.set_xlabel('AP')
    ax.set_ylabel('ML')
    ax.set_zlabel('DV')  
    plt.colorbar(sc, label=selectivity_metric, orientation='vertical')
    
    return fig



def compute_selectivity(data, class0, class1, selectivity_metric='dprime'):
    """
    Compute selectivity of a single channel/unit to two stimulus classes. Note
    that each 'class' can in turn consist of several different individual stimulus
    conditions (e.g. all background-less objects vs all object-less backgrounds, etc.). 

    Parameters
    ----------
    data : dict
        Dict of PSTHs. Each key denotes a stimulus condition and corresponds to
        a value of a t_s-by-b numpy array of spike counts from a single channel/
        unit, where t_s is the number of trials of the corresponding stimulus 
        and b is the number of time bins. Note that b must be the same for all 
        stimuli. 
        
    class0 : array-like
        List or array of strings, each a key into `data` parameter defined
        above. Each element should correspond to a single stimulus condition in 
        the first class of stimuli to compute selectivity for. Note again that each 'class' 
        can consist of multiple individual stimulus conditions. 
        
    class1 : array-like
        Same format as `class0` above, but each key should denote an indivudual 
        stimulus condition within the second class of stimuli to compute 
        selectivity for.
        
    selectivity_metric : 'dprime' | 'SI', optional
        How to quantify selectivity. The default is 'dprime'.

    Returns
    -------
    result : 
        Selectivty of given channel/unit to given stimulus classes.

    """
    
    # Raise an error if class0 and class1 include overlapping individual 
    # stimulus conditions:
    unique_stim = np.union1d(class0, class1)
    overlapping_stim = [x for x in unique_stim if x in class0 and x in class1]
    if len(overlapping_stim) > 0:
        raise AssertionError('Following stimulus conditions included in both class0 and class 1: {}'.format(overlapping_stim))
    
    # Compute responses for each class:
    class_responses = []
    for cidx, cl in enumerate([class0, class1]):
        
        # Retrieve PSTHs for all stimulus conditions *within* current class; 
        # returned as s-element tuple where each element is a t-by-b numpy array 
        # corresponding one condition within the current class, where s is the
        # number of individual stimulus conditions in the current class, t is 
        # the number of trials of the corresponding stimulus condition, and b 
        # is number of time bins:
        curr_psths = itemgetter(*cl)(data) # tuple
        
        # Raise error if no matching stimuli discovered in data:
        if len(curr_psths) == 0:
            raise AssertionError('No stimuli of class {} discovered in input data.'.format(cidx))
        
        # Convert to u-by-b array, where u is total number of trials across all 
        # stimulus conditions within current class:
        curr_psths = np.hstack(curr_psths)  
                    
        class_responses.append(curr_psths)
        
    # Compute selectivity of responses:
    mu0 = np.nanmean(class_responses[0])
    mu1 = np.nanmean(class_responses[1])
    
    # If using d':
    if selectivity_metric == 'dprime':
        sigma0 = np.nanvar(class_responses[0])
        sigma1 = np.nanvar(class_responses[1])        
        result = (mu0-mu1) / np.sqrt( (sigma0+sigma1)/2 )
    
    # If using selectivity index:
    elif selectivity_metric == 'SI':
        result = (mu0+mu1) / (np.abs(mu0)+np.abs(mu1))
    
    return result




def get_stim_timing_metadata(stim_conditions_meta):
    
    # Get bin widths:
    binwidths = [x['binwidth'] for x in stim_conditions_meta]
    if np.ptp(binwidths) == 0: # If bin widths are the same for all stimulus conditions:
        binwidth = binwidths[0]
    else:
        # TODO DDK 2024-01-26: Add exception handling for this case?
        binwidth = None
    
    # Get bin edges:
    psth_bins = [x['psth_bins'] for x in stim_conditions_meta]
    if not np.any(np.ptp(psth_bins, axis=0)): # If bin centers are the same for all stimulus conditions:
        psth_bins = psth_bins[0]
        stim_on_index = np.where(psth_bins==0)[0][0] # < TODO DDK 2024-01-26: Add exception handling for (presumably rare) case where there is no bin centered at 0
    else:
        # TODO DDK 2024-01-26: Add exception handling for this case?
        psth_bins = None
        stim_on_index = None
        pass         
    
    return psth_bins, stim_on_index, binwidth 




def rolling_std(x, window):
    """
    Compute rolling standard deviation of input array. 

    Parameters
    ----------
    x : array-like
        Data to compute rolling stdev for.

    window : int
        Window half-width, in samples.

    Returns
    -------
    std_rolling : numpy.ndarray
        n-element aray, where n is the number of elements in input `x`. The i-th
        element of `std_rolling` is the standard deviation np.std(x[i-window:i+window]) 
        of the window extending `window` samples before and after i.

    """
    
    # Leading edge:
    leading = np.std(x[0:window])*np.ones(window)
    
    # Center:
    center_indices = np.arange(window, len(x)-window)
    center = np.array([np.std(x[i-window:i+window]) for i in center_indices])
    
    # Trailing edge:
    trailing = np.std(x[-window:])*np.ones(window)
    
    # Combine:
    std_rolling = np.concatenate((leading, center, trailing), axis=0)
    
    return std_rolling



def time_window2bin_indices(plot_window, psth_bins):
    
    if plot_window[0] < np.min(psth_bins):
        raise AssertionError('Requested plot window extends before first available sample in input data.')
    if plot_window[1] > np.max(psth_bins):        
        raise AssertionError('Requested plot window extends after last available sample in input data.')
    
    geq_lower_index = np.argwhere(psth_bins >= plot_window[0])
    first_index = np.min(geq_lower_index)    
    
    leq_greater_index = np.argwhere(psth_bins <= plot_window[1])    
    last_index = np.max(leq_greater_index)
        
    return [first_index, last_index]



def time_2_bin_index(t, psth_bins, round_dir='down'):
    
    # Raise warnings, return None if matching PSTH bins not found:
    within_psth_start = t >= min(psth_bins)
    within_psth_stop = t <= max(psth_bins)
    valid = within_psth_start and within_psth_stop
    if not valid:
        if not within_psth_start:
            warnings.warn('Requested time before PSTH start, returning None.')
        elif not within_psth_stop:
            warnings.warn('Requested time after PSTH, returning None.')
        return None
    
    if round_dir == 'up':
        geq = np.argwhere(psth_bins >= t)
        idx = np.min(geq)
    elif round_dir == 'down':
        leq = np.argwhere(psth_bins <= t)
        idx = np.max(leq)
        
    return idx



def bin_channels(trial_df, bin_size=4, sites_df=None):
    """
    Average spike count data across bins of adjacent channels.

    Parameters
    ----------
    trial_df : pandas.core.frame.DataFrame
        Dataframe containing PSTH data. Must at least define column 'psth', each
        element of which is a c-by-b array, where c is the number of channels and
        b is the number of timebins per peristimulus period (like in output of
        h5_2_df() function). 
    
    bin_size : int, optional
        Number of adjacent channels to average across. The default is 4.
    
    sites_df : pandas.core.frame.DataFrame, optional
        Dataframe of recording site location coordinates. Must at least define
        columns 'depth' and 'channel' (as in output of get_site_coords() 
        function). If None, will just average channels with adjacent indices.
    

    Returns
    -------
    trial_df : pandas.core.frame.DataFrame
        Same format as input, except each array in `psth` column is now d-by-b,
        where d is the number of channel bins. If `bin_size` evenly divides the
        total number of channels, then d = c/bin_size. Otherwise, 
        d = floor(c/bin_size) + 1, where the first d - 1 bins average across
        bin_size bins, and the last bin averages across the remain

    """
    
    # Partition channels into bins of adjacent channels:
    if sites_df is not None:
        bin_indices = partition_adjacent_channels(sites_df, bin_size=bin_size)
    else:
        n_channels = trial_df.iloc[0].psth.shape[0] # < Hack; assuming the same for all trials (probably fine)
        channels = np.arange(n_channels)
        bin_indices = partition_list(channels, bin_size)
    
    # Extract spike count data from dataframe:
    psth_data = np.array(list(trial_df.psth)) # < trials -by- channels -by- timebins
    psth_data = np.transpose(psth_data, axes=[1, 2, 0]) # < convert to channels -by- timebins -by- trials
    
    # Average across bins:
    psth_data_binned = bin_channels_mat(deepcopy(psth_data), bin_indices) # < channelbins -by- timebins -by- trials
    
    # Deal data back to dataframe:
    psth_data_binned = list(np.transpose(psth_data_binned, axes=[2, 0, 1])) # < convert back to trials -by- channelbins -by- trials
    trial_df['psth'] = psth_data_binned
    
    return trial_df

    

def bin_channels_mat(data, bin_indices):
    """
    Bin spike count data into arbitrary partitions of channels and average
    across channels.

    Parameters
    ----------
    data : array-like
        c-by-b-by-s array of spike counts, where c is the  number of channels,
        b is the number of time bins per peristimulus period, t is the number of 
        trials, and s is the number of stim presentations.
    
    bin_indices : list
        List of d numpy arrays, where d is the number of channel bins. Each 
        inner array must be of length f, where f is the number of channels per
        bin. 

    Returns
    -------
    binned_data : numpy.ndarray
        d-by-b-by-s, where again d is the number of channel bins. The i-th
        row (i.e. axis 0) represents spike counts averaged across all channels 
        specified in the i-th bin in `bin_indices`. 

    """
    
    # Initialize array of binned data:
    data_in_shape = data.shape
    binned_data = np.empty((len(bin_indices), data_in_shape[1], data_in_shape[2]))
    binned_data[:] = np.nan
    
    # Iterate over bins:
    for i, b in enumerate(bin_indices):
        curr_bin_mean = np.nanmean(data[b, :, :], axis=0)
        binned_data[i, :, :] = curr_bin_mean
    
    return binned_data



def rsvp_from_df(df, stim_inds):
    """
    Retrieve dataframe rows corresponding to specific RSVP trials. 

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe containing information about stimulus presentations. Each row
        must correspond to a single stimulus presentation (e.g. same format as 
       output of h5_2_df() function). Must define columns 'trial_num' and 'rsvp_num'. 
        
    stim_inds : array-like
        s-by-2 array, where s is the number of individual stimulus presentations
        to get data for. Col 0: trial number (absolute), col 1: RSVP stim number.  

    Returns
    -------
    df_out : pandas.core.frame.DataFrame
        Same format as input dataframe, but only subset of rows for requested
        stimulus presentations.

    """
    inds_df = pd.DataFrame(columns=['trial_num', 'rsvp_num'])
    inds_df['trial_num'] = stim_inds[:, 0]
    inds_df['rsvp_num'] = stim_inds[:, 1]
    
    # Find which rows of df correspond to requested (trial_num, rsvp_num) pairs:
    #compare_cols = ['trial_num', 'rsvp_num']
    #mask = pd.Series(list(zip(*[df[c] for c in compare_cols]))).isin(list(zip(*[inds_df[c] for c in compare_cols])))
    
    f = lambda r : df[np.array(df.trial_num==r.trial_num) & np.array(df.rsvp_num==r.rsvp_num)].index[0]
    
    print('inds_df.shape = {}'.format(inds_df.shape))
    matches = inds_df.apply(f, axis=1)
    
    df_out = df.loc[matches.values]
    
    return df_out



def df_2_psth_mat(df, dtype=np.float64):
    """
    Convert spike counts encoded in dataframe to numpy array.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        s-row dataframe of spike counts. where s is the number of individual
        stimulus presentations. Must define column 'psth' containing c-by-b
        numpy arrays, where c is the number of channels and b is the number of 
        time bins per stimulus presentation. 

    Returns
    -------
    data : numpy.ndarray
        If input dataframe contains PSTH data for multiple channels, c-by-b-by-s 
        array of spike count data. If input dataframe contains PSTH data for 
        only one channel, s-by-b. 

    """
    
    data = np.array(list(df.psth)).astype(dtype)
    
    # If data are 3-dimensional, assume repeat-by-channel-by-timebin
    if len(data.shape)==3:
        data = np.transpose(data, axes=[1, 2, 0]) # Convert to c-by-b-by-s
        
    return data



def df_2_rsvp_mat(df):

    # Sort dataframe by trial then rsvp num:
    df = df.sort_values(by=['trial_num', 'rsvp_num'])

    # Get data dimensions:
    min_trial = min(df.trial_num)
    max_trial = max(df.trial_num)
    n_trials = max_trial - min_trial + 1

    max_rsvp = max(df.rsvp_num)    
    n_rsvp = max_rsvp + 1

    shapes = np.array([x.shape for x in df.psth])
    n_chan = max(shapes[:,0])
    n_timebins = max(shapes[:,1])
    
    # Initialize output array:
    A = np.empty((n_chan, n_timebins, n_trials, n_rsvp))    
    A[:] = np.nan

    # Populate output array: 
    for r in np.arange(n_rsvp):
        
        curr_rsvp_rows = df[df.rsvp_num==r] 
        curr_rsvp_data = np.array(list(curr_rsvp_rows.psth)) # Repeats-by-channels-by-timebins
        curr_rsvp_data = np.transpose(curr_rsvp_data, axes=[1,2,0]) # Convert to channels-by-timebins-by-repeats
        curr_rsvp_trial_nums = curr_rsvp_rows.trial_num
        A[:,:,curr_rsvp_trial_nums-min_trial,r] = curr_rsvp_data
        
    return A



def select_channels(df, channels):
    """
    Extract spike counts for a single channel from a dataframe of trialized PSTH 
    data.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe of trialized PSTHs. Must define a column 'psth', each element
        of which is a c-by-b array of spike counts, where c is the number of
        channels in the recording and b is the number of timebins per
        peristimulus period.
        
    channels : array-like | int
        List of channel indices to extract data for.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Same as input, except each element of 'psth' column is now merely a 
        b-element array of spike counts for a single channel.

    """
    
    data = df_2_psth_mat(df) # < channels-by-timebins-by-repeats
    curr_data = data[channels, :, :] # < channels-by-timebins-by-repeats
    
    if np.ndim(curr_data)==3:
        curr_data = np.transpose(curr_data, axes=[2, 0, 1])
    elif np.ndim(curr_data)==2:
        curr_data = curr_data.T
        
    df.psth = list(curr_data)
    
    return df



def visual_drive(trial_df, baseline_window, psth_bins=None, sig=1, classes=None, 
     smooth_window=None):
    """
    Find channels that are significantly visually driven by any of a given set 
    of stimulus classes. Note that a 'class' of stimuli may consist of more than one 
    individual stimulus condition. 

    Parameters
    ----------
    trial_df : pandas.core.frame.DataFrame
        Dataframe containing spike count data. Must define columns 'stim_id' and
        'psth'. Each element of the 'psth' column must be a c-by-b array,
        where c is the number of channels and b is the number of timebins per
        peristimulus period.
    
    baseline_window : array-like
        2-element list or array specifying time window that counts as baseline.
        Element 0: start of baseline window, in seconds relative to trial start
        (negative if starting before trial). Element 1: end of baseline window. 
    
    psth_bins : array-like
        b+1-element array of timebin edges, in seconds relative to trial start.
        Specifies the edeges of the time bins corresponding to each column of
        each c-by-b array in the 'psth' field of the input dataframe. 
    
    sig : float, optional
        The number of standard deviations a channel's spike rate must be modulated
        by (in either direction) to count as driven by a stimulus calss. The default is 1.
    
    classes : list, optional
        List of stim_ids. Channels will be scored as visually-driven if their
        spike rate deflects either up or down by the specified number of standard
        deviations for ANY of the stimulus classes specified in `classes`. 
        
    smooth_window : int, optional
        Number of adjacent time bins in each direction to average each sample 
        with if temporally smoothing data. If None, then no temporal smoothing 
        is applied.

    Returns
    -------
    max_responses : numpy.ndarray
        [TODO : add]
    
    visually_driven : numpy.ndarray
        Array of indices of visually-driven channels. Note these are indices into
        rows of elements of `trial_df.psth`, which may not be absolute channel
        indices if channels have already been excluded in the input dataframe.

    """
    
    # TODO: If classes is None, just make every individual stimulus condition a class
    # by default:
    if classes is None:
        B = [i for i,x in trial_df.iterrows() if x.stim_id is not None]
        stim_id_rows = trial_df.loc[B]
        unique_stim_ids = np.unique(stim_id_rows.stim_id)
        classes = [[x] for x in unique_stim_ids]
        
    # Try to find psth_bin edges from input dataframe by default (maybe this should be a function):
    try: 
        P = np.array(list(trial_df['psth_bins']))
        # If all bin edges are the same, use bin edges from first trial:
        if sum(np.ptp(P,axis=0)) == 0:
            psth_bins = trial_df.iloc[0].psth_bins
    # Otherwise try to get bin edges from input param:
    except KeyError:
        psth_bins = psth_bins
    # If still can't find bin edges, raise error:
    if psth_bins is None:
        raise AssertionError('Could not find PSTH bin edges.')

    # Compute indices of baseline period:
    bl_edge_indices = time_window2bin_indices(baseline_window, psth_bins)
    bl_indices = np.arange(bl_edge_indices[0], bl_edge_indices[1])

    # Extract only rows corresponding to RSVP=0 stimulus presentations:
    rsvp0_rows = trial_df[trial_df.rsvp_num==0]

    # Baseline-subtract all trials:
    rsvp0_rows['psth'] = rsvp0_rows.apply(lambda x : bl_subtract_data(x.psth, baseline_window, psth_bins), axis=1)
    all_rsvp0_data = df_2_psth_mat(rsvp0_rows)

    # Compute stdev for all channels:
    #sds = get_ch_stdev(trial_df, baseline_window, psth_bins)

    #"""
    
    # Extract only pre-trial baseline period data, compute standard dev:
    all_bline = all_rsvp0_data[:, bl_indices, :] 
    n_channels = all_bline.shape[0]
    n_bins = all_bline.shape[1]
    n_repeats = all_bline.shape[2]
    all_bline_reshape = np.reshape(all_bline, (n_channels, len(bl_indices)*n_repeats), order='F')
    sds = np.nanstd(all_bline_reshape, axis=1)     
    

    #"""
    
    # Initialize arrays 
    #n_channels = len(sds)
    P = np.zeros((n_channels, len(classes))) # < Whether each channel is significantly deflected
    R = np.zeros((n_channels, len(classes))) # < Response of each channel to each stim
    
    # Iterate over classes:
    for cx, cl in enumerate(classes):
        
        if type(cl) == list:
            cl_ar = np.array(cl)
        else:
            cl_ar = cl
        
        curr_class_rows = rsvp0_rows[rsvp0_rows.stim_id.isin(cl_ar)] # c-by-b-by-s 
        if curr_class_rows.shape[0] == 0: # If there are no RSVP 0 presentations of the current stim, move on to the next one
            continue
        
        # Center each trial around 0:
        curr_class_rows_centered = bl_subtract_data(curr_class_rows, baseline_window, psth_bins)
        
        # Average over individual presentations:
        curr_class_data = df_2_psth_mat(curr_class_rows_centered)
        curr_class_means = np.nanmean(curr_class_data, axis=2) # c-by-b
        
        # Smooth mean PSTH:
        if smooth_window is not None:
            kernel = np.ones(smooth_window)/smooth_window
            curr_class_means = np.array([np.convolve(row, kernel, 'same') for row in curr_class_means])
        
        # Select only data from after baseline period:
        curr_class_means = curr_class_means[:, bl_edge_indices[1]:]
        curr_class_means = np.nanmax(curr_class_means, axis=1)
        
        #curr_class_blines = curr_class_means[:, bl_indices]
        #curr_class_bline_means = np.nanmean(curr_class_blines, axis=1)
        
        # Test whether peak of mean is above specified number of standard deviations:
        sig_deflection = np.abs(curr_class_means) > sig * sds
        #sig_deflection = np.abs(curr_class_means - curr_class_bline_means) > sig * sds
        
        # Update table:
        P[:, cx] = sig_deflection
        R[:, cx] = curr_class_means
        
    # Test whether each channel significantly deflects above specified number
    # of standard deviations for *any* requested class:    
    P = np.any(P, axis=1)
    visually_driven = np.where(P)[0]
        
    # Take max response of each channel to any stimulus:
    max_responses = np.max(R, axis=1)
        
    return visually_driven, max_responses



def get_bl_data(trial_df, baseline_window, psth_bins):
    """
    

    Parameters
    ----------
    trial_df : TYPE
        DESCRIPTION.
    
    baseline_window : TYPE
        DESCRIPTION.
    
    psth_bins : TYPE
        DESCRIPTION.

    Returns
    -------
    trial_bl : pandas.core.frame.DataFrame
        Same format as input dataframe, except each element of 'psth' column 
        is just baseline period spike counts instead of for the whole peristim
        period.

    """
        

    trial_bl = deepcopy(trial_df)    

    # Compute indices of baseline period:
    bl_edge_indices = time_window2bin_indices(baseline_window, psth_bins)
    bl_indices = np.arange(bl_edge_indices[0], bl_edge_indices[1])    

    # Extract data, select only bins from baseline period:
    data = df_2_psth_mat(trial_bl)
    bline_data = data[:, bl_indices, :]
    
    # Deal data back to dataframe:
    trial_bl.psth = list(np.transpose(bline_data, axes=[2, 0, 1]))

    return trial_bl    
    


def bl_subtract_data(psth, baseline_window, psth_bins):
    """
    Subtract mean baseline epoch activity for all stimulus presentations. Works 
    on a repeat-by-repeat basis, so that all activity for a given repeat has the 
    baseline epoch activity for that specific repeat subtracted from it. In other 
    words, guarantees that mean baseline activity on every repeat is 0. 

    Parameters
    ----------
    psth : pandas.core.frame.DataFrame
        Channels-by-timebins array of spike counts. 
    
    baseline_window : array-like
        2-element list or array specifying time window that counts as baseline.
        Element 0: start of baseline window, in seconds relative to trial start
        (negative if starting before trial). Element 1: end of baseline window. 
    
    psth_bins : array-like
        b+1-element array of timebin edges, in seconds relative to trial start.
        Specifies the edeges of the time bins corresponding to each column of
        each c-by-b array in the 'psth' field of the input dataframe. 

    Returns
    -------
    psth : pandas.core.frame.DataFrame
        Same as input, except data for each channel has had its mean baseline 
        activity subtracted from it.

    """
    
    edge_bins = time_window2bin_indices(baseline_window, psth_bins)
    bline_dat = psth[:,edge_bins[0]:edge_bins[1]]
    mu = np.mean(bline_dat, axis=1)
    Mu = np.matlib.repmat(np.expand_dims(mu, axis=1), 1, psth.shape[1])
    psth = psth - Mu
    
    return psth



def subtract_trial_bl(df, baseline_window, psth_bins):
    
    # TODO: Verify that all trials included in df have data for RSVP baseline=0
    
    # Get only RSVP=0 stim:
    rsvp0_rows = df[df.rsvp_num==0]
    
    # Get baseline period data:
    bline_df = get_bl_data(rsvp0_rows, baseline_window, psth_bins) # < Select baseline data
    
    # Extract spike counts:
    bl_data = df_2_psth_mat(bline_df) # < channels-by-timebins-by-trials
    all_data = df_2_psth_mat(df)
    n_channels = all_data.shape[0]
    n_bins = all_data.shape[1]
    n_repeats = all_data.shape[2]    
    
    # Take mean across time:
    bl_mean = np.nanmean(bl_data, axis=1) # < channels-by-trials
    
    # Make dimensions compatible:
    BL = np.empty((n_channels, n_bins, n_repeats))
    for tx, t in enumerate(np.unique(df.trial_num)):
        curr_trial_num_slice_indices = np.where(df.trial_num==t)[0]
        curr_trial_bl0 = np.expand_dims(bl_mean[:,tx],axis=1)
        curr_trial_bl1 = np.matlib.repmat(curr_trial_bl0, 1, n_bins)
        curr_trial_bl2 = np.array([curr_trial_bl1] * len(curr_trial_num_slice_indices))
        curr_trial_bl3 = np.transpose(curr_trial_bl2, axes=[1, 2, 0])
        BL[:, :, curr_trial_num_slice_indices] = curr_trial_bl3
    
    # Subtract:
    all_data = all_data - BL
    
    # Deal data back to dataframe:
    all_data = np.transpose(all_data, axes=[2, 0, 1])
    df.psth = list(all_data)
    
    return df



def get_ch_stdev(df, baseline_window, psth_bins):
    """
    Compute baseline period (before fixation dot onset) standard deviation for 
    each channel.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
            Dataframe containing spike count data. Must define column 'psth'. 
            Each element of the 'psth' column must be a c-by-b array, where c is 
            the number of channels and b is the number of timebins per
            peristimulus period.
    
    baseline_window : array-like
        2-element list or array specifying time window that counts as baseline.
        Element 0: start of baseline window, in seconds relative to trial start
        (negative if starting before trial). Element 1: end of baseline window. 
    
    psth_bins : array-like
        b+1-element array of timebin edges, in seconds relative to trial start.
        Specifies the edeges of the time bins corresponding to each column of
        each c-by-b array in the 'psth' field of the input dataframe. 

    Returns
    -------
    ch_stdevs : numpy.ndarray
        c-element array of baseline period standard deviation in spike rate, 
        where c is the number of channels in the input data.

    """
    
    # Get only RSVP=0 stim:
    rsvp0_rows = df[df.rsvp_num==0]
    
    # Get baseline period data:
    bline_df = bl_subtract_data(rsvp0_rows, baseline_window, psth_bins) # < 0-center all trials
    bline_df = get_bl_data(bline_df, baseline_window, psth_bins) # < Select baseline data
    
    # Extract spike counts:
    bl_data = df_2_psth_mat(bline_df)
    
    # Reshape: 
    n_channels = bl_data.shape[0]
    n_bins = bl_data.shape[1]
    n_repeats = bl_data.shape[2]
    bl_data = np.reshape(bl_data, (n_channels, n_bins*n_repeats), order='F')
        
    # Take stdev across time:
    ch_stdevs =  np.nanstd(bl_data, axis=1)
    
    return ch_stdevs 



def zscore_df(df, baseline_window, psth_bins=None, reference='baseline'):
    """
    Z-score spike counts in dataframe.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe containing spike count data. Must define column 'psth'. 
        Each element of the 'psth' column must be a c-by-b array, where c is 
        the number of channels and b is the number of timebins per
        peristimulus period.
    
    baseline_window : TYPE
        2-element list or array specifying time window that counts as baseline.
        Element 0: start of baseline window, in seconds relative to trial start
        (negative if starting before trial). Element 1: end of baseline window. 
    
    psth_bins : TYPE
        b+1-element array of timebin edges, in seconds relative to trial start.
        Specifies the edeges of the time bins corresponding to each column of
        each c-by-b array in the 'psth' field of the input dataframe. 

    Returns
    -------
    dfz : pandas.core.frame.DataFrame
        Same as input, except data in 'psth' column are Z-scores rather than
        raw spike counts.

    """
    
    # Try to find psth_bin edges from input dataframe by default (maybe this should be a function):
    try: 
        P = np.array(list(df['psth_bins']))
        # If all bin edges are the same, use bin edges from first trial:
        if sum(np.ptp(P,axis=0)) == 0:
            psth_bins = df.iloc[0].psth_bins
    # Otherwise try to get bin edges from input param:
    except KeyError:
        psth_bins = psth_bins
    # If still can't find bin edges, raise error:
    if psth_bins is None:
        raise AssertionError('Could not find PSTH bin edges.')
    bline_bin_edges = time_window2bin_indices(baseline_window, psth_bins)
    
    # Try to find number of channels; raise error if not consistent across repeats:
    n_chans_per_rep = df.apply(lambda x : x.psth.shape[0], axis=1) 
    if np.ptp(n_chans_per_rep) == 0:
        n_channels = n_chans_per_rep.iloc[0]
    else:
        raise AssertionError('Different numbers of channels detected on different trials.')
    
    # Zero-center baseline period for each trial:
    df['psth'] = df.apply(lambda x : bl_subtract_data(x.psth, baseline_window, psth_bins), axis=1)
    
    # Get stdev from whole time series:
    if reference == 'full':
        data = np.concatenate(list(df['psth']), axis=1)
        stdevs = np.nanstd(data, axis=1)
        
    # Get stdev just from baseline period data:
    elif reference == 'baseline':
        data = df.apply(lambda x : x.psth[bline_bin_edges[0]:bline_bin_edges[1],:], axis=1)
        data = np.concatenate(list(data), axis=1)
        stdevs = np.nanstd(data, axis=1)# < channels-element array
    
    # Garbage collection:
    del data
    gc.collect()
    
    # Create matrix of stdevs, multiply all activity:
    smat = np.zeros((n_channels, n_channels))
    np.fill_diagonal(smat, 1/stdevs)
    df['psth'] = df.apply(lambda x : np.matmul(smat, x.psth), axis=1)
    
    return df



def partition_list(L, partition_size):
    """
    Partition an ordered list into a list of smaller, disjoint lists. If the
    requested partition size does not divide the length of the input list evenly,
    then the first floor(len(L)/partition_size) partitions will be of the
    requested partition size, while the final remaining partition will include
    the remaining elements. 

    Parameters
    ----------
    L : array-like
        List or array of ordered elements to be partitioned.
    
    partition_size : int
        Number of elemnts to go in each resulting partition.

    Returns
    -------
    partitions : list
        List of arrays. The inner arrays are disjoint subsets of the elements of
        the original input list. 

    """
    
    # Define partition edge indices:
    n_els = len(L)
    inds = np.arange(0, n_els+partition_size, partition_size)
    
    # Get indices for each partition:
    partitions = [L[inds[x]:inds[x+1]] for x in np.arange(len(inds)-1)]
    
    return partitions



def remove_duplicate_rsvp_indices(indices):
    """
    Remove duplicate RSVP indices.

    Parameters
    ----------
    indices : array-like
        s-by-2 array of RSVP indices, where s is the number of individual stimulus
        presentations queried. Col 0: trial number, col 1: RSVP number.

    Returns
    -------
    indices : numpy.ndarray
        t-by-2 array of RSVP indices, where t is the number of *unique* rows of
        input array `indices`.

    """
    p = pd.DataFrame(columns=['trial_num', 'rsvp_num'])
    p['trial_num'] = indices[:, 0]
    p['rsvp_num'] = indices[:, 1]
    p = p.drop_duplicates()
    indices = np.array([p.trial_num, p.rsvp_num]).T
    
    return indices



def find_signed_angle(ref, tgt, axis=1):
    """
    Find signed angle between reference and target vector. Finds sign and 
    magnitude of smallest angle (in terms of absolute magnitude) by which 
    reference vector needs to be rotated to obtain target vector (so e.g. will
    return 90 degrees rather than -270 degrees, etc.)

    Currently assumes that target vector equals reference vector rotated around
    a single cardinal axis. Raises an error otherwise. 

    Parameters
    ----------
    ref : array-like
        n-dimensional reference vector.
        
    tgt : TYPE
        n-dimensional target vector.
        
    axis : int, optional
        Cardinal axis around which refrence vector is rotated to yield target
        vector. The default is 1.

    Returns
    -------
    signed_angle : float
        Signed angled between reference and target vector, in degrees.

    """
    # Very hacky, currently only works for rotations around one cardinal axis. 

    # TODO: Make this more general, handle arbitrary 3D rotations even not 
    # around cardinal axes using formula:
        
    # R = I + (rq^T - qr^T)sin\theta + (qq^T + rr^T)(cos\theta - 1) 
    
    
    # Normalize angles:
    ref = ref/np.linalg.norm(ref)
    tgt = tgt/np.linalg.norm(tgt)
    
    # Project, find absolute magnitude of angle:
    p = np.dot(tgt, ref)
    p = np.round(p, decimals=6) # Eliminate any rounding errors
    angle_abs = np.arccos(p)
    
        
    # Define positive and negative rotation matrices:
    pos_R = define_rotation_matrix(angle_abs, axis=axis)
    neg_R = define_rotation_matrix(-angle_abs, axis=axis)
    
    # Multiply reference vector by positive and negative rotation matrices:
    pos_Rt = np.matmul(pos_R, ref)
    neg_Rt = np.matmul(neg_R, ref)
    
    # Round things off:
    pos_Rt = np.round(pos_Rt, decimals=5)
    neg_Rt = np.round(neg_Rt, decimals=5)
    tgt = np.round(tgt, decimals=5)
    
    # Check whether positive or negative rotation yields target vector:
    is_pos = np.all(np.abs(pos_Rt-tgt)<1e-3)
    is_neg = np.all(np.abs(neg_Rt-tgt)<1e-3)
    
    # Return an error if neither positive nor negative rotation about specified
    # axis results in target vector:
    if not is_pos and not is_neg:
        raise AssertionError('Neither positive nor negative rotation about axis {} results in specified target vector; please ensure rotation is purely around specified major axis.'.format(axis))
    
    if is_pos:
        signed_angle = np.degrees(angle_abs)
    elif is_neg:
        signed_angle = -1*np.degrees(angle_abs)

    return signed_angle 



def stitch_rsvps(trial):

    # Initialize stuff:    
    n_rsvp_slots = trial.shape[0] 
    PSTH = []
    Bins = []
    
    # Get data, bins for 0-th RSVP slot (including prestim baseline):
    r0_start_time = trial.iloc[0].psth_bins[0]
    r0_stop_time = trial.iloc[0].dur/1000
    [r0_start_idx, r0_stop_idx ] = time_window2bin_indices([r0_start_time, r0_stop_time], trial.iloc[0].psth_bins)
    r0_dat = trial.iloc[0].psth[:, r0_start_idx:r0_stop_idx+1]
    r0_psth_bins = deepcopy(trial.iloc[0].psth_bins[r0_start_idx:r0_stop_idx+1])    
    PSTH.append(r0_dat)
    Bins.append(r0_psth_bins)
    
    for i in np.arange(n_rsvp_slots)[1:]:
        
        curr_start_time = 0 # < ASSUMING NO ITI! Might eventually want to relax assumption
        curr_stop_time = trial.iloc[i].dur/1000 # < Stop time is just stim duration
        curr_psth_bins_all = deepcopy(trial.iloc[i].psth_bins)
        [start_idx, stop_idx] = time_window2bin_indices([curr_start_time, curr_stop_time], curr_psth_bins_all)
        curr_dat = trial.iloc[i].psth[:, start_idx:stop_idx+1]
        if i == np.arange(n_rsvp_slots)[-1]:
            offset = 2 # want last bin to be bracketed by trailing bin edge; these are supposed to be bin edges, i.e. including both the lower bound of the first bin and the upper bound of the last bin; so bin array should be one element longer than number of bins in PSTH array
        else:
            offset = 1
        curr_psth_bins_truncated = deepcopy(trial.iloc[i].psth_bins[start_idx:stop_idx+offset]) 
        curr_psth_bins_truncated_offset = curr_psth_bins_truncated +  np.cumsum(trial.iloc[0:i].dur/1000).values[-1]
        curr_psth_bins_truncated_offset = np.round(curr_psth_bins_truncated_offset, decimals=4) # deal with some rounding errors
    
        PSTH.append(curr_dat)        
        Bins.append(curr_psth_bins_truncated_offset)
    
    PSTH = np.concatenate(PSTH, axis=1)
    Bins = np.concatenate(Bins)
    
    return PSTH, Bins



def define_rotation_matrix(angle, axis=1):
    """
    Define 3D rotation matrix around a single cardinal axis. 

    Parameters
    ----------
    angle : float
        Angle by which output matrix should rotate any vector it multiplied.
        
    axis : int, optional
        Cardinal axis around which output matrix should rotate any vector it
        multiplied. The default is 1.

    Returns
    -------
    R : numpy.ndarray
        Rotation matrix .

    """
    if axis == 1:
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
            ])
    
    return R



def add_rsvp0_inds(inds):

    unq = np.unique(inds[:,0])
    append = np.array([unq, np.zeros(len(unq))]).T
    inds = np.concatenate([inds, append], axis=0)    
    inds = remove_duplicate_rsvp_indices(inds)

    return inds    



def split_inds(inds, n_splits=2):
    """
    Split RSVP indices into partitions. 

    Parameters
    ----------
    inds : array-like
        s-by-2 array of trial/RSVP indices, where s is the number of indiviudal
        stimulus presentations. Col 0: trial number, col 1: RSVP number. 
        
    n_splits : int, optional
        Number of partitions to split RSVP indices into. The default is 2.

    Returns
    -------
    splits : list
        n_splits element list. Each element is a p_i-by-2 array, where p_i is
        the number of individual stimulus presentations in the i-th split. Note
        the number of stimulus presentations per split can be different if 
        n_splits does not evenly divide s. 

    """
    
    np.random.shuffle(inds)
    splits = np.array_split(inds, n_splits,axis=0)
    
    return splits



def split_grouped_inds(grouped_inds, n_splits=2):
    """
    Split multiple groups of RSVP indices into partitions. 

    Parameters
    ----------
    grouped_inds : list
        g-element list of arrays, where g is the number of trial groups. Each
        element is s_i-by-2, where s_i is the number of individual stimulus
        presentations in the i-th trial group. For each array in grouped_inds,
        col 0 is the trial number and col 1 is the RSVP number. 
    
    n_splits : int, optional
        Number of partitions to split each trial group into. The default is 2.

    Returns
    -------
    splits : list
        List of lists of arrays. Each inner list corresponds to a split. Each
        array within each inner list corresponds to one partition of one trial
        group. 

    """
    
    splits = []
    for s in np.arange(n_splits):
        splits.append([])
    
    for i, group in enumerate(grouped_inds):
        curr_splits = split_inds(group, n_splits)
        for j, split in enumerate(curr_splits):
            splits[j].append(split)
        
    return splits



def round_ud(x, decimals=2, direction='up'):

    if direction == 'up':
        func = np.ceil
    elif direction == 'down':
        func = np.floor
    else:
        raise ValueError('Invalid rounding direction; please specify ''up'' or ''down''.')
    
    y = x*(10**decimals) 
    z = func(y)
    xhat = z/(10**decimals)

    return xhat    
