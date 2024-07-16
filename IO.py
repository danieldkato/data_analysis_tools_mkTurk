import os
import sys
import time
import pathlib as Path
import h5py
import pickle
import warnings
import re
import numpy as np
import pandas as pd
from itertools import product
from data_analysis_tools_mkTurk.utils_meta import find_channels, get_recording_path, get_coords_sess, get_all_metadata_sess
from data_analysis_tools_mkTurk.stim_info import filter_stim_trials, expand_classes, get_class_trials, create_trial_df, create_stim_idx_mat, reverse_lookup_rsvp_stim, session_dicts_2_df, sess_meta_dict_2_df
from data_analysis_tools_mkTurk.npix import get_site_coords
from data_analysis_tools_mkTurk.general import time_window2bin_indices, remove_duplicate_rsvp_indices, rsvp_from_df
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError:
    warnings.warn('Failed to import analysis_metadata module.')



def ch_dicts_2_h5(base_data_path, monkey, date, preprocessed_data_path, channels=None, 
    chunk_size=100, dtype=float, save_output=False, output_directory=None):
    """
    Combine pickled dicts of single-channel PSTHs into single HDF5. 

    Parameters
    ----------
    base_data_path : str
        Path to directory where raw data files are saved. One level above 
        monkey-level directories.
    
    monkey : str
        Monkey name.
    
    date : str
        Session date, formatted <yyyymmdd>.
    
    preprocessed_data_path : str 
        Path to where preprocessed data files (e.g. 'ch<iii>_psth_stim') are saved. 
        Must contain file named 'data_dict_<sess>', where <sess> is the name of 
        the directory immediately containing the raw data for the session. .
    
    channels : array-like, optional
        Array of channels indices to include data from. The default is None.
    
    chunk_size : int, optional
        Number of trials to include in a singl HDF5 chunk. Can have up to 2-3x
        impact on read/write speeds. The default is 100.
    
    dtype : type, optional
        Type data should be saved as. The default is float.
    
    save_output : bool, optional
        Whether to save output to disk. The default is False.
    
    output_directory : str, optional
        Path to directory where HDF5 should be saved. The default is None.

    Returns
    -------
    trial_info : dict
        Dictionary defining session metadata.
    
    spike_counts : numpy.ndarray
        Slab of spike count data. c-by-b-by-t-by-r, where c is the number of 
        included channels, b is the *maximum* number of time bins per stimulus
        presentations, t is the number of trials in the session, and r is the
        *maximum* number of RSVP stimuli per trial within the session. For 
        stimuli with a < b time bins b, only the first a columns will havve 
        numberic values; all further columns will be nan. Similarly, for any 
        trial with fewer than q < r RSVP stimuli, only the first q slices for
        the corresponding trial will have numeric values; all other slices from
        the same trial will be all nan.
        
    In addition to the above formal returns, saves to disk an HDF5 with the 
    following datasets/attributes:
        
        data : dataset
            Same as spike_counts above. 
            
        stim_inds : dataset
            t-by-r matrix of indices into ordered list of stim_ids. i.e., the 
            i,j-th element of stim_inds is k iff the k-th stim_id was presented
            on trial i, RSVP stim j. 
            
        stim_ids : attribute
            Ordered list of stim_id strings
            
        scenefile_by_stim_mat : attribute
            s-by-f matrix, where s is the number of unique stimuli and f is the
            number of unique scenefiles. i,j True iff stim i is included in the
            j-th scenefile. 

    """
        
    # Define default preprocessed data path if necessary:
    if preprocessed_data_path is None:
        preprocessed_data_path = get_recording_path(base_data_path, monkey, date, depth=4)[0]
    
    # Load metadata for current session:
    recording_dir = get_recording_path(base_data_path, monkey, date, depth=4)[0].split(os.sep)[-1]
    sess_meta, scenefile_meta, stim_meta = get_all_metadata_sess(preprocessed_data_path)
    sess_meta_df = sess_meta_dict_2_df(sess_meta)
    stim_ids = list(sess_meta.keys()) # < Get list of all individual stimulus conditions
    data_dict_path = os.path.join(preprocessed_data_path, 'data_dict_' + recording_dir)
    data_dicts = pickle.load(open(data_dict_path, 'rb'))
    D = [data_dicts[x] for x in np.arange(len(data_dicts))]
    
    # Find widest range of PSTH bins (these can differ between stimulus 
    # within a session conditions)
    n_bins_per_stim = [stim_meta[x]['n_bins'] for x in stim_ids]
    max_n_bins = max(n_bins_per_stim)
    longest_stim = np.where([stim_meta[x]['n_bins'] == max_n_bins for x in stim_ids])
    example_long_stim = stim_ids[longest_stim[0][0]]
    psth_bins = stim_meta[example_long_stim]['psth_bins']    
    
    # Create dataframe of all trial parameters: 
    trial_params_df = session_dicts_2_df(D)
    n_rows = trial_params_df.shape[0]
    trial_params_df['idx_merge'] = np.arange(n_rows)
    trial_params_df = trial_params_df.set_index('idx_merge')
    
    # Add a few general parameters to trial_params_df:
    # TODO: think about adding following parameters as well:
    # From stim_meta (one value per dict): iti_dur, t_before, t_after 
    # From sess_meta: reward, reward_dur
    monkey_col = [monkey] * n_rows 
    date_col = [date] * n_rows
    reward_bool = sess_meta_df['reward_bool']
    extra_params_df = pd.DataFrame({'monkey' : monkey_col, 
                                    'date' : date_col, 
                                    'reward_bool' : reward_bool})
    extra_params_df['idx'] = np.arange(n_rows) # < Only need this temporarily to concatenate dataframes
    extra_params_df = extra_params_df.set_index('idx')
    trial_params_df = pd.concat([extra_params_df, trial_params_df], axis=1)
    
    # Try to get paths to saved images:
    trial_params_df = find_im_full_paths(trial_params_df, base_data_path)
        
    # Copy general timing params to own dict as formal return:
    bin_width = stim_meta[stim_ids[0]]['binwidth'] # < Hack; assuming (probably safely) that same for all stim
    t_before = stim_meta[stim_ids[0]]['t_before'] # < Hack; assuming (probably safely) that same for all stim
    t_after = stim_meta[stim_ids[0]]['t_after'] # < Hack; assuming (probably safely) that same for all stim
    trial_info = {}
    trial_info['psth_bins'] = psth_bins
    trial_info['binwidth'] = bin_width
    trial_info['t_before'] = t_before
    trial_info['t_after'] = t_after
    #trial_info['trials'] = trial_df
   
    # Create s-by-g matrix specifying which stimulus ids are associated with which 
    # scenefiles, where s is the number of individual stimulus conditions and g
    # is the number of scenefiles sampled in current session; i,j-th element is
    # True if and only if j-th element sampled from i-th scenebag (note same stim
    # can be sampled from multiple scenebags, i.e. there can be more than one
    # True entry per column):
    print('Generating dataframe of stimulus conditions...')
    stim_indices = create_stim_idx_mat(sess_meta)
    scenefiles = [str(x) for x in list(scenefile_meta.keys())]
    scenefile_mat = np.array([[x in scenefile_meta[y]['stim_ids'] for y in scenefiles] for x in stim_ids])
    scenefile_mat = scenefile_mat.T 
   
    # Get coordinates of different sites:
    coords_df = get_site_coords(base_data_path, monkey, date, spacing=20, tip_length=175) # < Hard-coded for now
    coords_df = coords_df.sort_values('depth', ascending=False) # < Assume that channels have already been sorted by depth (lower channel # > deeper, higher channel # > more superficial)
    coords_df = coords_df.drop(columns='channel') # < Drop absolute channel IDs from IMRO table, which will conflict with new channel indices ordered by depth
   
    
    # Initialize data array:
    if channels is None:
        channels = find_channels(preprocessed_data_path)
    n_bins = len(psth_bins) - 1
    n_trials = np.max(trial_params_df['trial_num']) + 1
    n_rsvp = 3 # < Assuming 3 RSVP stim per trial 
    spike_counts = np.empty((len(channels), max_n_bins, n_trials, n_rsvp)) 
    spike_counts[:] = np.nan

    
    # Iterate over channels:
    input_files = []
    for cx, channel in enumerate(channels):
        
        print('Loading data for channel {} of {}...'.format(cx+1, len(channels)))
        
        # Load data for current channel:
        stim_fname = 'ch{}'.format(str(channel).zfill(3)) + '_psth_stim'
        fullpath = os.path.join(preprocessed_data_path, stim_fname)
        curr_ch_dict = pickle.load(open(fullpath,'rb')) 
        input_files.append(fullpath)
            
        # Iterate over stimulus conditions:
        for sx, stim_id in enumerate(stim_ids):
            
            curr_stim_n_trials = curr_ch_dict[stim_id].shape[0]
            
            # Iterate over presentations of current stimulus condition:
            for px, presentation in enumerate(sess_meta[stim_id]['trial_num']):
                
                curr_trial_num = sess_meta[stim_id]['trial_num'][px]
                curr_rsvp_num = sess_meta[stim_id]['rsvp_num'][px]
                curr_data = curr_ch_dict[stim_id][px,:]
                
                # Write current trial data to HDF5. IMPORTANT NOTE!!: This code 
                # currently assumes that even stimulus conditions of different durations 
                # have the same t_before. TODO: relax this assumption by determining 
                # start_idx programmatically. 
                start_idx = 0
                spike_counts[cx, start_idx:start_idx+len(curr_data), curr_trial_num, curr_rsvp_num] = curr_data
     
    # Save output if requested: 
    if save_output:
        
        # Set/create output directory if necesary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        if not os.path.exists(output_directory):
            Path.Path(output_directory).mkdir(parents=True, exist_ok=True)
            
        output_path = os.path.join(output_directory, 'all_psth.h5'.format(monkey, date)) 
        
        print('Saving HDF5 to disk...')
        with h5py.File(output_path, 'w') as f:
            #dset = f.create_dataset('data', data=spike_counts, dtype='int32')
            
            # Define chunk size:
            if chunk_size is True:
                spike_chunks = True
                stim_id_chunks = True
            elif chunk_size is None:
                spike_chunks = None
                stim_id_chunks = None
            else:
                spike_chunks = (spike_counts.shape[0], spike_counts.shape[1], chunk_size, spike_counts.shape[3])
                stim_id_chunks = (chunk_size, stim_indices.shape[1])
            
            # Create dataset containing actual spike counts:
            dset = f.create_dataset('data', data=spike_counts, dtype=dtype, rdcc_nbytes=8*(10**9)*3, chunks=spike_chunks)
            #dset.attrs['trial_df'] = trial_df
            
            # Write scenefile-by-stim_id boolean matrix specifying which stim
            # came from which scenefiles:
            scenefile_lookup = f.create_dataset('stim_indices', data=stim_indices, dtype=dtype, chunks=stim_id_chunks)
            
            # Write full dataframe of trial parameters:
            trial_params_df_out = trial_params_df.copy()
            trial_params_df_out = standardize_col_types(trial_params_df_out)
            trial_params_df_out.to_hdf(output_path, 'trial_params', 'a', format='table')
            
            #"""
            # Write truncated dataframe of select trial parameters:
            short_cols = ['monkey', 'date', 'trial_num', 'rsvp_num', 'stim_id', 'scenefile']
            trial_params_short = trial_params_df[short_cols] 
            trial_params_short = trial_params_short.rename(columns={'stim_info_short' : 'stim_id'})
            trial_params_short.to_hdf(output_path, 'trial_params_short', 'a', format='fixed')
            #"""
            
            # Write channel coordinates:
            coords_df.to_hdf(output_path, 'site_coordinates', 'a', format='fixed')
            
            # Write metadata for session:
            f.attrs['psth_bins'] = psth_bins
            f.attrs['binwidth'] = bin_width
            f.attrs['t_before'] = t_before
            f.attrs['t_after'] = t_after
            f.attrs['scenefile_meta_path'] = os.path.join(preprocessed_data_path, 'ch383_psth_scenefile_meta') # < Hacky but should work for now
            f.attrs['stim_meta_path'] = os.path.join(preprocessed_data_path, 'stim_info_sess') # < Hacky but should work for now
            f.attrs['stim_ids'] = stim_ids
            f.attrs['scenefiles'] = scenefiles
            f.attrs['scenefile_by_stim_mat'] = scenefile_mat
            
            
        if 'analysis_metadata' in sys.modules:
            M = Metadata()
            for i in input_files:
                M.add_input(i)
            M.add_output(output_path)
            M.add_param('chunk_size', chunk_size)
            M.add_param('dtype', str(dtype))
            metadata_path = os.path.join(output_directory, 'chpsths_2_h5.json')
            write_metadata(M, metadata_path, get_checksum=False)
            
    return trial_info, spike_counts



def h5_2_trial_df(h5path, params='short'):
    """
    Get dataframe of trial paramters from HDF5 file of recording session. 

    Parameters
    ----------
    h5 : str
        Path to HDF5 file with same format as output of ch_dicts_2_h5().
    
    params : 'short' | 'all'
        Whether to include all trial parameters or just a subset in output 
        dataframe. If 'all', will include all trial parameters in 'trial_params'
        dataset of input HDF5 file. If 'short', will only include columns `trial_num`,
        `rsvp_num`, `stim_id`, and `scenefile`. 

    Returns
    -------
    trial_df : pandas.core.DataFrame
        Dataframe of trial/RSVP stim parameters. If `params` is set to 'all', 
        then columns will be the same as in 'trial_params' dataset of input HDF5
        file; else if `params` is set to 'short', will define only the following
        columns:
            
            trial_num : int
                Trial number.
                
            rsvp_num : int
                RSVP stimulus number within trial.
                
            stim_id : str
                Stimulus description. 
                
            scenefile : str
                Name of scenefile stimulus was drawn from. 

    """
    
    
    # Load trial parameters:
    if params == 'short':
        trial_df = pd.read_hdf(h5path, 'trial_params_short', 'r')
        #trial_df = pd.read_hdf(h5path, 'trial_params', 'r', columns=['trial_num', 'rsvp_num', 'scenefile'])
    elif params == 'all':
        trial_df = pd.read_hdf(h5path, 'trial_params', 'r')
    
    n_rows = trial_df.shape[0]
    
    # Add psth bins:
    with h5py.File(h5path, 'r') as f:
        psth_bins = f.attrs['psth_bins']
    B = [psth_bins] * n_rows
    
    # Add path to source file:
    P = [h5path] * n_rows
    
    trial_df['psth_bins'] = B
    trial_df['source_path'] = P
    
    #print('Returning dataframe...')
    return trial_df



def h5_2_dat_array_rsvp(h5, trials=None, channels=None, time_window=None, dset_name='data'):
    """
    Retrieve PSTHs for specific stimulus presentations, indexed by trial number
    and RSVP stim number. 

    Parameters
    ----------
    h5 : h5py._hl.files.File
        HDF5 file object, same format as output of ch_dicts_2_h5().
    
    trials : array-like | None
        s-by-2 array, where s is the number of stimulus presentations to retrive
        data for. Col 0: trial number, col 1: RSVP stim number. If None, will
        include data for all trials and RSVP stim. 
    
    channels : array-like | None
        Indices of channels to get data for. If None, will include data for all
        channels.
        
    time_window : array-like | None
        2-element list or array. First element is *index* of first time bin of
        continuous time window to get data for, second element is index of last
        time bin. If None, will include data for entire peristim epoch included
        in each slice of source H5 dataset.
    
    dset_name : str
        Name of HDF5 dataset to retrieve data from.

    Returns
    -------
    slices : numpy.ndarray
        c-by-b-by-s, where c is the number of channels included in the input 
        HDF5 file, b is the maximum number of time bins per trial, and s is the
        number of requested stimulus presentations.
        
    # TODO: Don't see any reason not to also filter by channel and time here. 

    """
    
    # Define default trial, channel, and time bin ranges if any are set to None:
    if trials is None:
        n_trials = h5[dset_name].shape[2]
        n_rsvp_stim = h5[dset_name].shape[3]
        all_trials = np.arange(n_trials)
        all_rsvp = np.arange(n_rsvp_stim)
        trials = np.arange(list(product(all_trials, all_rsvp)))
    if channels is None:
        n_chan = h5[dset_name].shape[0]
        channels = np.arange(n_chan)
    if time_window is None:
        n_bins = h5[dset_name].shape[1]
        window = [0, n_bins]
    
    # Define requested trial range:
    min_trial = min(trials[:,0])
    max_trial = max(trials[:,0])

    # Pre-fetch data from requested trial range:
    print('Pre-fetching PSTHs from HDF5...')
    start_load = time.time()
    data = h5[dset_name][:, :, min_trial:max_trial+1, :]
    stop_load = time.time()
    print('... done.')
    print('Duration={} minutes'.format((stop_load-start_load)/60))

    # Offset trials by min trial:
    trials_offset = trials
    trials_offset[:,0] = trials_offset[:,0] -  min_trial
    
    # Define boolean filter for which slices to grab:
    B = np.empty((data.shape[2], data.shape[3])).astype(bool)
    B[:] = False
    B[trials[:,0], trials[:,1]] = True
    
    # Grab specific slices:
    print('Fancy slicing numpy array...')
    start_slice= time.time()
    slices = data[:, :, B]
    slices = slices[channels, time_window[0]:time_window[1], :]
    stop_slice= time.time()
    print('... done.')
    print('Duration={} minutes'.format((stop_slice-start_slice)/60))

    # Hack; input HDF5s are saved as int32 to reduce space, I/O time, but this 
    # has effect of turning nan into -2*10^9; convert back to nan here:
    slices = slices.astype(float)
    slices[slices<-2e9] = np.nan 

    return slices



def h5_2_df(h5_path, trials=None, channels=None, time_window=None, dset_name='data', trial_params='short'):
    """
    Populate dataframe of trial info with spike counts from HDF5 file. 

    Parameters
    ----------    
    h5 : str 
        Path to HDF5 file with same format as output of ch_dicts_2_h5().

    trials : array-like | pandas.core.DataFrame | None
        Trials to fetch PSTHs for. If an array, must be s-by-2, where s is the 
        number of stimulis presentations to retrieve PSTH data for. Col 0: trial number, 
        col 1: RSVP stim number.
        
        If a pandas dataframe, will retrieve PSTH data for all trials included
        in dataframe. Should be same format as output of h5_2_trial_df(). Must
        at least define columns 'trial_num' and 'rsvp_num'. Note that 'trial_num'
        should contain *absolute* trial numbers.
        
        If None, will retrieve PSTH data for all trials and RSVP stim. 

    channels : array-like | None
        Indices of channels to get data for. If None, will include data for all
        channels.
        
    time_window : array-like | None
        2-element list or array. First element is time of first bin of continuous 
        window to get data for relative to stim onset (in seconds), second 
        element is time of last bin. If None, will include data for entire peristim 
        epoch included in each slice of source H5 dataset.
    
    dset_name : str
        Name of HDF5 dataset to retrieve data from.

    Returns
    -------
    trial_df : pandas.core.DataFrame        
        Dataframe of trial/RSVP stim parameters. Defines following columns:
            
            trial_num : int
                Trial number.
                
            rsvp_num : int
                RSVP stimulus number within trial.
                
            stim_id : str
                Stimulus description.
                
            psth : numpy.ndarray
                c-by-b, where c is the number of channels included in analysis 
                and b is the maximum number of time bins per trial. 

    """
    
    # Get some channel and timing metadata:
    with h5py.File(h5_path, 'r') as h5:
        
        # Asisgn default channel range if necessary:
        if channels is None:
            n_chan = h5[dset_name].shape[0]
            channels = np.arange(n_chan)
        
        # Convert requested peristim time window to indices:
        if time_window is not None:    
            psth_bins = h5.attrs['psth_bins']
            bin_indices = time_window2bin_indices(time_window, psth_bins)
        
        # Or assign default time window if not specified:
        elif time_window is None:
            n_bins = h5[dset_name].shape[1]
            bin_indices = [0, n_bins]

    # If passed `trials` argument is not already a dataframe, create df of trial info:
    if type(trials) != pd.core.frame.DataFrame:
   
        print('Fetching trial parameters...')
        tdf_start = time.time()
        trial_df = h5_2_trial_df(h5_path, params=trial_params)
        tdf_stop = time.time()
        print('... done ({} sec).'.format(tdf_stop - tdf_start))
    
        # Furthermore, if `trials` is a non-empty array, select the trials 
        # specified therein:
        if trials is not None:
            trial_df = rsvp_from_df(trial_df, trials)
            
    # Otherwise, if `trials` was already a dataframe: 
    elif type(trials) == pd.core.frame.DataFrame:
        trial_df = trials
        
    # Get indices from appropriate columns:
    trial_df = trial_df.sort_values(by=['trial_num', 'rsvp_num'])
    trials = np.array([trial_df['trial_num'], trial_df['rsvp_num']]).T
    
    # Retrieve requested PSTH data:
    with h5py.File(h5_path, 'r') as h5:
        slices = h5_2_dat_array_rsvp(h5, trials=trials, channels=channels, time_window=bin_indices)
        slices = np.transpose(slices, axes=[2, 0, 1])
        slice_list = list(slices)
    
    # Write PSTSHs back into dataframe:
    trial_df.insert(trial_df.shape[1], 'psth', slice_list, True)
    
    return trial_df



def trim_rsvp_stim(df, h5, stim_dur=3.0):
    
    # Find start index:
    psth_bins = h5.attrs['psth_bins']
    rsvp_start_idx = min(np.where(psth_bins>=0)[0])     
    
    # Find stop index:
    rsvp_stop_idx = max(np.where(psth_bins<=stim_dur)[0]) # < Extreme hack, assuming fixed stim duration across trials, need to fix this!
    
    # Extract array of PSTHs, select only stimulus time:
    data = np.array(list(df.psth))
    data = np.transpose(data, axes=[1, 2, 0])
    data = data[:, rsvp_start_idx, rsvp_stop_idx, :]
    
    # Write back to df:
    df.psth = np.transpose(data, axes=[2,0,1])
    
    return df



def h5_2_psths_by_class(h5, classes):
    """
    

    Parameters
    ----------
    h5 : str
        Path to HDF5 file of same format as that returned by psths2slab().
    
    classes : list
        g-element list, where g is the number of stimulus 'classes.' A stimulus 
        'class' may consist of one or more individual stimuli (e.g. 'novel',
        'familiar', etc.). Each element should itself be a list of individual 
        stim ids comprising the corresponding 'class'.

    Returns
    -------
    data_by_class : list
        List of g numpy arrays, where g is the number of elements in the `classes`
        input. Each array is c-by-b-by-t_i, where c is the number of channels,
        b is the number of time bins per stimulus presentation, and t_i is the
        overall number of stimulus presentations for stimulus 'class' i. Note 
        that c and b are constant across arrays, but t_i may be different 
        for different stimulus classes. 

    """
    
    # Create HDF5 file object:
    f = h5py.File(h5, 'r', rdcc_nbytes=8*(10**9)*3)
    #f = h5py.File(h5, 'r', rdcc_nbytes=2.5e9)
    
    # Get scenefile metadata for session:
    scenefiles= f.attrs['scenefiles']
    all_stim_ids = f.attrs['stim_ids']
    scenefile_by_stim_mat = f.attrs['scenefile_by_stim_mat']
    stim_indices = f['stim_indices'][:,:]
    # < Hack; input HDF5s are saved as int32 to reduce space, I/O time, but this has effect of turning nan into -2*10^9; convert back to nan here
    stim_indices = stim_indices.astype(float)
    nanindices = np.where(stim_indices<-2e9)
    stim_indices[nanindices[0], nanindices[1]] = np.nan
    
    # Read data for requested classes:
    indices_by_class = []
    for cx, curr_class in enumerate(classes):
        
        curr_class_indices = []
        
        # Iterate over individual stimulus conditions in current class:
        for sx, stim_id in enumerate(curr_class):
            
            # Retrieve data for current stimulus condition:
            curr_stim_indices = reverse_lookup_rsvp_stim(stim_id, all_stim_ids, stim_indices)
            curr_class_indices.append(curr_stim_indices)
            
        # Average across presentations:                        
        indices_by_class.append(curr_class_indices)

    # Get min and max indices:
    min_trial = min([   min(  [min(np.where(x)[0]) for x in y]  ) for y in indices_by_class   ])
    max_trial = max([   max(  [max(np.where(x)[0]) for x in y]  ) for y in indices_by_class   ])
    
    # Extract data from trials within range:
    print('Loading data...')
    start_load = time.time()
    data = f['data'][:, :, min_trial:max_trial, :]
    f.close()
    stop_load = time.time()
    print('... done.')
    print('Duration={} minutes'.format((stop_load-start_load)/60))
    
    # Iterate over classes extracting data:
    data_by_class = []
    for curr_class_indices in indices_by_class:
        
        # Iterate over stimulus conditions:
        for sx, curr_stim_indices in enumerate(curr_class_indices):
            
            curr_stim_indices_offset = curr_stim_indices[min_trial:max_trial, :] # < apply offset
            curr_stim_data = data[:, :, curr_stim_indices_offset]
            curr_stim_data = curr_stim_data.astype(float)
            curr_stim_data[curr_stim_data<-2e9] = np.nan # < Hack; input HDF5s are saved as int32 to reduce space, I/O time, but this has effect of turning nan into -2*10^9; convert back to nan here
            if sx == 0:
                curr_class_data = curr_stim_data
            else:
                curr_class_data = np.concatenate((curr_class_data, curr_stim_data), axis=2)            

        data_by_class.append(curr_class_data)            
    
    return data_by_class



def standardize_col_types(df):
    # Find any dataframe columns of more than one type (which causes an error
    # when saving with pd.to_hdf) then take appropriate steps to make all of one
    # type
    
    # Find columns with more than one datatype:
    cols = df.columns
    f = lambda y : len(np.unique([str(type(x)) for x in y])) # Define function for counting how many datatypes there are in a column
    typenums = np.array([f(df[c]) for c in cols])
    multitype_col_inds = np.where(typenums > 1)
    multitype_cols = cols[multitype_col_inds]
    
    # Iterate over columns with multiple types:
    for col in multitype_cols:
        
        print(col)
        
        # Get types in current column:
        curr_types = np.unique([str(type(x)) for x in df[col]])
    
        # Define specific fixes for different combinations of types; this part a bit hack-y:
        if "<class 'float'>" in curr_types and "<class 'str'>" in curr_types:
            
            # If all floats are NaN, make everything string:
            floats = np.where([type(x)==float for x in df[col]])[0]
            nans = np.where(df[col].isna())[0]
            if len(floats)==len(nans) and np.all(floats==nans):
                df[col]= df[col].astype(str)
    
            # Otherwise, convert everything to float:
            else: 
                df.loc[df[col]=='', col] = np.nan # Convert any empty strings to nan
                df[col] = df[col].astype(float)
                
    return df



def find_im_full_paths(trial_params_df, base_data_path):
    
    # Try to find base image directory:
    base_imdir = find_saved_imgs_dir(trial_params_df)
    
    # If unsuccessful, raise warning and return original dataframe...:
    if base_imdir is None:
        warnings.warn('Base image directory not found for input dataframe, will not add full image paths.')
     
    # ... otherwise, if base image directory successfully discovered, find paths to PNGs:
    else:
        
        # base_imdir is relative to axon root; convert to relative to local machine: 
        base_imdir_local = os.path.join(base_data_path+'\\', os.sep.join(base_imdir.split('\\')[5:])) 
        
        # Get paths to scenefile-specific image directories relative to local machine: 
        j = lambda x : os.path.join(base_imdir_local, x.scenefile.split('/')[-1][:-5])
        trial_params_df['sfile_imdir'] = trial_params_df.apply(j,axis=1)
        
        # HACK; data_dicts appear to include a mistake where stim indices are 
        # off by some offset; correct here:
        sfile_imdirs = np.unique(trial_params_df.sfile_imdir)
        for s in sfile_imdirs:
            curr_rows = trial_params_df.sfile_imdir == s
            trial_params_df.loc[curr_rows, 'stim_idx'] = trial_params_df.loc[curr_rows, 'stim_idx'] - min(trial_params_df.loc[curr_rows, 'stim_idx'])
        
        # Create dataframe of unique images:
        unique_images = trial_params_df[['sfile_imdir', 'stim_idx']].drop_duplicates()
        unique_images = unique_images.sort_values(by=['sfile_imdir', 'stim_idx'])
        
        # Get all PNGs in base_imdir:
        all_pngs = []
        for s in sfile_imdirs:
            pngs = [os.path.join(s,x) for x in os.listdir(s) if re.search('_index\d+.png',x) is not None]
            all_pngs += pngs
            
        # Find saved PNGs matching rows of unique_images:
        find_matching_pngs = lambda x : [y for y in all_pngs if x.sfile_imdir in y and '_index{}.png'.format(x.stim_idx) in y]
        matching_pngs = unique_images.apply(find_matching_pngs, axis=1)
        matching_pngs = [y[0] if len(y)>0 else None for y in matching_pngs] # < Take first matching PNG if it exists, otherwise use None; ASSUMING FIRST MATCHING PNG IS THE SAME AS OTHERS
        unique_images['img_full_path'] = matching_pngs
        
        # Merge unique images with full paths to trial_params_df:
        trial_params_df = pd.merge(trial_params_df, unique_images, on=['sfile_imdir', 'stim_idx'], how='outer')

    return trial_params_df



def find_saved_imgs_dir(trial_params):
    
    base_dir = os.path.join('mnt', 'smb', 'locker', 'issa-locker', 'Data')
    
    # Get all unique scenefiles in current experiment:
    if type(trial_params) == list:
        sfiles = np.unique([x['scenefile'] for x in trial_params])
    elif type(trial_params) == pd.core.frame.DataFrame:
        sfiles = np.unique(trial_params['scenefile'])
        
    # Get monkey names:
    mnames = [x.split('/')[3] for x in np.unique(sfiles)]
    
    # Verify that all scenefiles refer to the same monkey:
    if np.all([x==mnames[0] for x in mnames]):
        monkey = mnames[0]
    # If more than one monkey detected, raise warning and return None
    else:
        warnings.warn('More than one monkey name discovered among scenefile paths.')
        return None
    
    # Try to get stim set number:
    stim_set_regex = 'neural_stim_\d+_'
    h = lambda x : re.search(stim_set_regex, x)
    stim_sets = [h(x).group()[-2] for x in sfiles if h(x) is not None]
        
    # Verify that all scenefiles of the same stim set:
    if np.all([x==stim_sets[0] for x in stim_sets]): 
        stim_set = stim_sets[0]
    # Otherwise, raise warning and return None:
    else:
        warnings.warn('More than one stim set discovered among scenefile paths.')
        return None
        
    # Try to get experiment ID from scenefile ending 'ABCDEFGHIJUVWXYZ_<ID>.json'
    exp_regex = '[A-Z]{5,}_\d{2,2}.json'
    f = lambda x : re.search(exp_regex, x)
    exp_ids = [f(x).group()[-7:-5] for x in sfiles if f(x) is not None]
    
    # Verify that all exp_ids are the same; if not, raise warning and return None
    if np.all([x==exp_ids[0] for x in exp_ids]):
        exp_id = exp_ids[0]
    else: 
        warnings.warn('More than one experiment ID discovered among scenefile paths.')
        return None
        
    saved_imgs_dirname = 'Saved_Images_{}_neural_stim_{}_{}'.format(monkey, stim_set, exp_id)
    saved_imgs_base_dir = os.path.join(base_dir, monkey, 'Saved_Images', saved_imgs_dirname)     
    
    return saved_imgs_base_dir
        