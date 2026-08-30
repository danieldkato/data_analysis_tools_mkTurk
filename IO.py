import os
import sys
import time
import socket
import hashlib
from datetime import datetime
from pathlib import Path
import glob
import h5py
import glob
import pickle
import json
import warnings
import re
import numpy as np
import pandas as pd
import json
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from .utils_meta import find_channels, find_units, get_recording_path, get_coords_sess, get_all_metadata_sess, resolve_ks_h5_path
from .stim_info import filter_stim_trials, expand_classes, get_class_trials, create_trial_df, create_stim_idx_mat, reverse_lookup_rsvp_stim, session_dicts_2_df, sess_meta_dict_2_df
from .npix import get_sess_metadata_path, extract_imro_table, get_site_coords, h5_2_ch_meta
from .spike_sorting.quality_metrics import build_unit_info_dfs
from .general import time_window2bin_indices, remove_duplicate_rsvp_indices, rsvp_from_df, abs2rel_ind
from mkutils_ddk.env import get_engram_drive
from mkanalysis.general import matches_any_predicate
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError:
    warnings.warn('Failed to import analysis_metadata module.')


# Suggested shared default — size to comfortably hold several chunks
# at the new (smaller) chunk size, not the old 24GB figure that assumed
# gigantic chunks:
_DEFAULT_RDCC_NBYTES = 512 * (2**20)   # 512 MB
_DEFAULT_RDCC_NSLOTS = 1_000_003       # prime, per h5py's recommendation (~100x number of chunks you expect to cache)


def ch_dicts_2_h5(base_data_path, monkey, date, preprocessed_data_path, channels=None,
    chunk_size=20, dtype=np.float32, save_output=False, fname='all_psth', output_directory=None,
    source='mua', bin_chunk_size=None, compress_data=False, trial_params_format='table'):
    """
    Combine pickled dicts of single-channel (or single-unit) PSTHs into single HDF5.

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
        Array of entity indices to include data from. When source='mua' these are
        SpikeGLX channel indices; when source='ks' these are Kilosort cluster ids.
        If None, the entities are discovered from the per-entity PSTH files in the
        appropriate directory. The default is None.

    source : 'mua' | 'ks', optional
        Which preprocessed PSTHs to combine. 'mua' uses per-channel files named
        'ch<nnn>_psth_stim' in preprocessed_data_path (fixed channel count;
        original behavior). 'ks' uses per-unit files named 'clu<nnn>_psth_stim'
        in <preprocessed_data_path>/kilosort4 (variable number of sorted units on
        axis-0 of the data slab). The default is 'mua'.

    chunk_size : int, optional
        Number of trials to include in a singl HDF5 chunk. Can have up to 2-3x
        impact on read/write speeds. The default is 100.
    
    dtype : type, optional
        Type data should be saved as. The default is float.
    
    save_output : bool, optional
        Whether to save output to disk. The default is False.
    
    output_directory : str, optional
        Path to directory where HDF5 should be saved. The default is None.

    bin_chunk_size : int, optional
        Number of time bins to include in a single HDF5 chunk for the `data`
        dataset. If None (the default — matches original behavior exactly),
        each chunk spans the full time-bin extent, as before. Passing a value
        smaller than the total number of bins lets partial reads restricted to
        a peristimulus sub-window actually touch fewer bytes on disk, instead
        of every chunk still spanning the entire recorded epoch regardless of
        what a caller asks for.

    compress_data : bool, optional
        Whether to apply 'lzf' compression to the `data` dataset (the
        `stim_indices` dataset is already compressed this way regardless).
        The default is False, matching original behavior exactly.

    trial_params_format : 'table' | 'fixed', optional
        PyTables storage format for the full `trial_params` table. The
        default 'table' matches original behavior exactly, but is dramatically
        slower to read back over a network filesystem (observed ~350x on one
        real 259-column session table) than 'fixed', since 'table' uses an
        indexed, per-dtype-block on-disk layout meant for partial/`where=`
        queries -- expensive to deserialize even for object/mixed-type
        columns fully read every time regardless. No code anywhere queries
        this table partially (`h5_2_trial_df` always reads it in full), so
        'fixed' -- which stores it as a single flat, quick-to-deserialize
        blob -- is safe to opt into with no functional loss.

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
        preprocessed_data_path = get_recording_path(Path(base_data_path), Path(monkey), date, depth=4)[0]
    
    pen_id = preprocessed_data_path.split(os.path.sep)[-1]

    # Load metadata for current session
    # For source='ks', fall back to the kilosort4 subdir for psth_stim_meta (ks-only sessions).
    stim_meta_dir = os.path.join(preprocessed_data_path, 'kilosort4') if source == 'ks' else None
    recording_dir = get_recording_path(Path(base_data_path), Path(monkey), date, depth=4)[0].split(os.sep)[-1]
    sess_meta, scenefile_meta, stim_meta = get_all_metadata_sess(preprocessed_data_path, stim_meta_dir=stim_meta_dir)
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
    
    # Apply offsets to stim_idx; recall if scenefile b follows scenefile a with
    # m images, then index of first image of scenefile b will be m, not 0:
    offsets_df = trial_params_df[['scenefile', 'stim_idx']].groupby('scenefile').min().reset_index()
    offsets_df = offsets_df.rename(columns={'stim_idx':'offset'})
    trial_params_df = pd.merge(trial_params_df, offsets_df, on=['scenefile'])
    trial_params_df['stim_idx'] = trial_params_df['stim_idx'] - trial_params_df['offset']
    trial_params_df['stim_idx'] = trial_params_df['stim_idx'].drop(columns='offset')
    
    # Try to retrieve THREEJS params directly from behavior files:
    rsvp_dframes_list = []
    behav_df = pd.DataFrame()
    sess_dirs = [x for x in os.listdir(os.path.join(base_data_path, monkey)) if pen_id in x]
    if len(sess_dirs) == 1 and os.path.exists(os.path.join(base_data_path, monkey, sess_dirs[0])):
        sess_dir = os.path.join(base_data_path, monkey, sess_dirs[0])
        behav_files = np.unique(trial_params_df.behav_file)

        # Iterate over behavior files:
        for b in behav_files:
            bpath = os.path.join(sess_dir, b+'.json')
            bfile = json.load(open(bpath, 'rb'))
            curr_scenefiles = bfile['TASK']['ImageBagsSample']
            
            # Iterate over scenefiles:
            for s, sfile in enumerate(curr_scenefiles):
                curr_sfile_df = pd.DataFrame()
                n_stim = bfile['SCENES']['SampleScenes'][s]['nimages']
                curr_sfile_df['stim_idx'] = np.arange(n_stim)
                dims = ['x', 'y', 'z']
                for dim in dims:
                    dat =  bfile['SCENES']['SampleScenes'][s]['CAMERAS']['camera00']['targetTHREEJS'][dim]
                    if len(dat) == n_stim:
                        curr_sfile_df['targetTHREEJS_'+dim] = dat
                    elif len(dat) == 1:
                        curr_sfile_df['targetTHREEJS_'+dim] = dat[0]*np.ones(n_stim)
                    else:
                        curr_sfile_df['targetTHREEJS_'+dim] = [None]*n_stim
                curr_sfile_df['scenefile'] = sfile
                curr_sfile_df['behav_file'] = b
                behav_df = pd.concat([behav_df, curr_sfile_df], axis=0)

            # Find which individual RSVP slots were completed vs. broken fixation during:
            curr_rsvp_dframe = find_complete_rsvp_slots(bfile)
            curr_rsvp_dframe['behav_file'] = b 
            rsvp_dframes_list.append(curr_rsvp_dframe) 

        rsvp_dframes = pd.concat(rsvp_dframes_list, axis=0)

    # Merge general trial parameters with THREEJS params from behavior files:
    trial_params_df = pd.merge(trial_params_df, behav_df, on=['scenefile', 'behav_file', 'stim_idx'], how='left')
    trial_params_df['trial_num'] = trial_params_df.trial_num.astype(int)

    # Merge general trial parameters with whether each RSVP slot was completed:
    trial_params_df = abs2rel_ind(trial_params_df, grouping_col='behav_file', idx_col='trial_num', rename_col=True)
    trial_params_df = pd.merge(trial_params_df, rsvp_dframes.rename(columns={'trial_num':'trial_num_rel'}), on=['behav_file', 'trial_num_rel', 'rsvp_num', 'stim_idx'])

    # Add a few general parameters to trial_params_df:
    # TODO: think about adding following parameters as well:
    # From stim_meta (one value per dict): iti_dur, t_before, t_after 
    # From sess_meta: reward, reward_dur
    trial_params_df['monkey'] = monkey
    trial_params_df['date'] = date
    trial_params_df = pd.merge(trial_params_df, sess_meta_df[['trial_num', 'rsvp_num', 'reward_bool', 't_on']], on=['trial_num', 'rsvp_num'])
    #trial_params_df['reward_bool'] = sess_meta_df.reward_bool
    #trial_params_df['t_on'] = sess_meta_df.t_on
    
    # Try to get paths to saved images:
    trial_params_df = add_im_full_paths(trial_params_df, base_data_path)

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
   
    # Get stereotaxic coordinates of zero point (where probe touches surface of brain) for current session:
    zero_coords = get_coords_sess(Path(base_data_path), Path(monkey), date)
    glx_meta_path = get_sess_metadata_path(Path(base_data_path), Path(monkey), date)
    if glx_meta_path is not None:
        if 'win' in sys.platform:
            glx_meta_path = '\\\\?\\' + glx_meta_path
        imro_tbl = extract_imro_table(glx_meta_path)
    else:
        imro_tbl = pd.DataFrame()
        warnings.warn('No .ap.meta file discovered for {} session {}.'.format(monkey, date))
        
    # Resolve which directory and per-entity filename prefix to read PSTHs from.
    # source='mua' reads per-channel files ('ch<nnn>_psth_stim') directly from
    # preprocessed_data_path; source='ks' reads per-unit files ('clu<nnn>_psth_stim')
    # from the kilosort4 subdir, where the number of entities (sorted units) varies
    # by session rather than being a fixed channel count.
    match source:
        case 'mua':
            psth_dir = preprocessed_data_path
            entity_prefix = 'ch'
        case 'ks':
            psth_dir = os.path.join(preprocessed_data_path, 'kilosort4')
            entity_prefix = 'clu'
        case _:
            raise ValueError("source must be 'mua' or 'ks', got {}".format(source))

    # Initialize data array:
    if channels is None:
        if source == 'mua':
            channels = find_channels(preprocessed_data_path)
        else:
            channels = find_units(psth_dir)

    # For single units, assemble the per-unit metrics tables (one row per unit,
    # raw metrics only) and align them to the unit order on axis-0 of the slab so
    # good-unit filtering and spatial lookups can be applied later from the H5
    # alone. Two tables by purpose: unit_quality (is_good_unit inputs + KSLabel)
    # and unit_spatial (probe location + amplitude). Missing metric files are
    # tolerated (filled with NaN) inside build_unit_info_dfs.
    unit_quality_df = None
    unit_spatial_df = None
    if source == 'ks':
        try:
            unit_quality_df, unit_spatial_df = build_unit_info_dfs(psth_dir)
            # Reindex to axis-0 unit order ('unit_id' index -> column):
            unit_quality_df = unit_quality_df.reindex(np.asarray(channels)).reset_index()
            unit_spatial_df = unit_spatial_df.reindex(np.asarray(channels)).reset_index()
        except Exception as e:
            unit_quality_df = pd.DataFrame()
            unit_spatial_df = pd.DataFrame()
            warnings.warn('Failed to build unit metrics tables for {} {}: {}'.format(monkey, date, e))

    n_bins = len(psth_bins) - 1
    n_trials = np.max(trial_params_df['trial_num']) + 1
    n_rsvp = len(trial_params_df.rsvp_num.unique())  
    spike_counts = np.empty((len(channels), max_n_bins, n_trials, n_rsvp)) 
    spike_counts[:] = np.nan
    
    # Iterate over entities (channels for source='mua', sorted units for source='ks'):
    entity_label = 'unit' if source == 'ks' else 'channel'
    input_files = []
    for cx, channel in enumerate(channels):

        print('Loading data for {} {} of {}...'.format(entity_label, cx+1, len(channels)))

        # Load data for current entity:
        stim_fname = '{}{}'.format(entity_prefix, str(channel).zfill(3)) + '_psth_stim'
        fullpath = os.path.join(psth_dir, stim_fname)
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
            Path(output_directory).mkdir(parents=True, exist_ok=True)
            
        output_path = os.path.join(output_directory, fname+'.h5') 
        
        print('Saving HDF5 to disk...')
        # NOTE: the h5py datasets/attrs and the pandas (PyTables) dataframes are
        # written in two separate phases. Calling DataFrame.to_hdf() while the
        # h5py handle below is still open would open the same file twice at once,
        # which fails on network filesystems (SMB/NFS) with an HDF5 file-lock
        # error ("unable to lock the file, errno = 11"). The h5py block therefore
        # closes before any to_hdf() call runs.
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
                # Time-bin chunk extent: full extent by default (original
                # behavior, unchanged unless a caller explicitly opts in via
                # `bin_chunk_size`), or a caller-specified sub-extent so that
                # a read restricted to a peristimulus sub-window can actually
                # touch fewer bytes on disk instead of every chunk spanning
                # the whole recorded epoch regardless of what's requested:
                if bin_chunk_size is None:
                    n_bin_chunk = spike_counts.shape[1]
                else:
                    n_bin_chunk = min(bin_chunk_size, spike_counts.shape[1])
                spike_chunks = (spike_counts.shape[0], n_bin_chunk, chunk_size, spike_counts.shape[3])
                stim_id_chunks = (chunk_size, stim_indices.shape[1])

            # Sanity-check chunk size before writing — warn if a single chunk is
            # going to be large enough to make the read-side cache moot:
            if spike_chunks not in (True, None):
                chunk_bytes = np.prod(spike_chunks) * np.dtype(dtype).itemsize
                if chunk_bytes > 64 * (2**20):  # 64 MB, adjust to taste
                    warnings.warn(
                        'HDF5 chunk size is {:.1f} MB (chunks={}). Consider a smaller '
                        '`chunk_size` for more efficient partial reads.'.format(
                            chunk_bytes / 2**20, spike_chunks))

            # Create dataset containing actual spike counts. `compress_data`
            # defaults to False, matching original (uncompressed) behavior
            # exactly unless a caller explicitly opts in:
            data_compression = 'lzf' if compress_data else None
            dset = f.create_dataset('data', data=spike_counts, dtype=dtype, chunks=spike_chunks, compression=data_compression)
            #dset.attrs['trial_df'] = trial_df

            # Write scenefile-by-stim_id boolean matrix specifying which stim
            # came from which scenefiles:
            scenefile_lookup = f.create_dataset('stim_indices', data=stim_indices, dtype=dtype, chunks=stim_id_chunks, compression='lzf')

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

            # Record which entities populate axis-0 of the data slab. For source='ks'
            # also store the cluster ids, since (unlike channels) the unit count is
            # variable and the ids are not implied by position.
            f.attrs['source'] = source
            if source == 'ks':
                f.attrs['unit_ids'] = np.asarray(channels)

        # h5py handle is now closed; append the pandas dataframes (PyTables) to
        # the same file, one open at a time.

        # Write full dataframe of trial parameters. format='fixed' reads back
        # dramatically faster than format='table' for a full-table read (the
        # only way this table is ever read -- no partial/`where=` queries
        # exist anywhere on it) since it avoids format='table's indexed,
        # per-dtype-block on-disk layout; format='table' remains the default
        # here only for backward compatibility (opt in via
        # trial_params_format='fixed'):
        trial_params_df_out = trial_params_df.copy()
        trial_params_df_out = standardize_col_types(trial_params_df_out)
        trial_params_df_out.to_hdf(output_path, 'trial_params', 'a', format=trial_params_format)

        #"""
        # Write truncated dataframe of select trial parameters:
        short_cols = ['monkey', 'date', 'trial_num', 'rsvp_num', 'stim_id', 'stim_idx', 'scenefile', 'behav_file', 'reward_bool', 'stim_completed', 'frac_completed', 't_on']
        if 'img_full_path' in trial_params_df.columns:
            short_cols.append('img_full_path')
        trial_params_short = trial_params_df[short_cols]
        trial_params_short = trial_params_short.rename(columns={'stim_info_short' : 'stim_id'})
        trial_params_short.to_hdf(output_path, 'trial_params_short', 'a', format='fixed')
        #"""

        # Write channel coordinates:
        zero_coords.to_hdf(output_path, key='zero_coordinates', mode='a', format='fixed')
        imro_tbl.to_hdf(output_path, key='imro_table', mode='a', format='fixed')

        # For single units, write the per-unit metrics tables (each row-aligned
        # to axis-0 of `data`): unit_quality for good-unit filtering downstream,
        # unit_spatial for probe location / amplitude.
        if source == 'ks':
            if unit_quality_df is not None:
                standardize_col_types(unit_quality_df.copy()).to_hdf(output_path, 'unit_quality', 'a', format='table')
            if unit_spatial_df is not None:
                standardize_col_types(unit_spatial_df.copy()).to_hdf(output_path, 'unit_spatial', 'a', format='table')

        if 'analysis_metadata' in sys.modules:
            M = Metadata()
            for i in input_files:
                M.add_input(i)
            M.add_output(output_path)
            M.add_param('chunk_size', chunk_size)
            M.add_param('dtype', str(dtype))
            M.date = datetime.now().strftime('%Y-%m-%d')
            M.time = datetime.now().strftime('%H:%M:%S')
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

    """

    # `h5` may already be an open h5py.File (existing behavior) or a path.
    # If it's a path, open it with a chunk cache sized for the new chunking
    # scheme instead of the 1MB h5py default:
    if isinstance(h5, (str, Path)):
        h5 = h5py.File(h5, 'r', rdcc_nbytes=_DEFAULT_RDCC_NBYTES, rdcc_nslots=_DEFAULT_RDCC_NSLOTS)

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
        time_window = [0, n_bins]

    # Define requested trial range:
    min_trial = min(trials[:,0])
    max_trial = max(trials[:,0])

    # h5py's fancy indexing requires strictly increasing, unique indices along
    # any axis, whereas `channels` (like plain NumPy indexing) may be given in
    # any order and may contain repeats. Read the unique, sorted channels the
    # underlying dataset actually supports, then reconstruct the originally
    # requested channel order (with any repeats) afterward via `inverse`:
    unique_channels, inverse = np.unique(channels, return_inverse=True)

    # Pre-fetch data from requested trial range, restricted to the requested
    # channels and time window at the HDF5 read itself, rather than reading
    # the full channel/time extent and discarding the unneeded parts in NumPy
    # afterward:
    print('Pre-fetching PSTHs from HDF5...')
    start_load = time.time()
    data = h5[dset_name][unique_channels, time_window[0]:time_window[1], min_trial:max_trial+1, :]
    stop_load = time.time()
    print('... done.')
    print('Duration={} minutes'.format((stop_load-start_load)/60))

    # Restore the originally requested channel order/repeats:
    data = data[inverse, :, :, :]

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
    stop_slice= time.time()
    print('... done.')
    print('Duration={} minutes'.format((stop_slice-start_slice)/60))

    # Hack; input HDF5s are saved as int32 to reduce space, I/O time, but this
    # has effect of turning nan into -2*10^9; convert back to nan here:
    slices = slices.astype(np.float32)
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
    with h5py.File(h5_path, 'r', rdcc_nbytes=_DEFAULT_RDCC_NBYTES, rdcc_nslots=_DEFAULT_RDCC_NSLOTS) as h5:
        
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
        
        # Get types in current column:
        curr_types = np.unique([str(type(x)) for x in df[col]])
    
        # Define specific fixes for different combinations of types; this part a bit hack-y:
        if "<class 'float'>" in curr_types and "<class 'str'>" in curr_types:
            
            # Convert str 'true' to 1:

            df.loc[df[col]=='true', col] = 1

            # If all floats are NaN, make everything string:
            floats = np.where([type(x)==float for x in df[col]])[0]
            nans = np.where(df[col].isna())[0]
            if len(floats)==len(nans) and np.all(floats==nans):
                df[col]= df[col].astype(str)
    
            # Otherwise, convert everything to float:
            else: 
                df.loc[df[col]=='', col] = np.nan # Convert any empty strings to nan
                df[col] = df[col].astype(float)
        
        # If all non-NaNs are arrays:
        if np.all(df[~df[col].isna()][col].apply(lambda x : type(x)==np.ndarray)):

            # If all arrays are singleton:
            if np.all(df[~df[col].isna()][col].apply(lambda x : len(x)==1)):
                df[col] = df.apply(lambda x : x[col][0] if type(x[col])==np.ndarray else x[col], axis=1)

    # Second pass: catch object columns PyTables can't serialize with
    # format='table' even when they are NOT multi-type. A column that is uniformly
    # dict / list / nested (e.g. a behavior field like SampleStartTime that comes
    # back as a dict in some files) has typenum==1 and so is missed above, but
    # to_hdf(format='table') still raises "Cannot serialize the column". Coerce any
    # object column holding non-scalar values: to numeric if possible (malformed
    # entries -> NaN), otherwise to string.
    def _is_nonscalar(x):
        return isinstance(x, (dict, list, tuple, set, np.ndarray))

    for col in df.columns:
        if df[col].dtype != object:
            continue
        if df[col].apply(_is_nonscalar).any():
            coerced = pd.to_numeric(df[col], errors='coerce')
            # Use the numeric coercion only if it preserved the real (scalar) values;
            # if everything became NaN the column was non-numeric, so stringify instead.
            scalar_mask = ~df[col].apply(_is_nonscalar) & df[col].notna()
            if scalar_mask.any() and coerced[scalar_mask].notna().all():
                df[col] = coerced
            else:
                df[col] = df[col].apply(lambda x : '' if _is_nonscalar(x) else x).astype(str)

    return df



def df_2_img_full_paths(df, base_data_directory=os.path.join('/', 'mnt', 'smb', 'locker', 'issa-locker', 'Data')):

    # Find unique scenefiles:
    sfiles_df = df[['monkey', 'scenefile']].drop_duplicates()
    sfiles_df['scenefile_short'] = sfiles_df.apply(lambda x: x.scenefile.split('/')[-1][:-5], axis=1) # Extract core scenefile name

    # Find saved image directory for each scenefile:
    saved_imgs_directories = sfiles_df.apply(lambda x : sfile_2_sv_img_dir(x.scenefile_short, monkey=x.monkey, base_data_directory=base_data_directory), axis=1).values

    # Find all images in each saved image directory:
    im_path_df = pd.DataFrame()
    for s, sdir in enumerate(saved_imgs_directories):

        curr_im_path_df = sv_img_dir_2_im_paths(sdir)
        curr_im_path_df['monkey'] = sfiles_df.iloc[s].monkey
        curr_im_path_df['scenefile'] = sfiles_df.iloc[s].scenefile
        im_path_df = pd.concat([im_path_df, curr_im_path_df], axis=0)

    # Go back and grab scenefiles, image indices for any scene where images were not found:
    files_not_found_sfiles = sfiles_df[np.array([x is None for x in saved_imgs_directories])]
    files_not_found_imgs = pd.merge(df[['monkey', 'date', 'scenefile', 'scenefile_img_idx']].drop_duplicates(), files_not_found_sfiles, on=['monkey', 'scenefile'])
    files_not_found_imgs['img_full_path'] = None
    im_path_df = pd.concat([im_path_df, files_not_found_imgs], axis=0)

    # Reorder columns, rows:
    im_path_df = im_path_df[['monkey', 'scenefile', 'scenefile_img_idx', 'img_full_path']]
    im_path_df = im_path_df.sort_values(by=['monkey', 'scenefile', 'scenefile_img_idx'])
    im_path_df.index = np.arange(im_path_df.shape[0])
    
    return im_path_df



def sfile_2_sv_img_dir(sfile_name, base_data_directory=os.path.join('/', 'mnt', 'smb', 'locker', 'issa-locker', 'Data'), monkey=None):

    # Optionally restrict search to monkey-specific saved images directory; speeds things up ~20X
    if monkey is not None:
        search_root = os.path.join(base_data_directory, monkey, 'Saved_Images') 
    else:
        search_root = base_data_directory

    # Define search string:
    search_str = os.path.join(search_root, '**', sfile_name+'.json')
    matches = glob.glob(search_str, recursive=True)

    # If exactly one scenefile matches search term, return it:
    if len(matches) == 1:
        match = matches[0]
        scenefile_directory = os.path.split(match)[0]

    # If no scenefiles match search term:
    elif len(matches) == 0:
        warnings.warn('No saved images folder for requested scenefile {} discovered.'.format(sfile_name))
        scenefile_directory = None

    # If more than one scenefile matches search term: 
    elif len(matches) > 1:

        # HACK: Exclude any path with 'other' in title:
        matches = [m for m in matches if 'other' not in m]
        
        # HACK: Exclude any path with 'scenefiles_update' in title:
        matches = [m for m in matches if 'scenefile_update' not in m]

        # If there are stil multiple saved image directories matching query, just arbitrarily choose first one and raise warning 
        # ASSUMES ALL SCENEFILES WITH SAME NAME INCLUDE SAME IMAGES!!!
        # TODO: Do a better job of resolving ambiguities!!
        warn_str = '\n'.join(['More than one saved images folder discovered for scenefile {}:'.format(sfile_name, matches),
            '\n'.join(matches),
            'Selecting path to first saved images directory {}.'.format(matches[0]),
        ]) + '\n'
        warnings.warn(warn_str)
            
        #warnings.warn('More than one saved images folder discovered for scenefile {}:'.format(sfile_name, matches))  
        #warnings.warn('\n'.join(matches))
        match = matches[0]
        scenefile_directory = os.path.split(match)[0]
    
    return scenefile_directory



def sv_img_dir_2_im_paths(sv_img_dir):

    # Select image files:
    imgs = [x for x in os.listdir(sv_img_dir) if re.search(r'_index\d+.png', x) is not None]

    # Extract image indices:
    img_indices = [int(re.search(r'_index\d+.png', img).group()[6:][:-4]) for img in imgs] 
    
    # Create dataframe:
    im_paths_df = pd.DataFrame()
    im_paths_df['scenefile_img_idx'] = img_indices
    im_paths_df['img_full_path'] = [os.path.join(sv_img_dir, img) for img in imgs]    
    
    return im_paths_df



def add_im_full_paths(trial_params_df, local_data_path=None):
    
    # If input dataframe already has img_full_path columns, delete it; will replace
    if 'img_full_path' in trial_params_df.columns:
        trial_params_df = trial_params_df.drop(columns=['img_full_path'])
    
    # Iterate over monkeys, dates:
    unique_images_df = pd.DataFrame()
    sessions = trial_params_df[['monkey', 'date']].drop_duplicates()
    for r, row in sessions.iterrows():
    
        monkey = row['monkey']
        date = row['date']
        curr_date_trial_params = trial_params_df[trial_params_df.date==date]
        
        # Try to find saved image directories for all scenefiles:    
        sfiles = np.unique(curr_date_trial_params.scenefile)
        sfile_basenames = [x.split('/')[-1][:-5] for x in sfiles] 
        sfile_saved_img_dirs = [scenefile_2_img_dir(x, monkey, local_data_path) for x in sfiles]
        
        # HACK: if sfiles includes ABC scenefiles, change the saved image directories to 
        # those inside experiment directory for UVW, XYZ:
        if np.any(['ABC' in x for x in sfiles]):
            
            # Find experiment directories for novel scene
            novel_img_dirs = [x for x in sfile_saved_img_dirs if x is not None and ('UVW' in x or 'XYZ' in x)]
            novel_exp_dirs = []
            for n in novel_img_dirs:
                novel_exp_dirs.append(n.split(os.path.sep)[-2])
            
            # If all novel scenefiles are from the same experiment:
            if len(np.unique(novel_exp_dirs)) == 1:
                novel_exp_dir = novel_exp_dirs[0]
                
                for i, s in enumerate(sfile_saved_img_dirs):
                    if s is not None and 'ABC' in s:
                        sfile_parts = s.split(os.path.sep)
                        sfile_parts[-2] = novel_exp_dir 
                        new_sfile = os.path.sep.join(sfile_parts)
                        sfile_saved_img_dirs[i] = new_sfile
        
        # HACK: If any scenefiles are missing an experiment directory, and if all 
        # other scenefiles are in the same known experiment directory, then just 
        # assume all scenefiles are in that same known experiment directory; in a
        # worst case scenario, the search will fail at the call to stim_idx_2_img_path
        # below and return None.
        expt_dirs = [x.split(os.path.sep)[-2] for x in sfile_saved_img_dirs if x is not None]
        if len(np.unique(expt_dirs)) == 1:
            base = [x for x in sfile_saved_img_dirs if x is not None][0].split(os.path.sep)[:-2]
            base = os.path.sep.join(base)
            expt_dir = expt_dirs[0]
            sfile_saved_img_dirs = [os.path.join(base, expt_dir, x) for x in sfile_basenames]
         
        # HACK: data_dicts appear to include a mistake where stim indices are 
        # off by some offset; correct here:
        for s in sfiles:
            curr_rows = curr_date_trial_params.scenefile == s
            curr_date_trial_params.loc[curr_rows, 'stim_idx'] = curr_date_trial_params.loc[curr_rows, 'stim_idx'] - min(curr_date_trial_params.loc[curr_rows, 'stim_idx'])
        
        # Create dataframe of unique images, add image directories:
        curr_unique_images_df = curr_date_trial_params[['scenefile', 'stim_idx']].drop_duplicates()
        tmp = pd.DataFrame(columns=['scenefile', 'sfile_imdir'])
        tmp['scenefile'] = sfiles
        tmp['sfile_imdir'] = sfile_saved_img_dirs
        curr_unique_images_df = pd.merge(curr_unique_images_df, tmp, on=['scenefile'], how='outer')
        #print('sfile_saved_img_dirs={}'.format(np.array(sfile_saved_img_dirs)))
        
        # Get full paths to saved images, add to dataframe:
        impaths = curr_unique_images_df.apply(lambda x : stim_idx_2_img_path(x.sfile_imdir, x.stim_idx), axis=1)
        curr_unique_images_df['img_full_path'] = impaths
        curr_unique_images_df['monkey'] = monkey
        curr_unique_images_df['date'] = date
        unique_images_df = pd.concat([unique_images_df, curr_unique_images_df], axis=0)
    
    # Merge unique images with full paths to trial_params_df:
    trial_params_df = pd.merge(trial_params_df, unique_images_df[['monkey', 'date', 'scenefile', 'stim_idx', 'img_full_path']].drop_duplicates(), 
           on=['monkey', 'date', 'scenefile', 'stim_idx'], how='left')    

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
    stim_set_regex = r'neural_stim_\d+_'
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
    exp_regex = r'[A-Z]{5,}_\d{2,2}.json'
    f = lambda x : re.search(exp_regex, x)
    exp_ids = [f(x).group()[-7:-5] for x in sfiles if f(x) is not None]
    
    # Verify that all exp_ids are the same; if not, raise warning and return None
    if len(np.unique(exp_ids)) == 1:
        exp_id = exp_ids[0]
    elif len(np.unique(exp_ids)) < 1: 
        warnings.warn('No experiment ID discovered among scenefile paths.')
        return None
    elif len(np.unique(exp_ids)) > 1: 
        warnings.warn('More than one experiment ID discovered among scenefile paths.')
        return None
        
    saved_imgs_dirname = 'Saved_Images_{}_neural_stim_{}_{}'.format(monkey, stim_set, exp_id)
    saved_imgs_base_dir = os.path.join(base_dir, monkey, 'Saved_Images', saved_imgs_dirname)     
    
    return saved_imgs_base_dir



def scenefile_2_img_dir(scenefile_name, monkey=None, local_base=None):
    """
    Find saved image directory for input scenefile.

    Parameters
    ----------
    scenefile_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if local_base is None:
        base = os.path.join('mnt', 'smb', 'locker', 'issa-locker', 'Data')
    else:
        base = local_base
    
    # Get monkey name:
    sfile_parts = scenefile_name.split('/')
    if monkey is None:
        monkey = sfile_parts[3]
    monkey_dir = os.path.join(base, monkey, 'Saved_Images')
    monkey_dir_contents = os.listdir(monkey_dir)
    
    # Get scenefile basename:
    sfile_basename = scenefile_name.split('/')[-1][:-5]
    
    ####

    scene_regex = r'neural_stim_\d+'
    if monkey == 'West':
        
        is_scene = re.search(scene_regex, scenefile_name) is not None
        is_natural_images = 'Rust' in scenefile_name and 'NaturalImages' in scenefile_name
        is_faces = 'elias' in scenefile_name or 'neptune' in scenefile_name
        is_hvm = re.search(r'hvm\d{2}_\w+_\d{2}_\d{8}', sfile_basename) is not None
        
        if is_scene or is_natural_images or is_faces:
        
            # If dealing with scene stimuli:        
            if is_scene:
                
                # Get stim set number:
                stim_set_str = re.search(scene_regex, scenefile_name).group()
                stim_set = int(stim_set_str[12:])
                                    
                # If stim set is less than 5
                if stim_set < 5:
        
                    expt_dirname = 'Saved_Images_{}_{}'.format(monkey, stim_set_str)
                    # POSSIBLY IMPORTANT? For stim_set = 4, this will automatically 
                    # default to Saved_Images_West_neural_stim_4 rather than 
                    # Saved_Images_West_neural_stim_4_1ABC2DEF_RSVP44; don't know how
                    # differentiate between which of these is appropriate based just on
                    # scenefile name; does it matter?
                    
                # If stim set is greater than or equal to 5, try to additionally get experiment ID:
                elif stim_set >= 5:
                    
                    expt_regex = r'_\d+[A-Z]{3,}\d*_\w{2,2}'
                    expt_search = re.search(expt_regex, scenefile_name)
                    if expt_search is not None:
                        expt_str = expt_search.group()[-2:]
                    else:
                        warnings.warn('No experiment ID discovered in scenefile name {}.'.format(scenefile_name))
                        return None
                    
                    # Define experiment directory:
                    expt_dirname = 'Saved_Images_{}_{}_{}'.format(monkey, stim_set_str, expt_str)
    
                expt_directory = os.path.join(monkey_dir, expt_dirname)
    
            ####
            # Else if dealing with natural image stimuli:
            elif is_natural_images:
                
                # Look for experiment directories containing 'Rust' and 'NaturalImages'
                matches = [x for x in monkey_dir_contents if 'Rust' in x and 'NaturalImages' in x]
                if len(matches) == 1:
                    expt_dirname = matches[0]
                    expt_directory = os.path.join(monkey_dir, expt_dirname)
                elif len(matches) < 1:
                    warnings.warn('No directories matching requested scenefile discovered in {}'.format(monkey_dir))
                    return None
                elif len(matches) > 1:
                    warnings.warn('More than one directory matching requested scenefile discovered in {}'.format(monkey_dir))
                    return None
            
                # Random exception handling:
                if monkey == 'West':
                    expt_directory = os.path.join(expt_directory, 'Save_Images_West_RustDiCarlo')
            
                
            ####
            # Else if dealing with face stimuli:
            elif is_faces:
                
                face_expt_dirs = [x for x in monkey_dir_contents if 'Elias' in x and 'Neptune' in x]
                if len(face_expt_dirs) == 1:
                    expt_dirname = face_expt_dirs[0]
                elif len(face_expt_dirs) < 1:
                    warnings.warn('No face experiment directory discovered in {}.'.format(monkey_dir))
                    return None
                elif len(face_expt_dirs) > 1:
                    warnings.warn('More than one face experiment directory discovered in {}.'.format(monkey_dir))    
                    return None
            
                expt_directory = os.path.join(monkey_dir, expt_dirname)    
                
                # Random exception handling:
                if monkey == 'West':
                    expt_directory = os.path.join(expt_directory, 'Save_Images_West_EliasNeptune')        

            # Find directory in expt_directory with same name as scenefile basename:
            if not os.path.exists(expt_directory):
                warnings.warn('Experiment directory for scenefile {} not found in saved image folder {}; returning None.'.format(scenefile_name, expt_directory))
                return None
            expt_dir_contents = os.listdir(expt_directory)
            if sfile_basename in expt_dir_contents:
                img_dir = os.path.join(expt_directory, sfile_basename)
            else:
                
                #HACK: If scenefile folder not found in experiment directory, check E6 folder instead:
                e6_dir = os.path.join(monkey_dir, 'Saved_Images_{}_neural_stim_{}_E6'.format(monkey, stim_set))
                
                if os.path.exists(e6_dir):
                    e6_contents = os.listdir(e6_dir)
                    if sfile_basename in e6_contents:
                        img_dir = os.path.join(e6_dir, sfile_basename)
                    else:
                        warnings.warn('Scenefile directory for {} not found in {}; returning None.'.format(sfile_basename, expt_directory))
                        return None
                else:
                    warnings.warn('Scenefile directory for {} not found; returning None.'.format(sfile_basename))
                    img_dir = None

        ####
        # Else if dealing with HvM stimuli:
        elif is_hvm:            
            img_dir = os.path.join(monkey_dir, 'hvm10', sfile_basename)
        
        ###
        # Otherwise, raise warning and return None
        else:
            warnings.warn('Input scenefile {} does not match any specified pattern.'.format(scenefile_name))
            return None
        
        
    elif monkey == 'Bourgeois':
            
        all_saved_img_dirs = [x[0] for x in os.walk(monkey_dir) if os.path.isdir(x[0])]
        
        # If dealing with scene stimuli:        
        if re.search(scene_regex, scenefile_name) is not None:
            matches_sfile_basename = [x for x in all_saved_img_dirs if re.search(sfile_basename+r'$', x) is not None]
            if len(matches_sfile_basename) == 1:
                img_dir = matches_sfile_basename[0]
            elif len(matches_sfile_basename) == 0:
                warnings.warn('No directory matching pattern {} discovered in {}; setting saved image directory to None.'.format(sfile_basename, monkey_dir))
                img_dir = None
            elif len(matches_sfile_basename) > 1: 
                warnings.warn('More than one directory matching pattern {} discovered in {}; setting saved image directory to None.'.format(sfile_basename, monkey_dir))
                img_dir = None
        
        # Otherwise, raise warning and return None
        else:
            warnings.warn('Input scenefile {} does not match any specified pattern.'.format(scenefile_name))
            img_dir = None
    
    return img_dir



def stim_idx_2_img_path(sfile_img_dir, stim_idx):
    
    if sfile_img_dir is None or stim_idx is None or not os.path.exists(sfile_img_dir):
        return None
    
    stim_idx = int(stim_idx)
    matching_imgs = [x for x in os.listdir(sfile_img_dir) if '_index{}.png'.format(stim_idx) in x] # Get all PNGs in base_imdir:       
    
    # If one or more images with matching index discovered:
    if len(matching_imgs) >= 1:
        
        # Raise warning if more than one match:
        if len(matching_imgs) > 1:
            warnings.warn('More than one image with index {} discovered in {}; returning path to first image.'.format(stim_idx, sfile_img_dir))
        
        imname = matching_imgs[0] # Just selecting first image if there are duplicate images with same index; assuming that's fine for now
        impath = os.path.join(sfile_img_dir, imname)
    
    # If no images with matching indices discovered:
    else:
        warnings.warn('No images with index {} discovered in {}; returning None.'.format(stim_idx, sfile_img_dir))
        impath =None
    
    return impath



def coerce_trial_timing(value: dict | list | np.ndarray, n_trials: int) -> np.ndarray:
    """Normalize a per-trial timing field to a flat float array of length n_trials.

    Behavior files occasionally store a timing field (e.g. SampleStartTime) as a
    dict or a short/ragged list rather than one value per trial. np.array() then
    yields a 0-d or wrong-length array that breaks downstream arithmetic. This
    coerces `value` to a length-n_trials float array: parses numerics
    (errors -> NaN), and pads/truncates to n_trials, NaN-filling anything missing.

    Parameters
    ----------
    value : dict | list | array-like
        Raw TRIALEVENTS timing field for one behavior file.
    n_trials : int
        Number of trials (use a reliably per-trial field such as NReward).

    Returns
    -------
    numpy.ndarray
        Float array of shape (n_trials,).
    """
    if isinstance(value, dict):
        vals = np.empty(0, dtype=float)
    else:
        arr = pd.to_numeric(pd.Series(value).squeeze(), errors='coerce')
        vals = np.asarray(arr, dtype=float).ravel()

    if vals.shape[0] == n_trials:
        return vals

    out = np.full(n_trials, np.nan)
    out[:min(len(vals), n_trials)] = vals[:n_trials]  # keep what aligns, NaN the rest
    return out



def find_complete_rsvp_slots(bfile):

    # Get some scene metadata:
    sample_scenes = bfile['SCENES']['SampleScenes'] 
    durations = [s['durationMS'][0] if (type(s['durationMS'])==list and len(s['durationMS'])==1) else s['durationMS'] for s in sample_scenes]
    nstims = [s['nimages'] for s in sample_scenes]
    try:
        feedback_pre = bfile['TASK']['FeedbackPRE']
    except:
        feedback_pre = 0
    scene_df = pd.DataFrame(np.array([nstims, durations]).T, columns=['nstim', 'stim_duration'])
    scene_df['scenefile_idx'] = np.arange(scene_df.shape[0])

    # Get single-trial data:
    trial_df = pd.DataFrame()

    # Get trial timing data. Some behavior files have a malformed timing field
    # (e.g. SampleStartTime stored as a dict instead of a per-trial list), which
    # np.array() turns into a 0-d / object array of the wrong length and breaks the
    # arithmetic below. coerce_trial_timing() normalizes each field to a flat float
    # array of length n_trials (reward/NReward is reliably per-trial), NaN-filling
    # where it can't be parsed.
    reward = np.array(bfile['TRIALEVENTS']['NReward'])
    n_trials = len(reward)

    start_time = coerce_trial_timing(bfile['TRIALEVENTS']['StartTime'], n_trials)
    sample_start_time = coerce_trial_timing(bfile['TRIALEVENTS']['SampleStartTime'], n_trials)
    reinforcement_time = coerce_trial_timing(bfile['TRIALEVENTS']['ReinforcementTime'], n_trials)
    end_time = coerce_trial_timing(bfile['TRIALEVENTS']['EndTime'], n_trials)
    reward = coerce_trial_timing(bfile['TRIALEVENTS']['NReward'], n_trials)
    trial_df['start_time'] = start_time
    trial_df['sample_start_time'] = sample_start_time
    trial_df['reinforcement_time'] = reinforcement_time
    trial_df['end_time'] = end_time
    trial_df['sample_duration'] = trial_df['reinforcement_time'] - trial_df['sample_start_time'] - feedback_pre + 16 # HACK!!! Hard-coding assumed 16-ms (1-frame) quantization effect; West rewarded RSVP trials have nominal duration of ~884 ms instead of expected 900; single frame drop?
    trial_df['trial_rewarded'] = reward.astype(bool)
    trial_df['trial_num'] = np.arange(trial_df.shape[0])

    # Get individual "absolute" stim indices (i.e., index into images pooled across scenefiles):
    sample_idx_abs = bfile['TRIALEVENTS']['Sample']
    sample_idx_ar = np.array(list(sample_idx_abs.values()))
    n_slots = sample_idx_ar.shape[0]
    stim_cols = ['stim'+str(r) for r in np.arange(n_slots)]
    for c, col in enumerate(stim_cols):
        trial_df[col] = sample_idx_ar[c]


    # Get scenefile and stim duration of each RSVP slot. Unlike the original
    # version, slots within a trial may come from *different* scenefiles (some
    # experiments draw one image per slot from a different scenefile), so the
    # scenefile and duration are tracked per slot rather than asserted equal.
    dur_cols = []
    offsets = np.cumsum(scene_df.nstim) - 1
    get_sfile_index = lambda x : min(np.where(offsets.values >= x)[0])
    for c, col in enumerate(stim_cols):

        curr_sfile_colname = 'sfile'+str(c)
        curr_dur_colname = 'stim{}_duration'.format(c)
        dur_cols.append(curr_dur_colname)
        trial_df[curr_sfile_colname] = list(map(get_sfile_index, trial_df[col].values))

        # Merge stim duration for each slot:
        trial_df = pd.merge(trial_df,
                            scene_df[['scenefile_idx', 'stim_duration']].rename(columns={'scenefile_idx':curr_sfile_colname}),
                            on=curr_sfile_colname)\
                        .rename(columns={'stim_duration':curr_dur_colname})

    # Aggregate stimulus durations into single array (leading 0 so cumsum gives slot boundaries):
    trial_df['stim_durs'] = trial_df.apply(lambda x : np.array([0] + [x[col] for col in dur_cols]), axis=1)

    # Compute number of completed stim from the per-slot duration cumsum:
    trial_df['n_stim_complete'] = trial_df.apply(lambda x :
        max(np.where(x.sample_duration > np.cumsum([x[c] for c in dur_cols]))[0])+1
        if len(np.where(x.sample_duration > np.cumsum([x[c] for c in dur_cols]))[0]) > 0
        else 0, axis=1)

    # Expand trials to individual slots, assigning each slot its own scenefile_idx:
    T = []
    for t in np.arange(n_slots):
        base_cols = [x for x in trial_df.columns if 'stim' not in x and 'sfile' not in x] + ['stim_durs', 'n_stim_complete']
        curr_df = trial_df.copy()[base_cols + ['stim'+str(t), 'sfile'+str(t)]].rename(columns={'stim'+str(t):'stim_idx', 'sfile'+str(t):'scenefile_idx'})
        curr_df.insert(curr_df.shape[1], 'rsvp_num', t)
        curr_df['stim_idx'] = trial_df['stim'+str(t)]
        T.append(curr_df)
    rsvp_df = pd.concat(T)
    rsvp_df = rsvp_df.sort_values(by=['trial_num', 'rsvp_num'])

    # Convert "absolute" stim indices to within-scenefile stim index:
    offsets_hat = np.array([0] + list(offsets.values+1))
    rsvp_df['stim_idx'] = rsvp_df.apply(lambda x : x.stim_idx - offsets_hat[x.scenefile_idx], axis=1)

    # Determine whether each stim. presentation was successfully fixated through:
    fixation_broken = rsvp_df.apply(lambda x : ~x.trial_rewarded and x.rsvp_num > x.n_stim_complete-1, axis=1)
    rsvp_df['stim_completed'] = ~fixation_broken.values.astype(bool)
    rsvp_df['frac_completed'] = rsvp_df.apply(lambda x : (x.sample_duration - np.cumsum(x.stim_durs[0:x.n_stim_complete+1])[-1])/x.stim_durs[x.rsvp_num+1] if not x.stim_completed else 1.0, axis=1).values

    # Drop unneeded columns:
    rsvp_df = rsvp_df[['trial_num', 'rsvp_num', 'scenefile_idx', 'stim_idx', 'stim_completed', 'frac_completed', 'trial_rewarded', 'start_time', 'sample_start_time', 'reinforcement_time', 'end_time']]

    return rsvp_df
            

# ---------------------------------------------------------------------------
# The functions below (find_h5_path through sessions2trials_cached) were
# moved here from mkutils_ddk/IO.py (a separate git repository), where they
# originally lived. Full commit-level history for this code is not directly
# traceable via `git log`/`git blame` from here, since git's history
# tracking doesn't cross repository boundaries -- see these commits in
# mkutils_ddk for the original authorship/history:
#   c8284da  Add unit_type param to sessions2trials(), pass to find_h5_path()
#   439d504  Add support for finding kilosorted HDF5 in find_h5_path()
#   9d12b79  Add per-session cached loader for spike data
#   6716382  Key the PSTH cache on concrete trial tuples, not filter source text
#   14953b4  Fetch sessions concurrently in the trial-params and PSTH caches
# ---------------------------------------------------------------------------



def find_h5_path(monkey, date, unit_type='mua'):
    """
    Find path to HDF5 of preprocessed PSTHs for given monkey and session.

    Parameters
    ----------
    monkey : str
        Monkey name.

    date : str
        Date, formatted <yyyymmdd>.

    Returns
    -------
    h5_path : str
        Path to HDF5 of preprocessed PSTHs for requested monkey, session.

    """

    if unit_type == 'ks':
        # Delegate to resolve_ks_h5_path, which matches the path spike_sorting.process_ks_data()
        # actually writes to (derived from the save-out recording dir, not the raw data one --
        # the two can diverge, unlike the mua case handled below).
        try:
            return str(resolve_ks_h5_path(monkey, date))
        except Exception:
            print('H5 file not found for {}, {}'.format(monkey, date))
            return None

    #engram_drive = get_engram_drive()

    hostname = socket.gethostname()
    try:
        if 'rc.zi.columbia.edu' in hostname:
            engram_drive = get_engram_drive()
            base_data_path = os.path.join(engram_drive, 'Data')
            folder_level_offset = 4
            recording_path = get_recording_path(Path(base_data_path), Path(monkey), date, depth=4)[0]
            dirname = recording_path.split(os.path.sep)[folder_level_offset+3]
            preprocessed_data_dir = os.path.join(engram_drive, 'processed_h5', monkey, dirname, 'mua')
        else:
            if hostname == 'DESKTOP-1PVCRAF':
                local_preprocessed_dir = 'F:\\'
            elif hostname == 'DESKTOP-PJOJ7HT':
                local_preprocessed_dir = os.path.join('C:\\', 'Users', 'danie', 'Documents')
            h5dir = os.path.join(local_preprocessed_dir, 'h5s_test')
            preprocessed_data_dir = os.path.join(h5dir, monkey)

        h5_path = os.path.join(preprocessed_data_dir, '{}.h5'.format(date))
    except:
        print('H5 file not found for {}, {}'.format(monkey, date))
        h5_path = None

    return h5_path



_CACHE_SCHEMA_VERSION = 2  # bumped: psth cache key is now value-based (trial/rsvp tuples), not filter-text-based



def _build_fetch_flt(misc_flt, group_defs):
    """
    Build the row-predicate callable used to filter trials, from misc_flt +
    group_defs only. A trial passes if it satisfies misc_flt AND matches at
    least one group_def's 'definition'. Used by `filter_trial_params`.
    """
    group_predicates = [gd['definition'] for gd in group_defs] if group_defs else []

    def flt(row):
        if misc_flt is not None and not misc_flt(row):
            return False
        if len(group_predicates) == 0:
            return True
        return matches_any_predicate(row, group_predicates)

    return flt



def filter_trial_params(trial_params_df, misc_flt=None, group_defs=None):
    """
    Apply misc_flt + group_defs (conditions) to a trial_params dataframe --
    e.g. the output of `sessions2trial_params_cached` -- returning only the
    rows that pass. Deliberately does NOT take class_defs/dichotomies; those
    are applied downstream on already-loaded PSTH data.

    The point of doing this as an explicit, separate step (rather than
    passing misc_flt/group_defs into a PSTH-fetching function directly) is
    that its *output* -- concrete (monkey, date, trial_num, rsvp_num) rows --
    is what should determine the PSTH cache key, not the filter functions
    themselves. Two notebooks with differently-worded but equivalent
    misc_flt/group_defs will produce the same filtered rows and therefore
    share the same `sessions2trials_cached` cache entries; two notebooks
    that only shared a cache key by filter-function source text would not.

    Returns
    -------
    pandas.DataFrame
        Subset of `trial_params_df` whose rows satisfy the filter, columns
        unchanged.
    """
    flt = _build_fetch_flt(misc_flt, group_defs)
    return trial_params_df[trial_params_df.apply(flt, axis=1)].reset_index(drop=True)



def _h5_fingerprint(h5_path):
    if h5_path is not None and os.path.exists(h5_path):
        stat = os.stat(h5_path)
        return [stat.st_mtime, stat.st_size]
    return None



def _trial_params_cache_key(monkey, date, unit_type, h5_path):
    """
    Cache key for the *unfiltered* per-session trial_params cache -- depends
    only on which session/file is being read, never on any filter.
    """
    key_parts = {
        'schema_version': _CACHE_SCHEMA_VERSION,
        'monkey': monkey,
        'date': date,
        'unit_type': unit_type,
        'h5_path': h5_path,
        'h5_fingerprint': _h5_fingerprint(h5_path),
    }
    key_str = json.dumps(key_parts, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
    return key_hash, key_parts



def _fetch_one_trial_params(monkey, date, unit_type, cache_root, force_refresh):
    """
    Single-session worker for `sessions2trial_params_cached`, factored out
    so it can be dispatched to a thread pool. Returns None (with a warning)
    if the session's H5 can't be found, so callers can filter Nones out of
    a parallel-map result list.
    """
    h5_path = find_h5_path(monkey, date, unit_type=unit_type)
    if h5_path is None or not os.path.exists(h5_path):
        warnings.warn('H5 file for {}, {} not found.'.format(monkey, date))
        return None

    key_hash, key_parts = _trial_params_cache_key(monkey, date, unit_type, h5_path)
    cache_dir = os.path.join(cache_root, 'trial_params', unit_type, '{}_{}'.format(monkey, date), key_hash)
    trial_params_path = os.path.join(cache_dir, 'trial_params.h5')
    meta_path = os.path.join(cache_dir, 'meta.json')

    if not force_refresh and os.path.exists(trial_params_path):
        print('{}, {}... (trial params cache hit)'.format(monkey, date))
        tdf = pd.read_hdf(trial_params_path, 'trial_params')
    else:
        print('{}, {}... (fetching trial params from HDF5)'.format(monkey, date))
        tdf = h5_2_trial_df(h5_path, 'all')
        os.makedirs(cache_dir, exist_ok=True)
        tdf.to_hdf(trial_params_path, 'trial_params', mode='w', format='fixed')
        with open(meta_path, 'w') as f:
            json.dump(key_parts, f, indent=1, default=str)

    return tdf



def sessions2trial_params_cached(sessions_df, unit_type='mua', cache_root=None, force_refresh=False, max_workers=8):
    """
    Load (and cache) the FULL, unfiltered trial_params table for each
    requested session -- step 1-2 of the recommended workflow: load once,
    cache per-session, then filter in-memory (see `filter_trial_params`) as
    many times as needed without re-touching HDF5.

    Cheap to call repeatedly with a growing `sessions_df`: only sessions not
    already cached (or whose source HDF5 has changed) trigger an actual
    HDF5 read via `h5_2_trial_df`; already-cached sessions are served
    straight from disk. Unlike `sessions2trials_cached`, there is no filter
    dependence at all in this cache's key, since it holds every trial in the
    session, unfiltered.

    Sessions are fetched concurrently (see `max_workers`) since each
    session's read is independent and I/O-bound (dominated by network
    filesystem latency, not CPU), rather than serially as in earlier
    versions of this function.

    Parameters
    ----------
    sessions_df : pandas.DataFrame
        Must define 'monkey', 'date' columns.

    unit_type : str
        'mua' | 'ks'.

    cache_root : str
        Root directory for the on-disk cache. Required -- e.g.
        `os.path.join(mnt, 'users', 'Dan', 'ephys', 'spike_cache')`. Uses a
        `trial_params/` subdirectory, so it can share a `cache_root` with
        `sessions2trials_cached` without colliding.

    force_refresh : bool
        If True, ignore any existing cache entry and re-fetch + overwrite it.

    max_workers : int | None
        Number of sessions to fetch concurrently, via a thread pool (threads,
        not processes -- the work here is I/O-bound network reads, not CPU,
        and threads keep everything in one process/address space so a cache
        hit's memmapped array stays a lazy view rather than being pickled
        across a process boundary). Set to 1 or None to fetch serially
        (e.g. for cleaner interleaved console output while debugging). The
        default of 8 is a reasonable starting point, not a tuned value --
        the network filesystem itself may cap useful concurrency lower (SMB2
        has its own flow-control credit limit observed elsewhere in this
        codebase's investigation), so reduce it if higher settings don't
        help or if fetches start erroring under contention.

    Returns
    -------
    pandas.DataFrame
        Concatenated, unfiltered trial_params for all requested sessions.
    """
    if cache_root is None:
        raise ValueError('cache_root must be specified (no hardcoded default -- '
                          'e.g. os.path.join(mnt, "users", "Dan", "ephys", "spike_cache")).')

    sessions_df = sessions_df[['monkey', 'date']].drop_duplicates()
    session_list = list(sessions_df.itertuples(index=False, name=None))

    if max_workers is not None and max_workers > 1 and len(session_list) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dfs = list(executor.map(
                lambda s: _fetch_one_trial_params(s[0], s[1], unit_type, cache_root, force_refresh),
                session_list))
    else:
        dfs = [_fetch_one_trial_params(monkey, date, unit_type, cache_root, force_refresh)
               for monkey, date in session_list]

    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, axis=0).reset_index(drop=True) if dfs else pd.DataFrame()



def _psth_cache_key(monkey, date, unit_type, h5_path, channels, time_window, pairs_sorted):
    """
    Cache key for the PSTH cache -- depends on the *exact set* of
    (trial_num, rsvp_num) pairs being requested for this session, not on any
    filter function's identity/source text. Two independently-constructed
    request sets that resolve to the same pairs produce the same key.
    """
    channels_key = 'all' if channels is None else sorted(np.asarray(channels).tolist())
    time_window_key = 'all' if time_window is None else list(time_window)
    key_parts = {
        'schema_version': _CACHE_SCHEMA_VERSION,
        'monkey': monkey,
        'date': date,
        'unit_type': unit_type,
        'h5_path': h5_path,
        'h5_fingerprint': _h5_fingerprint(h5_path),
        'channels': channels_key,
        'time_window': time_window_key,
        'trial_rsvp_pairs': [list(p) for p in pairs_sorted],
    }
    key_str = json.dumps(key_parts, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
    return key_hash, key_parts



def _find_superset_psth_cache_entry(monkey, date, unit_type, h5_path, channels, time_window, h5_fingerprint, pairs_sorted, cache_root):
    """
    Look for an existing PSTH cache entry for this session (e.g. from an
    earlier run with broader misc_flt/group_defs, or a wider time_window)
    whose cached trial/RSVP pairs are a superset of `pairs_sorted` and whose
    cached time window contains `time_window`, so it can be served instead
    of re-fetching from HDF5 -- slicing the time-bin axis down as well as
    the trial axis, if the cached window is wider than what's requested.
    `channels` must still match exactly (not treated as a containment
    relationship). A single existing entry must cover the full request --
    this does not union rows/bins across multiple partial entries.

    Returns (index_df, psth_arr) for the first matching entry found --
    `psth_arr` already sliced down to `time_window` if the match came from a
    wider cached window -- or None if no existing entry is a superset.
    """
    session_dir = os.path.join(cache_root, unit_type, '{}_{}'.format(monkey, date))
    if not os.path.isdir(session_dir):
        return None

    channels_key = 'all' if channels is None else sorted(np.asarray(channels).tolist())
    time_window_key = 'all' if time_window is None else list(time_window)
    requested_pairs = set(pairs_sorted)
    psth_bins = None  # lazily loaded from the H5 attrs, only if actually needed for bin-index math

    for entry_name in os.listdir(session_dir):
        meta_path = os.path.join(session_dir, entry_name, 'meta.json')
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if meta.get('schema_version') != _CACHE_SCHEMA_VERSION:
            continue
        if meta.get('channels') != channels_key:
            continue
        if meta.get('h5_fingerprint') != h5_fingerprint:
            continue

        cached_pairs = set(tuple(p) for p in meta.get('trial_rsvp_pairs', []))
        if not requested_pairs.issubset(cached_pairs):
            continue

        cache_dir = os.path.join(session_dir, entry_name)
        psth_path = os.path.join(cache_dir, 'psth.npy')
        index_path = os.path.join(cache_dir, 'trial_index.h5')
        if not (os.path.exists(psth_path) and os.path.exists(index_path)):
            continue

        index_df = pd.read_hdf(index_path, 'trial_index')
        psth_arr = np.load(psth_path, mmap_mode='r')
        if len(index_df) != psth_arr.shape[0]:
            continue

        cached_time_window = meta.get('time_window')
        if cached_time_window == time_window_key:
            # Exact time-window match -- no bin slicing needed:
            return index_df, psth_arr

        # A specific (non-'all') cached window can never contain an 'all'
        # (time_window is None) request:
        if time_window is None:
            continue

        # h5_2_dat_array_rsvp's bin_indices convention (verified against a
        # real fetch) is a standard half-open [start, stop) range -- the
        # cached array's bin axis already has exactly (stop - start) bins,
        # 0-indexed from `start`, matching the slicing below:
        if psth_bins is None:
            with h5py.File(h5_path, 'r') as h5:
                psth_bins = h5.attrs['psth_bins']

        req_start, req_stop = time_window2bin_indices(time_window, psth_bins)

        if cached_time_window == 'all':
            cached_start, cached_stop = 0, psth_arr.shape[-1]
        else:
            cached_start, cached_stop = time_window2bin_indices(cached_time_window, psth_bins)

        if not (cached_start <= req_start and req_stop <= cached_stop):
            continue

        return index_df, psth_arr[:, :, req_start - cached_start : req_stop - cached_start]

    return None



def _fetch_one_session_psth(monkey, date, group, unit_type, channels, time_window, cache_root, force_refresh):
    """
    Single-session worker for `sessions2trials_cached`, factored out so it
    can be dispatched to a thread pool. Returns None (with a warning) if the
    session's H5 can't be found; otherwise returns
    (spikes_df_for_this_session, (zero_coords_row, imro_rows) | None).
    Deliberately returns rather than mutates shared accumulator state, so
    results from concurrently-running workers can be combined afterward in
    the calling thread without any risk of a race.

    On a miss against this request's own exact-key cache entry, also checks
    for an existing entry whose cached pairs are a superset of what's needed
    (see `_find_superset_psth_cache_entry`) before falling back to HDF5.
    """
    h5_path = find_h5_path(monkey, date, unit_type=unit_type)
    if h5_path is None or not os.path.exists(h5_path):
        warnings.warn('H5 file for {}, {} not found.'.format(monkey, date))
        return None

    # Canonical, order-independent representation of exactly which
    # stimulus presentations are being requested for this session:
    pairs_sorted = sorted(set(zip(group['trial_num'].astype(int), group['rsvp_num'].astype(int))))

    key_hash, key_parts = _psth_cache_key(monkey, date, unit_type, h5_path, channels, time_window, pairs_sorted)
    cache_dir = os.path.join(cache_root, unit_type, '{}_{}'.format(monkey, date), key_hash)
    psth_path = os.path.join(cache_dir, 'psth.npy')
    index_path = os.path.join(cache_dir, 'trial_index.h5')
    meta_path = os.path.join(cache_dir, 'meta.json')

    cache_hit = not force_refresh and os.path.exists(psth_path) and os.path.exists(index_path)

    if cache_hit:
        index_df = pd.read_hdf(index_path, 'trial_index')
        psth_arr = np.load(psth_path, mmap_mode='r')
        if len(index_df) != psth_arr.shape[0]:
            warnings.warn(
                'Cache entry for {}, {} is inconsistent (trial_index/psth row '
                'count mismatch) -- re-fetching.'.format(monkey, date))
            cache_hit = False

    # No exact-key entry -- look for an existing entry from a broader past
    # request (e.g. different misc_flt/group_defs) whose cached trial/RSVP
    # pairs are a superset of what's needed here, and serve the subset from
    # it instead of re-fetching from HDF5. Doesn't write a new exact-key
    # entry for this request -- a repeat of the same request re-scans rather
    # than hitting an O(1) hash lookup, which is fine unless/until that scan
    # becomes a hot path.
    superset_hit = False
    if not cache_hit and not force_refresh:
        superset_entry = _find_superset_psth_cache_entry(
            monkey, date, unit_type, h5_path, channels, time_window, key_parts['h5_fingerprint'], pairs_sorted, cache_root)
        if superset_entry is not None:
            index_df, psth_arr = superset_entry
            superset_hit = True

    if cache_hit:
        print('{}, {}... (cache hit, {} trials)'.format(monkey, date, len(index_df)))
    elif superset_hit:
        print('{}, {}... (cache hit via superset entry, {} of {} trials)'.format(
            monkey, date, len(pairs_sorted), len(index_df)))
    else:
        print('{}, {}... (fetching PSTHs from HDF5)'.format(monkey, date))
        spike_inds = np.array(pairs_sorted)
        # h5_2_dat_array_rsvp's output order is always ascending
        # (trial_num, rsvp_num), independent of the input order of
        # `trials` -- verified directly -- so `pairs_sorted` (built with
        # the same ascending-tuple sort) already matches row-for-row.
        #
        # h5_2_dat_array_rsvp takes time_window as *bin indices*, not
        # seconds (unlike this function's own `time_window`, which is in
        # seconds relative to stim onset, matching h5_2_df's contract).
        # h5_2_df normally does this conversion; called directly here
        # (bypassing h5_2_df's own redundant trial_params read), so the
        # conversion is replicated here instead:
        with h5py.File(h5_path, 'r', rdcc_nbytes=_DEFAULT_RDCC_NBYTES, rdcc_nslots=_DEFAULT_RDCC_NSLOTS) as h5:
            if time_window is not None:
                psth_bins = h5.attrs['psth_bins']
                bin_indices = time_window2bin_indices(time_window, psth_bins)
            else:
                bin_indices = None
            raw = h5_2_dat_array_rsvp(h5, trials=spike_inds.copy(), channels=channels, time_window=bin_indices)
        psth_arr = np.transpose(raw, axes=[2, 0, 1])  # c-by-b-by-s -> s-by-c-by-b
        index_df = pd.DataFrame(pairs_sorted, columns=['trial_num', 'rsvp_num'])

        os.makedirs(cache_dir, exist_ok=True)
        np.save(psth_path, psth_arr)
        index_df.to_hdf(index_path, 'trial_index', mode='w', format='fixed')
        with open(meta_path, 'w') as f:
            json.dump(key_parts, f, indent=1, default=str)
        psth_arr = np.load(psth_path, mmap_mode='r')

    # Attach psth to `group` in its OWN original row order (which may
    # differ from -- and may be a subset of -- the cache's sorted order):
    pair_to_idx = {pair: i for i, pair in enumerate(zip(index_df['trial_num'], index_df['rsvp_num']))}
    row_order = [pair_to_idx[(tn, rn)] for tn, rn in zip(group['trial_num'].astype(int), group['rsvp_num'].astype(int))]
    curr_out = group.copy()
    curr_out['psth'] = [psth_arr[i] for i in row_order]

    # `group`'s own 'psth_bins' column (if present) is per-trial metadata
    # inherited from the *unfiltered* session trial_params -- i.e. the full
    # recorded epoch, regardless of `time_window`. If time_window narrowed
    # what's actually in `psth` above, that column is now stale (still
    # describing the full epoch) and must be replaced with the bin edges
    # that actually correspond to the fetched/cached `psth` array, or
    # downstream code slicing/labeling by 'psth_bins' will misalign against
    # the now-shorter psth arrays. No correction needed when time_window is
    # None, since 'psth' then still spans the full epoch 'psth_bins' already
    # describes. Recomputed independently from the H5's own attrs (cheap,
    # attrs-only) rather than threaded through each cache-hit/miss branch
    # above, so it's correct regardless of which branch produced psth_arr.
    #
    # +1 on the upper slice bound: the full-epoch 'psth_bins' has N+1 edges
    # for N data bins (verified: len(h5.attrs['psth_bins']) ==
    # full_psth.shape[-1] + 1), and downstream code (e.g.
    # trial_avg_psths.ipynb's "Verify PSTH bins" cell) unconditionally drops
    # the last edge via psth_bins[:-1] before comparing against psth's
    # column count. Slicing to bin_indices[1] (matching psth_arr's column
    # count exactly) would leave that convention one edge short after the
    # downstream trim:
    if time_window is not None and 'psth_bins' in curr_out.columns:
        with h5py.File(h5_path, 'r') as h5:
            full_psth_bins = h5.attrs['psth_bins']
        bin_indices = time_window2bin_indices(time_window, full_psth_bins)
        psth_bins_actual = full_psth_bins[bin_indices[0]:bin_indices[1] + 1]
        curr_out['psth_bins'] = [psth_bins_actual] * len(curr_out)

    # chs_meta is a cheap PyTables-format read, always re-fetched fresh
    # regardless of cache hit/miss (no need to cache it separately):
    ch_meta_result = None
    try:
        curr_zero_coords, curr_imro_tbl = h5_2_ch_meta(h5_path)
        curr_zero_coords['monkey'] = monkey
        curr_zero_coords['date'] = date
        curr_imro_tbl['monkey'] = monkey
        curr_imro_tbl['date'] = date
        curr_imro_tbl['ch_idx_glx'] = curr_imro_tbl.index
        ch_meta_result = (pd.DataFrame(curr_zero_coords).T, curr_imro_tbl)
    except Exception as e:
        warnings.warn('Failed to find recording site metadata for {} session {}.'.format(monkey, date))

    return curr_out, ch_meta_result



def sessions2trials_cached(trials_df, unit_type='mua', channels=None, time_window=None,
    cache_root=None, force_refresh=False, max_workers=8):
    """
    Attach PSTH arrays to `trials_df` -- step 3-4 of the recommended
    workflow. `trials_df` must have 'monkey', 'date', 'trial_num', 'rsvp_num'
    columns identifying exactly which stimulus presentations to fetch (e.g.
    the output of `filter_trial_params` applied to
    `sessions2trial_params_cached`'s output), plus whatever other trial
    parameter columns the caller wants preserved in the result.

    Unlike the previous (v1) design, this does NOT take misc_flt/group_defs
    directly -- the PSTH cache key for each session is the exact sorted set
    of (trial_num, rsvp_num) pairs present for that session in `trials_df`.
    This makes the cache shareable across notebooks/callers: two
    independently-built `trials_df` inputs (e.g. from differently-worded but
    semantically equivalent misc_flt/group_defs in different notebooks) that
    resolve to the same underlying tuples hit the same cache entry,
    regardless of how each notebook happened to compute them. Also depends
    on unit_type, channels, time_window, and each session's source HDF5 file
    (mtime+size fingerprint). Downstream-only parameters (class_defs,
    dichotomies, z-scoring, binning) never enter `trials_df` and so can
    never affect this cache.

    Parameters
    ----------
    trials_df : pandas.DataFrame
        Must define 'monkey', 'date', 'trial_num', 'rsvp_num'.

    unit_type : str
        'mua' | 'ks'.

    channels : array-like | None
        Indices of channels/units to get PSTH data for. If None, all
        channels.

    time_window : array-like | None
        2-element [start, stop] in seconds relative to stim onset. If None,
        the entire recorded peristim epoch.

    cache_root : str
        Root directory for the on-disk cache. Required -- e.g.
        `os.path.join(mnt, 'users', 'Dan', 'ephys', 'spike_cache')`.

    force_refresh : bool
        If True, ignore any existing cache entry and re-fetch + overwrite it.

    max_workers : int | None
        Number of sessions to fetch concurrently, via a thread pool -- see
        `sessions2trial_params_cached`'s `max_workers` docstring for the same
        reasoning (I/O-bound work, threads over processes to keep memmapped
        cache-hit arrays lazy, default of 8 not a tuned value). Set to 1 or
        None to fetch serially.

    Returns
    -------
    spikes_df : pandas.DataFrame
        `trials_df`, in its original row order, with a new 'psth' column
        attached (memmapped array views on a cache hit).

    chs_meta : dict
        Same shape as `sessions2trials`'s return value.
    """
    if cache_root is None:
        raise ValueError('cache_root must be specified (no hardcoded default -- '
                          'e.g. os.path.join(mnt, "users", "Dan", "ephys", "spike_cache")).')

    required_cols = {'monkey', 'date', 'trial_num', 'rsvp_num'}
    missing = required_cols - set(trials_df.columns)
    if missing:
        raise ValueError('trials_df is missing required columns: {}'.format(missing))

    groups = list(trials_df.groupby(['monkey', 'date'], sort=False))

    if max_workers is not None and max_workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_one_session_psth, monkey, date, group, unit_type, channels, time_window, cache_root, force_refresh)
                for (monkey, date), group in groups
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            _fetch_one_session_psth(monkey, date, group, unit_type, channels, time_window, cache_root, force_refresh)
            for (monkey, date), group in groups
        ]

    out_dfs = []
    zero_coords_df = pd.DataFrame()
    imro_df = pd.DataFrame()
    for r in results:
        if r is None:
            continue
        curr_out, ch_meta_result = r
        out_dfs.append(curr_out)
        if ch_meta_result is not None:
            curr_zero_coords_row, curr_imro_tbl = ch_meta_result
            zero_coords_df = pd.concat([zero_coords_df, curr_zero_coords_row], axis=0)
            imro_df = pd.concat([imro_df, curr_imro_tbl], axis=0)

    spikes_df = pd.concat(out_dfs, axis=0) if len(out_dfs) > 0 else pd.DataFrame()
    spikes_df.index = np.arange(spikes_df.shape[0])
    spikes_df.loc[:, 'unit_type'] = unit_type

    if zero_coords_df.shape[0] > 0:
        zero_coords_df = zero_coords_df[['monkey', 'date', 'hemisphere', 'hole_id', 'penetration', 'AP', 'DV', 'ML', 'Ang', 'HAng', 'depth']]
    zero_coords_df.loc[:, 'unit_type'] = unit_type
    imro_df.index = np.arange(imro_df.shape[0])
    if imro_df.shape[0] > 0:
        imro_df = imro_df[['monkey', 'date', 'ch_idx_glx', 'bank', 'ref_id', 'ap_gain', 'ap_hipass']]
    imro_df.loc[:, 'unit_type'] = unit_type
    ch_meta = {'zero_coords': zero_coords_df, 'imro_tbl': imro_df}

    return spikes_df, ch_meta
