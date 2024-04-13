import numpy as np
import pandas as pd
import re
import itertools
import numbers
from copy import deepcopy
from data_analysis_tools_mkTurk.utils_mkturk import expand_sess_scenefile2stim
from data_analysis_tools_mkTurk.general import find_signed_angle

def filter_stim_trials(d, filters):
    """
    Evaluate which trials of a given stimulus condition satisfy all of any 
    number of criteria. Note that criteria are applied *conjunctively*, i.e., 
    the i-th element of the output array is True only if the i-th trial satisfies
    all conditions.

    Parameters
    ----------
    d : dict
        Dictionary corresponding to a single stimulus condition. Same format as 
        single sub-dict of dict encoded in stim_info_sess file.
        
    filters : list
        List of 2-element tuples. The first element of each tuple is a key into 
        input dict `d`, and the second element is a lambda expression specifying 
        a function that maps individual elements of the specified key to True
        or False. Note that all keys specified in `filters` must map to 
        arrays with the same number of elements. Also note that the filters
        are applied conjunctively, i.e., the i-th element of the output array
        F will only be True if the i-th element of *all* the specified keys
        evaluate to True under their corresponding lambda functions. 

    Returns
    -------
    F : numpy.ndarray
        Boolean array specifying which trials satisfy all conditions specified
        in `filters`.

    """
    
    # Get overall number of trials for current stim:
    n_trials = len(d['trial_num'])
    
    # Initialize filter:
    F = np.ones(n_trials, dtype=bool)

    # Iterate over criteria:
    if filters is not None:
        for (key, fcn) in filters:
            curr_filter = list(map(fcn, d[key])) 
            curr_filter = np.array(curr_filter)
            F = F & curr_filter # Conjoin current filter with previous filter
        
        # Get absolute trial nums of retained trials:
        T = d['trial_num'][F]
    else:
        F = F
        T = d['trial_num']
    
    return F, T



def expand_classes(classes, scenefile_metadata):
    
    classes_expanded = []
    for curr_class in classes:
        curr_class_expanded = []
        for stim in curr_class:
            if '.json' in stim:
                stim_ids = expand_sess_scenefile2stim(scenefile_metadata, [stim])
                curr_class_expanded.append(stim_ids)
            else:
                curr_class_expanded.append(stim)
        curr_class_expanded = np.unique(curr_class_expanded)
        classes_expanded.append(curr_class_expanded)
        
    return classes_expanded



def expand_classesh5(classes, scenefiles, stim_ids, scene_by_stim):
    
    classes_expanded = []
    for curr_class in classes:
        curr_class_expanded = []
        for stim in curr_class:
            if '.json' in stim:
                scenefile_idx = np.where([np.array(scenefiles)==stim][0])[0][0] # Find index of current scenefile name in `scenefiles` list    
                t = np.where(scene_by_stim[scenefile_idx, :])[0] # Find column indices where `scene_by_stim` is True in row 1scenefile_idx`
                curr_stim_ids = np.array(stim_ids)[t]
                curr_class_expanded = curr_class_expanded + list(curr_stim_ids)    
            else:
                curr_class_expanded.append(stim)
        curr_class_expanded = np.unique(curr_class_expanded)
        classes_expanded.append(curr_class_expanded)
    
    return classes_expanded



def get_class_trials(classes, sess_metadata, trial_filters=None):
    
    trials_by_class = []
    used_stim = []
    for cx, curr_class in enumerate(classes):
        curr_class_trials = []
        for stim in curr_class:
            if trial_filters is not None:
                curr_filter, curr_stim_kept_trials = filter_stim_trials(sess_metadata[stim], trial_filters) 
                sess_metadata[stim]['kept_trials'] = curr_stim_kept_trials
                sess_metadata[stim]['curr_filter'] = curr_filter
            else:
                curr_stim_kept_trials = sess_metadata[stim]['trial_num']
            if len(curr_stim_kept_trials) > 0:
                used_stim.append(stim)
            curr_class_trials = curr_class_trials + list(curr_stim_kept_trials)
            curr_class_trials = list(np.sort(np.unique(curr_class_trials)))
        trials_by_class.append(curr_class_trials)
    
    return trials_by_class, sess_metadata, used_stim



def get_polyhaven_views(stimuli):
    """
    Create a hierarchically-organized dict of polyhaven environments, camera 
    positions within polyhaven environment, then camera direction within camera 
    position. 

    Parameters
    ----------
    stimuli : list
        List of individual stimulus condition names, including polyhaven stimuli
        names specifying polyhaven environment index, camera position, and 
        camera direction (i.e. camera target position). 

    Returns
    -------
    view_hierarchy : pandas.core.frame.DataFrame
        Dataframe of viewpoints included in input stim IDs. Defines following
        columns:
            
            environment : str
                Polyhaven environment name.
              
            posX: float
                Camera position X-coordinate.

            posY: float
                Camera position Y-coordinate.

            posZ: float
                Camera position Z-coordinate.
                
            pos_str: str
                String summarizing camera position coordinates.
                
            tgtX: float
                Lookat position X-coordinate.

            tgtY: float
                Lookat position Y-coordinate.

            tgtZ: float
                Lookat  position Z-coordinate.
                
            tgt_str: str
                String summarizing lookat coordinates.
                
            angle : float
                Angle between lookat position and origin (from camera position), 
                in degrees.
    """
    
    views = pd.DataFrame(columns=['environment', 'posX', 'posY', 'posZ', 'pos_str', 'pos_angle', 'targetX', 'targetY', 'targetZ', 'target_str', 'target_angle'])
    
    # Define some useful regular expressions:
    phaven_regex = '(polyhaven|background)/\d+'
    cam_pos_regex = 'camera00_posX_-{0,1}\\d+\\.*\\d*_posY_-{0,1}\\d+\\.*\\d*_posZ_-{0,1}\\d+\\.*\\d*'
    cam_dir_regex = 'targetX_-{0,1}\\d+\\.*\\d*_targetY_-{0,1}\\d+\\.*\\d*_targetZ_-{0,1}\\d+\\.*\\d*'
    
    # Get all polyhaven environments:
    phavens = [re.search(phaven_regex,x).group() for x in stimuli if re.search(phaven_regex,x) is not None]
    phavens = np.unique(phavens)
    print('phavens={}'.format(phavens))    
    
    # Within all environments, find all camera positions:
    for env in phavens:
        
        # Retrieve all camera positions within in current polyhaven environment:
        curr_phaven_stim = [x for x in stimuli if env in x]
        print('query = {}'.format('polyhaven/'+str(env)))
        print('curr_phaven_stim = {}'.format(curr_phaven_stim))
        cam_pos_matches = [re.search(cam_pos_regex,x).group() for x in curr_phaven_stim if re.search(cam_pos_regex,x) is not None]
        cam_positions = np.unique(cam_pos_matches)
        print('cam_positions = {}'.format(cam_positions))
        
        # Within all camera positions, find all different directions it's pointed at:
        for curr_pos in cam_positions:
                      
            posX, posY, posZ = extract_xyz(curr_pos, base_str='pos')
            pos = np.array([posX, posY, posZ])

            curr_pos_stim = [x for x in curr_phaven_stim if curr_pos in x]
            curr_tgt_matches = [re.search(cam_dir_regex,x).group() for x in curr_pos_stim if re.search(cam_dir_regex,x) is not None]                
            curr_tgt_matches = np.unique(curr_tgt_matches)            
            
            # Get signed angle between camera position and "front" of room:
            front = np.array([5.5, 0, 0]) # < Arbitrarily designating this "front" of room
            center = np.array([0, 0, 0])
            center2pos = pos - center
            center2front = front - center
            pos_angle = find_signed_angle(center2front, center2pos, axis=1)
            
            # Sort lookats by angle wrt. origin:
            angles = []
            for tgt_str in curr_tgt_matches:
                
                print(tgt_str)
                
                # Get X, Y, Z coords of lookat:
                tgtX, tgtY, tgtZ = extract_xyz(tgt_str, base_str='target')
                tgt = np.array([tgtX, tgtY, tgtZ])
            
                # Offest relative to current camera position:
                center = center - pos
                tgt = tgt - pos
                
                # Get signed angle between ref and target:
                target_angle = find_signed_angle(center, tgt, axis=1)
            
                # Define current row of dataframe:
                curr_row = dict() 
                curr_row['environment'] = env
                curr_row['posX'] = posX
                curr_row['posY'] = posY
                curr_row['posZ'] = posZ
                curr_row['pos_str'] = curr_pos
                curr_row['pos_angle'] = pos_angle
                curr_row['targetX'] = tgtX
                curr_row['targetY'] = tgtY
                curr_row['targetZ'] = tgtZ
                curr_row['target_str'] = tgt_str
                curr_row['target_angle'] = target_angle
                views.loc[len(views.index)] = curr_row
    
            
    return views



def reverse_lookup_rsvp_stim(tgt_stim, stim_ids, stim_indices):
    """
    Find which trials and RSVP presentations in a session are of a given 
    stimulus condition. 

    Parameters
    ----------
    tgt_stim : str
        Target stimulus condition to search for.
    
    stim_ids : list
        s-element list of stimulus descriptions.
    
    stim_indices : numpy.ndarray
        t-by-r array of stimulus condition indices, where t is the number of 
        trials in a session and r is the number of RSVP stimuli per trial 
        (assumed to be the same for every trial, usually 3). 

    Returns
    -------
    matches_tgt : numpy.ndarray
        t-by-r boolean array. The i,j-th element is True if and only if the j-th
        RSVP stim on the i-th trial is the target stimulus condition. 

    """
    
    fcn = lambda x : not np.isnan(x) and stim_ids[int(x)] == tgt_stim 
    matches_tgt = np.array([list(map(fcn,x)) for x in stim_indices])
    
    return matches_tgt



def create_stim_idx_mat(sess_meta):
    """
    

    Parameters
    ----------
    sess_meta : dict
        Same format as YJs 'stim_info_sess' files.

    Returns
    -------
    stim_inds_mat : numpy.ndarray
        t-by-r matrix, where t is the number of trials and r is the number of
        RSVP stim per trial within a session (assumed to be the same for all trials).
        The i,j-th element is s if and only if the j-th RSVP stimulus of the 
        i-th trial is the s-th stimulus condition, where stimulus conditions
        are ordered the same as in list(sess_meta.keys()). 

    """
    
    # Create dataframe of trial num, rsvp num, and stim id:
    trial_df = create_trial_df(sess_meta)
    
    # Get list of unique stimulus conditions in current session:
    stim_ids = list(sess_meta.keys())
    
    # Initialize matrix of stimulus indices:
    n_trials = max(trial_df.trial_num) + 1
    n_rsvp = max(trial_df.rsvp_num) + 1
    stim_inds_mat = np.empty((n_trials, n_rsvp))
    stim_inds_mat[:] = np.nan
    
    # Iterate over stimulus conditions:
    for sx, stim_id in enumerate(stim_ids):
        
        # Get matching rows:
        curr_stim_rows = trial_df[trial_df.stim_id==stim_id]
        
        # Iterate over matching rows:
        for idx, row in curr_stim_rows.iterrows():
            
            curr_trial = row['trial_num']
            curr_rsvp = row['rsvp_num']
            stim_inds_mat[curr_trial, curr_rsvp] = sx
        
    return stim_inds_mat



def create_trial_df(sess_meta):
    
    # Initialize output dataframe:
    trial_df = pd.DataFrame(columns=['trial_num', 'rsvp_num', 'stim_id'])
    
    # Iterate over stimulus ids, populate rows of dataframe:
    stim_ids = list(sess_meta.keys())
    for stim_id in stim_ids:
        
        curr_stim_trials = sess_meta[stim_id]['trial_num'] 
        
        # Iterate over rows of current stimulus:
        for tx, trial in enumerate(curr_stim_trials):
            
            curr_trial_dict = dict()
            curr_trial_dict['trial_num'] = trial
            curr_trial_dict['rsvp_num'] = sess_meta[stim_id]['rsvp_num'][tx]
            curr_trial_dict['stim_id'] = stim_id
            curr_trial_dict['dur'] = sess_meta[stim_id]['dur'][tx]
            
            trial_df.loc[len(trial_df.index)] = curr_trial_dict
            
    # Sort by trial then RSVP num:
    trial_df = trial_df.sort_values(['trial_num', 'rsvp_num'], ignore_index=True)
            
    return trial_df



def stim_meta_2_df(stim_meta):
    return



def extract_xyz(lookat_str, base_str='pos'):
    
    dims = ['X', 'Y', 'Z']
    
    coords = []
    for dim in dims:
        
        regex = '{}{}'.format(base_str, dim) + '_-{0,1}\\d+\\.*\d*'
        matches = re.search(regex, lookat_str).group() # < Assuming unique!
        coord_str = matches[len(base_str)+2:]
        coords.append(float(coord_str))

    return coords[0], coords[1], coords[2]     

    

def stim_ids_2_rsvp_inds(trial_df, stim_ids):
    """
    Get array of trial/RSVP indices associated with any in a list of stimulus
    IDs. 

    Parameters
    ----------
    trial_df : pandas.core.frame.DataFrame
        Dataframe of trial parameters. Must define columns 'trial_num', 
        'rsvp_num', and 'stim_id'.
    
    stim_ids : array-like
        List or array of individual stim IDs.

    Returns
    -------
    inds : numpy.ndarray
        s-by-2 array, where s is the number of all indivdual RSVP stimulus
        presentations matching any of the stim IDs passed in `stim_ids`. Col 0:
        trial number, col 1: RSVP stim number. 

    """
    
    if type(stim_ids) == list:
        stim_ids = np.array(stim_ids)
    
    B = [i for i,x in trial_df.iterrows() if np.any(x.stim_id==stim_ids)]
    rows = trial_df.loc[B]
    inds = np.array([rows.trial_num, rows.rsvp_num]).T
    
    return inds



def stim_groups_2_rsvp_inds(trial_df, groups):
    """
    Get trial/RSVP numbers associated with each of multiple stimulus groups, where 
    a stimulus group itself may consist of multiple individual stim IDs.  

    Parameters
    ----------
    trial_df : pandas.core.frame.DataFrame
        Dataframe of trial parameters. Must define columns 'trial_num', 
        'rsvp_num', and 'stim_id'.
    
    groups : list
        g-element list of lists, where g is the number of stimulus groups. Each 
        inner list should be a list of individual stim IDs.

    Returns
    -------
    grouped_inds : list
        g-element list of arrays, each corresponding to a stimulus group. Each
        array is s_i-by-2, where s_i is the total number of individual stimulus
        presentations associated with any stim ID in the i-th stimulus group.
        Col 0: trial number, col 1: RSVP number.

    """
    
    grouped_inds = []
    for group in groups:
        inds = np.array(trial_df[trial_df.stim_id.isin(group)][['trial_num', 'rsvp_num']])
        grouped_inds.append(inds)
    
    return grouped_inds



def session_dicts_2_df(data_dicts):
    
    # Get unique scenefiles:
    scenefiles = [data_dicts[x]['scenefile'] for x in np.arange(len(data_dicts))]
    scenefiles = np.unique(scenefiles)
    
    # Iterate over scenefiles:
    dfs = []
    for scenefile in scenefiles:
        curr_scenefile_dicts = [data_dicts[x] for x in np.arange(len(data_dicts)) if data_dicts[x]['scenefile']==scenefile]
        curr_scenefile_df = data_dicts_2_df(curr_scenefile_dicts)
        dfs.append(curr_scenefile_df)
        
    # Merge dataframes:
    df = pd.concat(dfs, axis=0)
    
    # Sort by trial number then RSVP number:
    df = df.sort_values(by=['trial_num', 'rsvp_num'])
    
    return df



def data_dicts_2_df(dict_list):
    
    # Get list of all scene elements and their attributes:
    attrs = list(dict_list[0]['stim_info'].keys()) # < Hack; assuming these are the same for all elements of dict_list
    els = list(dict_list[0]['stim_info']['type'].keys()) # < Hack; again assuming these are the same for all elements of dict_list AND attributes
    
    n_attrs = len(attrs)
    n_els = len(els)
    
    # Generate list of all combinations of elements and attributes:
    P = list(itertools.product(attrs, els))
    S = ['_'.join(np.flip(p)) for p in P]
    
    # Initialize dataframe:
    n_dicts = len(dict_list)
    df = pd.DataFrame(columns=S, index=np.arange(n_dicts))
    
    # Iterate over attributes:
    for i, attr in enumerate(attrs):
        
        col_offset = i*n_els
        curr_att_vals = np.array([x['stim_info'][attr] for x in dict_list])
        df.iloc[0:len(dict_list), col_offset:col_offset+n_els] = curr_att_vals
        
        # Convert columns to appropriate type:
        curr_cols = df.columns[col_offset:col_offset+n_els]
        for col in curr_cols:
            
            # Make all values of current column of same type (needed to save to HDF5 later):
            if np.any([type(x)==str for x in df[col]]):
                curr_type = str
            elif np.any([isinstance(x, numbers.Number) for x in df[col]]):
                curr_type = np.float32
                
            df[col] = df[col].apply(curr_type)

    # Eliminate all-nan columns:
    is_nan_col = [np.all([str(x)=='nan' for x  in df[y].to_numpy()]) for y in df.columns]
    nan_col_indices = np.where(is_nan_col)
    nan_cols = df.columns[nan_col_indices]
    df = df.drop(columns=nan_cols)

    # Add trial and RSVP number:
    trial_num = np.array([x['trial_num'] for x in dict_list])
    rsvp_num = np.array([x['rsvp_num'] for x in dict_list])
    scenefile = np.array([x['scenefile'] for x in dict_list])
    stim_info_short = np.array([x['stim_info_short'] for x in dict_list])
    df_basic = pd.DataFrame({'trial_num':trial_num, 'rsvp_num':rsvp_num, 'scenefile':scenefile, 'stim_id':stim_info_short})
    
    df_out = pd.concat([df_basic, df], axis=1)
    
    return df_out



def add_bg_info(df):
    
    cols = df.columns
    bkg_cols = [x for x in cols if 'bkg' in x and 'meta' in x]
    bg_df = df[bkg_cols]
    bg_df.index = np.arange(bg_df.shape[0])
    bg_df = bg_df.fillna(False)
    
    for col in bkg_cols:
        
        # Replace file path with background name:
        bkg_name = col[0:-5]
        curr_inds = np.where(bg_df[col])[0]
        bg_df.loc[curr_inds, col] = bkg_name
        
        # Replace any False with empty string '':
        empty_inds = np.where(bg_df[col]==False)[0]
        bg_df.loc[empty_inds, col] = ''
    
    bg_name_col = np.array(bg_df.agg(''.join, axis=1))
    df['background'] = bg_name_col
    
    return df
    
    
    
def add_cam_azimuth(df):
   
    center = np.array([0, 0, 0])
    center = np.expand_dims(center, axis=0)
    C = np.matlib.repmat(center, df.shape[0], 1) # < repeats-by-3
    
    # Get matrix of lookat coordinates:
    T = [np.array([row['camera00_targetX'], row['camera00_targetY'], row['camera00_targetZ']]) for i, row in df.iterrows()]
    T = np.array(T)  # < repeats-by-3
    
    # Get matrix of camera position coordinates:
    P = [np.array([row['camera00_posX'], row['camera00_posY'], row['camera00_posZ']]) for i, row in df.iterrows()]
    P = np.array(P) # < repeats-by-3
    
    # Offset everything by camera position:
    T = T - P
    C = C - P
    
    # Compute angles:
    angles = [find_signed_angle(T[x], C[x]) for x in np.arange(df.shape[0])]
    df['cam_azimuth'] = angles
    
    return df



def sess_meta_dict_2_df(sess_meta_dict):
    
    cols = ['trial_num', 'rsvp_num', 'reward_bool']
    A = [np.array([sess_meta_dict[y][x] for x in cols]).T for y in list(sess_meta_dict.keys())]
    A = np.concatenate(A)
    P = pd.DataFrame(data=A, columns=cols)
    P = P.sort_values(by=['trial_num', 'rsvp_num'])
    
    return P

