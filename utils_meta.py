#### useful functions for retrieving filepaths and other session info

import numpy as np
import os
import pathlib as Path
import re
from natsort import os_sorted
import pickle
import json
from data_analysis_tools_mkTurk.utils_mkturk import * 

def get_recording_path(base_data_path, monkey, date,depth = 4):
    # given the base data path, monkey, and date,
    # returns the filepath where the spikeglx binary files are stored
    print(Path.Path(base_data_path,monkey))
    long_path = next(Path.Path(base_data_path,monkey).glob('*' + date + '*'))
    
    sub_lists = [x[0] for x in os.walk(long_path) if len(Path.Path(x[0]).parents)-len(Path.Path(base_data_path).parents) ==depth]
    filepath = []
    for s in sub_lists:
        result = re.search('\Dep(.*?)\_', Path.Path(s).name)
        if result is not None and 'dk' not in Path.Path(s).name:
            depth_start = result.group(1).split('-')[0]
            depth_end = result.group(1).split('-')[1]

            if int(depth_start) == int(depth_end):
                filepath.append(s)

    return filepath


def get_scenefiles_sess(base_data_path, monkey, date):
    # returns scenefiles used in the sesssion
    folder = os_sorted(Path.Path(base_data_path, monkey).glob('*' +date + '*'))
    scenefiles = []
    for f in folder:
        print(f)
        if os.path.isdir(Path.Path(base_data_path, monkey, f)):
            behav_file_list= os_sorted(Path.Path(f).glob('*.json'))
            for b_f in behav_file_list:
                m = json.load(open(b_f,'rb'))
                if m['TASK']['RewardStage'] == 1:
                    scenefiles.extend(m['TASK']['ImageBagsSample'])  

    return np.unique(np.array(scenefiles))

def get_stims_sess(base_data_path, monkey, date):
    # returns all short stimulus id used in behavior files  
    folder = os_sorted(Path.Path(base_data_path, monkey).glob('*' +date + '*'))
    
    stims = []
    for f in folder:
        if os.path.isdir(Path.Path(base_data_path, monkey, f)):
            behav_file_list= os_sorted(Path.Path(f).glob('*.json'))
            if len(behav_file_list) ==  0:
                print('no behavior file in the folder')
                continue

            for b_f in behav_file_list:
                m = json.load(open(b_f,'rb'))
                if m['TASK']['RewardStage'] == 1:
                    for samplescene in m['SCENES']['SampleScenes']:
                        stims.extend(list(map(gen_short_scene_info,gen_scene_df(samplescene))))

    return np.unique(np.array(stims))

def get_allsess_scenefile(base_data_path,monkey,scenefile_list):
    folders = os_sorted([f for f in next(os.walk(base_data_path/monkey))[1] if monkey in f])

    allsess = []
    for f in folders:
        date = f.split('_')[1]
        data_path = get_recording_path(base_data_path, monkey, date,depth = 3)
        n_recordings = len(data_path)
        for dp in data_path:
            scenefiles_sess = get_scenefiles_sess(base_data_path, monkey, date)

            if len(set(scenefiles_sess).intersection(scenefile_list))>0:
                allsess.append(f)
                print(f)

    return allsess

def get_allsess_stim(base_data_path,monkey,stim_list):
    folders = os_sorted([f for f in next(os.walk(base_data_path/monkey))[1] if monkey in f])

    allsess = []
    for f in folders:
        date = f.split('_')[1]
        data_path = get_recording_path(base_data_path, monkey, date,depth = 3)
        n_recordings = len(data_path)
        for dp in data_path:
            stim_keys_sess = get_stims_sess(base_data_path, monkey, date)

            if len(set(stim_keys_sess).intersection(stim_list))>0:
                allsess.append(f)
                print(f)

    return allsess

def get_coords_sess(base_data_path, monkey, date):
    # returns hole id, ap, dv, ml coordinates, angle, and depth of recording

    data_path = get_recording_path(base_data_path, monkey, date,depth = 4)[0]

    name = Path.Path(data_path).name
    
    # Define field names and corresponding search patterns:
    fnames = np.array(['hole_id', 'penetration', 'AP', 'DV', 'ML', 'Ang', 'HAng']) 
    patterns = ['_H\d+\.*\d*_', '_P\d+\.*\d*_', 'AP\d+\.*\d*', 'DV\d+\.*\d*', 
                'ML\d+\.*\d*', '[^H]Ang\d+\.*\d', 'HAng\d+\.*\d*']
    regex_lut = pd.DataFrame({'regex':patterns}, index=fnames)
    
    # Iterate over numeric fields (except depth):
    zero_coord_series = pd.Series()
    for idx, row in regex_lut.iterrows():
        matches = re.findall(row.regex, name)
        if len(matches) == 1:
            val = float(re.search('\d+\.*\d*', matches[0]).group())
        else:
            val = None
        zero_coord_series[idx] = val

    # Find depth (requires separate treatment from other numeric parameters bc
    # filenames include both starting and stop depth):
    depth_regex = 'Dep\d+-\d+'
    depth_matches = re.findall(depth_regex, name)
    if len(depth_matches) == 1:
        depth_vals = re.findall('\d+', depth_matches[0])
        depth = float(depth_vals[1])
    else:
        depth = None
    zero_coord_series['depth'] = depth
        
    # Find brain hemisphere:
    hemisphere_str = re.findall('_(L|R)_', name)
    if len(hemisphere_str) == 1:
        hemisphere = re.search('(L|R)', hemisphere_str[0]).group()
    else:
        hemisphere = None
    zero_coord_series['hemisphere'] = hemisphere

    return zero_coord_series

def gen_meta_behavior(base_data_path, monkey):
    folders = os_sorted(Path.Path(base_data_path, monkey).glob(monkey + '*'))
    df_behavior = []
    columns_behavior = ['sub', 'date','behavior file', 'scenefile']
    for folder in folders:
        # get meta info on all behavior files in the path 
        date_str = Path.Path(folder).name.split('_')[1]
        behav_file_list= os_sorted(Path.Path(folder).glob('*.json'))
        if len(behav_file_list) == 0:
            print('no behavior file in this folder')
            for b_f in behav_file_list:
                m = json.load(open(b_f,'rb'))
                if m['TASK']['RewardStage'] == 1:
                    scenefile = m['TASK']['ImageBagsSample']
                else:
                    scenefile = 'calibration'
                df_behavior.append([monkey, date_str, b_f.stem,scenefile ])         

    df_behavior = pd.DataFrame(df_behavior, columns = columns_behavior)

    return df_behavior 

def update_meta_behavior(base_data_path,monkey,df_behavior):
    folders = os_sorted(Path.Path(base_data_path, monkey).glob(monkey + '*'))
    all_dates = [Path.Path(folder).name.split('_')[1] for folder in folders]
    dates_diff = set(list(map(str,df_behavior['date']))).symmetric_difference(all_dates)

    for date in dates_diff:
        folder  = os_sorted(Path.Path(base_data_path,monkey).glob('*' + date + '*'))[0]
        behav_file_list= os_sorted(Path.Path(folder).glob('*.json'))
        if len(behav_file_list) == 0:
            print('no behavior file in this folder')
        else:
            for b_f in behav_file_list:
                m = json.load(open(b_f,'rb'))
                if m['TASK']['RewardStage'] == 1:
                    scenefile = m['TASK']['ImageBagsSample']
                else:
                    scenefile = 'calibration'
                df_behavior.append([monkey, date, b_f.stem,scenefile ])     

    return df_behavior

def init_dirs(base_data_path, monkey, date, base_save_out_path):

    data_path = get_recording_path(base_data_path, monkey, date)

    n_recordings = len(data_path)
    print(n_recordings, ' recordings found')
    if n_recordings > 1:
        data_ind = input('multiple recording sessions. Please select a number between 0 and' + str(len(data_path)-1))  # waiting for user input
        data_path = Path.Path(data_path[int(data_ind)])
    else:
        data_path = Path.Path(data_path[0])
        
    penetration = data_path.relative_to(base_data_path/monkey).as_posix().split('/')[0]
    print(data_path)

    print('\nData path found: '+ str(data_path.exists()))

    save_out_root_path = base_save_out_path / monkey
    save_out_path = Path.Path(save_out_root_path, penetration)
    if n_recordings> 1:
        save_out_path = Path.Path(save_out_root_path, penetration, data_path.parts[len(data_path.parts)-2])
    print('All works will be saved to')
    print(save_out_path)
    os.makedirs(save_out_path, exist_ok=True)

    plot_save_out_path =  base_save_out_path/ (monkey + '_plots') / save_out_path.name

    if n_recordings > 1:
        plot_save_out_path = Path.Path(plot_save_out_path, data_path.name)
    print('All plots will be saved to ')
    print(plot_save_out_path)
    if not plot_save_out_path.exists():
        os.makedirs(plot_save_out_path)

    return data_path, save_out_path, plot_save_out_path

def find_channels(directory, prefix=None):
    """
    Find channels for which input directory contains preprocessed data. Assumes 
    that input directory includes files containing preprocessed data from one 
    channel each (e.g. pickle files containing PSTHs). Note however that there
    may be multiple files containing different data from a single channel. 

    Parameters
    ----------
    directory : str
        Path to directory to search.
        
    prefix : str, optional
        Use to restrict search to files containing certain strings immediately 
        following channel number. E.g., if prefix='psth', then will only search
        files whose names begin 'ch<nnn>_psth'. The default is None.

    Returns
    -------
    chans : numpy.ndarray
        Array of channels in current directory.

    """
    if prefix is None:
        prefix = ''
    
    # Define regex:
    regex = 'ch\d{3}' + prefix
    
    # List directory contents:
    filenames = os.listdir(directory)
    
    # Find patterns matching regex:
    prefixes = [re.search(regex,x).group() for x in filenames if re.search(regex,x) is not None]
    indices = [int(x[2:5]) for x in prefixes] # convert channel numbers to int
    chans = np.unique(indices)
    
    return chans

def get_all_metadata_sess(preprocessed_data_path):
    
    files = os.listdir(preprocessed_data_path)

    # Get session metadata:
    sess_meta_path = os.path.join(preprocessed_data_path, 'stim_info_sess') # < Hack; assuming this always exists
    sess_meta = pickle.load(open(sess_meta_path, 'rb'))
        
    # Get stim metadata:
    psth_stim_meta_regex = 'psth_stim_meta'
    psth_stim_meta_matches = [re.search(psth_stim_meta_regex,x).group() for x in files if re.search(psth_stim_meta_regex,x) is not None]
    stim_meta_file = psth_stim_meta_matches[0] # < Assume metadata is the same for all channels
    stim_meta_path = os.path.join(preprocessed_data_path, stim_meta_file) # < Hack; assuming this always exists
    stim_meta = pickle.load(open(stim_meta_path, 'rb'))

    # Get scenefile metadata:
    psth_scenefile_meta_regex = 'psth_scenefile_meta'
    psth_scenefile_meta_matches = [re.search(psth_scenefile_meta_regex,x).group() for x in files if re.search(psth_scenefile_meta_regex,x) is not None]
    scenefile_meta_file = psth_scenefile_meta_matches[0] # < Assume metadata is the same for all channels
    scenefile_meta_path = os.path.join(preprocessed_data_path, scenefile_meta_file) # < Hack; assuming this always exists
    scenefile_meta = pickle.load(open(scenefile_meta_path, 'rb'))
    
    return sess_meta, scenefile_meta, stim_meta



def scenefile2rsvp_inds(data_dicts, scenefile):
    
    data_dicts_sfile = [data_dicts[x] for x in np.arange(len(data_dicts)) if data_dicts[x]['scenefile']==scenefile]
    trial_nums = [x['trial_num'] for x in data_dicts_sfile]                                                         
    rsvp_nums = [x['rsvp_num'] for x in data_dicts_sfile]                                                         
    
    A = np.array([trial_nums, rsvp_nums]).T
    
    return A

    