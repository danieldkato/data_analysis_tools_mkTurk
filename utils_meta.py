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

    str_vec = Path.Path(data_path).name.split('_')
    hemisphere = str_vec[2]
    hole_id = int(str_vec[3].split('H')[1])
    pen_id = int(str_vec[4].split('P')[1])
    ap_coord = float(str_vec[5].split('AP')[1])
    dv_coord = float(str_vec[6].split('DV')[1])
    ml_coord = float(str_vec[7].split('ML')[1])
    ang = float(str_vec[8].split('Ang')[1])
    depth_start = int(str_vec[9].split('Dep')[1].split('-')[0])
    depth_end = str_vec[9].split('Dep')[1].split('-')[1]

    return ap_coord, dv_coord, ml_coord, ang, depth_start

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
