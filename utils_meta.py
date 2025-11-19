#### useful functions for retrieving filepaths and other session info

import numpy as np
import os
import pathlib as Path
import re
from natsort import os_sorted
import pickle
import json
from data_analysis_tools_mkTurk.utils_mkturk import * 
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta 

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
    patterns = {'hole_id' : '_H\d+\.*\d*_', 
                'penetration' : '_P\d+\.*\d*_', 
                'AP' : 'AP-{0,1}\d+\.*\d*', 
                'DV' : 'DV-{0,1}\d+\.*\d*', 
                'ML' : 'ML-{0,1}\d+\.*\d*', 
                'Ang'  : '[^H]Ang\d+\.*\d', 
                'HAng' : 'HAng\d+\.*\d*'}
    regex_lut = pd.DataFrame({'regex':patterns.values()}, index=patterns.keys())
    
    # Iterate over numeric fields (except depth):
    zero_coord_series = pd.Series()
    for idx, row in regex_lut.iterrows():
        matches = re.findall(row.regex, name)
        if len(matches) == 1:
            val = float(re.search('-{0,1}\d+\.*\d*', matches[0]).group())
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



# read imroTbl 
def read_imroTbl(meta):
    n_chans = int(meta['nSavedChans'])-1
    imroTbl = re.findall(r'([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)',
                              meta['imroTbl'])
    
    arr = np.nan * np.ones((n_chans,5))
    for n,row in enumerate(imroTbl):
        arr[n,0] = int(row.split(' ')[0]) # channel #
        arr[n,1] = int(row.split(' ')[1]) # bank #
        arr[n,2] = int(row.split(' ')[2]) # ref id
        arr[n,3] = np.float32(row.split(' ')[3]) # ap gain
        arr[n,4] = np.float32(row.split(' ')[4]) # lf gain

    return arr 



def read_snsChanMap(meta):
    n_chans = int(meta['nSavedChans'])-1
    snsChanMap = re.findall(r'([0-9]*;[0-9]*:[0-9]*)',
                              meta['snsChanMap'])
    
    arr = np.nan * np.ones((n_chans,2),dtype = int)
    for n,row in enumerate(snsChanMap):
        if n < n_chans: # last entry is sync
            arr[n,0] = int(row.split(';')[1].split(':')[0]) # channel #
            arr[n,1] = int(row.split(';')[1].split(':')[1]) # actual channel #

    return arr 



def get_chanmap(data_path):
    
    meta_iter = data_path.glob('*ap.meta')   
    meta_path = next(meta_iter) 

    meta = readMeta(meta_path)
    if ('fileTimeSecs' in meta.keys()): 
        Fs = float(meta['imSampRate'])
        filesize = float(meta['fileSizeBytes'])
        file_len_sec = float(meta['fileTimeSecs'])
        file_len_samps = int(file_len_sec * Fs)
        n_chans = int(meta['nSavedChans'])-1
    else: 
        print(f'meta file time not saved. recovering this info from OS file size:')
        # get filesize from the os due to frozen SpikeGLX
        n_chans = 384
        Fs = 30000.0
        filesize_os = os.path.getsize(bin_path)
        file_len_pred = filesize_os/(Fs*2*(n_chans+1)) # 385 channels, 2 bytes per sample
        print(f'file size: {filesize_os}')
        print(f'file len (sec): {file_len_pred}')
        file_len_sec = file_len_pred
        filesize = filesize_os
        meta['fileSizeBytes'] = filesize
        meta['nSavedChans'] = 385
        # checkerboard style
        # meta['imroTbl'] = '(0,384)(0 0 1 500 250 0)(1 0 1 500 250 0)(2 0 1 500 250 0)(3 0 1 500 250 0)(4 0 1 500 250 0)(5 0 1 500 250 0)(6 0 1 500 250 0)(7 0 1 500 250 0)(8 0 1 500 250 0)(9 0 1 500 250 0)(10 0 1 500 250 0)(11 0 1 500 250 0)(12 0 1 500 250 0)(13 0 1 500 250 0)(14 0 1 500 250 0)(15 0 1 500 250 0)(16 0 1 500 250 0)(17 0 1 500 250 0)(18 0 1 500 250 0)(19 0 1 500 250 0)(20 0 1 500 250 0)(21 0 1 500 250 0)(22 0 1 500 250 0)(23 0 1 500 250 0)(24 0 1 500 250 0)(25 0 1 500 250 0)(26 0 1 500 250 0)(27 0 1 500 250 0)(28 0 1 500 250 0)(29 0 1 500 250 0)(30 0 1 500 250 0)(31 0 1 500 250 0)(32 0 1 500 250 0)(33 0 1 500 250 0)(34 0 1 500 250 0)(35 0 1 500 250 0)(36 0 1 500 250 0)(37 0 1 500 250 0)(38 0 1 500 250 0)(39 0 1 500 250 0)(40 0 1 500 250 0)(41 0 1 500 250 0)(42 0 1 500 250 0)(43 0 1 500 250 0)(44 0 1 500 250 0)(45 0 1 500 250 0)(46 0 1 500 250 0)(47 0 1 500 250 0)(48 0 1 500 250 0)(49 0 1 500 250 0)(50 0 1 500 250 0)(51 0 1 500 250 0)(52 0 1 500 250 0)(53 0 1 500 250 0)(54 0 1 500 250 0)(55 0 1 500 250 0)(56 0 1 500 250 0)(57 0 1 500 250 0)(58 0 1 500 250 0)(59 0 1 500 250 0)(60 0 1 500 250 0)(61 0 1 500 250 0)(62 0 1 500 250 0)(63 0 1 500 250 0)(64 0 1 500 250 0)(65 0 1 500 250 0)(66 0 1 500 250 0)(67 0 1 500 250 0)(68 0 1 500 250 0)(69 0 1 500 250 0)(70 0 1 500 250 0)(71 0 1 500 250 0)(72 0 1 500 250 0)(73 0 1 500 250 0)(74 0 1 500 250 0)(75 0 1 500 250 0)(76 0 1 500 250 0)(77 0 1 500 250 0)(78 0 1 500 250 0)(79 0 1 500 250 0)(80 0 1 500 250 0)(81 0 1 500 250 0)(82 0 1 500 250 0)(83 0 1 500 250 0)(84 0 1 500 250 0)(85 0 1 500 250 0)(86 0 1 500 250 0)(87 0 1 500 250 0)(88 0 1 500 250 0)(89 0 1 500 250 0)(90 0 1 500 250 0)(91 0 1 500 250 0)(92 0 1 500 250 0)(93 0 1 500 250 0)(94 0 1 500 250 0)(95 0 1 500 250 0)(96 0 1 500 250 0)(97 0 1 500 250 0)(98 0 1 500 250 0)(99 0 1 500 250 0)(100 0 1 500 250 0)(101 0 1 500 250 0)(102 0 1 500 250 0)(103 0 1 500 250 0)(104 0 1 500 250 0)(105 0 1 500 250 0)(106 0 1 500 250 0)(107 0 1 500 250 0)(108 0 1 500 250 0)(109 0 1 500 250 0)(110 0 1 500 250 0)(111 0 1 500 250 0)(112 0 1 500 250 0)(113 0 1 500 250 0)(114 0 1 500 250 0)(115 0 1 500 250 0)(116 0 1 500 250 0)(117 0 1 500 250 0)(118 0 1 500 250 0)(119 0 1 500 250 0)(120 0 1 500 250 0)(121 0 1 500 250 0)(122 0 1 500 250 0)(123 0 1 500 250 0)(124 0 1 500 250 0)(125 0 1 500 250 0)(126 0 1 500 250 0)(127 0 1 500 250 0)(128 0 1 500 250 0)(129 0 1 500 250 0)(130 0 1 500 250 0)(131 0 1 500 250 0)(132 0 1 500 250 0)(133 0 1 500 250 0)(134 0 1 500 250 0)(135 0 1 500 250 0)(136 0 1 500 250 0)(137 0 1 500 250 0)(138 0 1 500 250 0)(139 0 1 500 250 0)(140 0 1 500 250 0)(141 0 1 500 250 0)(142 0 1 500 250 0)(143 0 1 500 250 0)(144 0 1 500 250 0)(145 0 1 500 250 0)(146 0 1 500 250 0)(147 0 1 500 250 0)(148 0 1 500 250 0)(149 0 1 500 250 0)(150 0 1 500 250 0)(151 0 1 500 250 0)(152 0 1 500 250 0)(153 0 1 500 250 0)(154 0 1 500 250 0)(155 0 1 500 250 0)(156 0 1 500 250 0)(157 0 1 500 250 0)(158 0 1 500 250 0)(159 0 1 500 250 0)(160 0 1 500 250 0)(161 0 1 500 250 0)(162 0 1 500 250 0)(163 0 1 500 250 0)(164 0 1 500 250 0)(165 0 1 500 250 0)(166 0 1 500 250 0)(167 0 1 500 250 0)(168 0 1 500 250 0)(169 0 1 500 250 0)(170 0 1 500 250 0)(171 0 1 500 250 0)(172 0 1 500 250 0)(173 0 1 500 250 0)(174 0 1 500 250 0)(175 0 1 500 250 0)(176 0 1 500 250 0)(177 0 1 500 250 0)(178 0 1 500 250 0)(179 0 1 500 250 0)(180 0 1 500 250 0)(181 0 1 500 250 0)(182 0 1 500 250 0)(183 0 1 500 250 0)(184 0 1 500 250 0)(185 0 1 500 250 0)(186 0 1 500 250 0)(187 0 1 500 250 0)(188 0 1 500 250 0)(189 0 1 500 250 0)(190 0 1 500 250 0)(191 0 1 500 250 0)(192 0 1 500 250 0)(193 0 1 500 250 0)(194 0 1 500 250 0)(195 0 1 500 250 0)(196 0 1 500 250 0)(197 0 1 500 250 0)(198 0 1 500 250 0)(199 0 1 500 250 0)(200 0 1 500 250 0)(201 0 1 500 250 0)(202 0 1 500 250 0)(203 0 1 500 250 0)(204 0 1 500 250 0)(205 0 1 500 250 0)(206 0 1 500 250 0)(207 0 1 500 250 0)(208 0 1 500 250 0)(209 0 1 500 250 0)(210 0 1 500 250 0)(211 0 1 500 250 0)(212 0 1 500 250 0)(213 0 1 500 250 0)(214 0 1 500 250 0)(215 0 1 500 250 0)(216 0 1 500 250 0)(217 0 1 500 250 0)(218 0 1 500 250 0)(219 0 1 500 250 0)(220 0 1 500 250 0)(221 0 1 500 250 0)(222 0 1 500 250 0)(223 0 1 500 250 0)(224 0 1 500 250 0)(225 0 1 500 250 0)(226 0 1 500 250 0)(227 0 1 500 250 0)(228 0 1 500 250 0)(229 0 1 500 250 0)(230 0 1 500 250 0)(231 0 1 500 250 0)(232 0 1 500 250 0)(233 0 1 500 250 0)(234 0 1 500 250 0)(235 0 1 500 250 0)(236 0 1 500 250 0)(237 0 1 500 250 0)(238 0 1 500 250 0)(239 0 1 500 250 0)(240 0 1 500 250 0)(241 0 1 500 250 0)(242 0 1 500 250 0)(243 0 1 500 250 0)(244 0 1 500 250 0)(245 0 1 500 250 0)(246 0 1 500 250 0)(247 0 1 500 250 0)(248 0 1 500 250 0)(249 0 1 500 250 0)(250 0 1 500 250 0)(251 0 1 500 250 0)(252 0 1 500 250 0)(253 0 1 500 250 0)(254 0 1 500 250 0)(255 0 1 500 250 0)(256 0 1 500 250 0)(257 0 1 500 250 0)(258 0 1 500 250 0)(259 0 1 500 250 0)(260 0 1 500 250 0)(261 0 1 500 250 0)(262 0 1 500 250 0)(263 0 1 500 250 0)(264 0 1 500 250 0)(265 0 1 500 250 0)(266 0 1 500 250 0)(267 0 1 500 250 0)(268 0 1 500 250 0)(269 0 1 500 250 0)(270 0 1 500 250 0)(271 0 1 500 250 0)(272 0 1 500 250 0)(273 0 1 500 250 0)(274 0 1 500 250 0)(275 0 1 500 250 0)(276 0 1 500 250 0)(277 0 1 500 250 0)(278 0 1 500 250 0)(279 0 1 500 250 0)(280 0 1 500 250 0)(281 0 1 500 250 0)(282 0 1 500 250 0)(283 0 1 500 250 0)(284 0 1 500 250 0)(285 0 1 500 250 0)(286 0 1 500 250 0)(287 0 1 500 250 0)(288 0 1 500 250 0)(289 0 1 500 250 0)(290 0 1 500 250 0)(291 0 1 500 250 0)(292 0 1 500 250 0)(293 0 1 500 250 0)(294 0 1 500 250 0)(295 0 1 500 250 0)(296 0 1 500 250 0)(297 0 1 500 250 0)(298 0 1 500 250 0)(299 0 1 500 250 0)(300 0 1 500 250 0)(301 0 1 500 250 0)(302 0 1 500 250 0)(303 0 1 500 250 0)(304 0 1 500 250 0)(305 0 1 500 250 0)(306 0 1 500 250 0)(307 0 1 500 250 0)(308 0 1 500 250 0)(309 0 1 500 250 0)(310 0 1 500 250 0)(311 0 1 500 250 0)(312 0 1 500 250 0)(313 0 1 500 250 0)(314 0 1 500 250 0)(315 0 1 500 250 0)(316 0 1 500 250 0)(317 0 1 500 250 0)(318 0 1 500 250 0)(319 0 1 500 250 0)(320 0 1 500 250 0)(321 0 1 500 250 0)(322 0 1 500 250 0)(323 0 1 500 250 0)(324 0 1 500 250 0)(325 0 1 500 250 0)(326 0 1 500 250 0)(327 0 1 500 250 0)(328 0 1 500 250 0)(329 0 1 500 250 0)(330 0 1 500 250 0)(331 0 1 500 250 0)(332 0 1 500 250 0)(333 0 1 500 250 0)(334 0 1 500 250 0)(335 0 1 500 250 0)(336 0 1 500 250 0)(337 0 1 500 250 0)(338 0 1 500 250 0)(339 0 1 500 250 0)(340 0 1 500 250 0)(341 0 1 500 250 0)(342 0 1 500 250 0)(343 0 1 500 250 0)(344 0 1 500 250 0)(345 0 1 500 250 0)(346 0 1 500 250 0)(347 0 1 500 250 0)(348 0 1 500 250 0)(349 0 1 500 250 0)(350 0 1 500 250 0)(351 0 1 500 250 0)(352 0 1 500 250 0)(353 0 1 500 250 0)(354 0 1 500 250 0)(355 0 1 500 250 0)(356 0 1 500 250 0)(357 0 1 500 250 0)(358 0 1 500 250 0)(359 0 1 500 250 0)(360 0 1 500 250 0)(361 0 1 500 250 0)(362 0 1 500 250 0)(363 0 1 500 250 0)(364 0 1 500 250 0)(365 0 1 500 250 0)(366 0 1 500 250 0)(367 0 1 500 250 0)(368 0 1 500 250 0)(369 0 1 500 250 0)(370 0 1 500 250 0)(371 0 1 500 250 0)(372 0 1 500 250 0)(373 0 1 500 250 0)(374 0 1 500 250 0)(375 0 1 500 250 0)(376 0 1 500 250 0)(377 0 1 500 250 0)(378 0 1 500 250 0)(379 0 1 500 250 0)(380 0 1 500 250 0)(381 0 1 500 250 0)(382 0 1 500 250 0)(383 0 1 500 250 0)'
        # meta['snsChanMap'] = '(384,384,1)(AP0;0:0)(AP1;1:192)(AP2;2:1)(AP3;3:193)(AP4;4:2)(AP5;5:194)(AP6;6:3)(AP7;7:195)(AP8;8:4)(AP9;9:196)(AP10;10:5)(AP11;11:197)(AP12;12:6)(AP13;13:198)(AP14;14:7)(AP15;15:199)(AP16;16:8)(AP17;17:200)(AP18;18:9)(AP19;19:201)(AP20;20:10)(AP21;21:202)(AP22;22:11)(AP23;23:203)(AP24;24:12)(AP25;25:204)(AP26;26:13)(AP27;27:205)(AP28;28:14)(AP29;29:206)(AP30;30:15)(AP31;31:207)(AP32;32:16)(AP33;33:208)(AP34;34:17)(AP35;35:209)(AP36;36:18)(AP37;37:210)(AP38;38:19)(AP39;39:211)(AP40;40:20)(AP41;41:212)(AP42;42:21)(AP43;43:213)(AP44;44:22)(AP45;45:214)(AP46;46:23)(AP47;47:215)(AP48;48:24)(AP49;49:216)(AP50;50:25)(AP51;51:217)(AP52;52:26)(AP53;53:218)(AP54;54:27)(AP55;55:219)(AP56;56:28)(AP57;57:220)(AP58;58:29)(AP59;59:221)(AP60;60:30)(AP61;61:222)(AP62;62:31)(AP63;63:223)(AP64;64:32)(AP65;65:224)(AP66;66:33)(AP67;67:225)(AP68;68:34)(AP69;69:226)(AP70;70:35)(AP71;71:227)(AP72;72:36)(AP73;73:228)(AP74;74:37)(AP75;75:229)(AP76;76:38)(AP77;77:230)(AP78;78:39)(AP79;79:231)(AP80;80:40)(AP81;81:232)(AP82;82:41)(AP83;83:233)(AP84;84:42)(AP85;85:234)(AP86;86:43)(AP87;87:235)(AP88;88:44)(AP89;89:236)(AP90;90:45)(AP91;91:237)(AP92;92:46)(AP93;93:238)(AP94;94:47)(AP95;95:239)(AP96;96:48)(AP97;97:240)(AP98;98:49)(AP99;99:241)(AP100;100:50)(AP101;101:242)(AP102;102:51)(AP103;103:243)(AP104;104:52)(AP105;105:244)(AP106;106:53)(AP107;107:245)(AP108;108:54)(AP109;109:246)(AP110;110:55)(AP111;111:247)(AP112;112:56)(AP113;113:248)(AP114;114:57)(AP115;115:249)(AP116;116:58)(AP117;117:250)(AP118;118:59)(AP119;119:251)(AP120;120:60)(AP121;121:252)(AP122;122:61)(AP123;123:253)(AP124;124:62)(AP125;125:254)(AP126;126:63)(AP127;127:255)(AP128;128:64)(AP129;129:256)(AP130;130:65)(AP131;131:257)(AP132;132:66)(AP133;133:258)(AP134;134:67)(AP135;135:259)(AP136;136:68)(AP137;137:260)(AP138;138:69)(AP139;139:261)(AP140;140:70)(AP141;141:262)(AP142;142:71)(AP143;143:263)(AP144;144:72)(AP145;145:264)(AP146;146:73)(AP147;147:265)(AP148;148:74)(AP149;149:266)(AP150;150:75)(AP151;151:267)(AP152;152:76)(AP153;153:268)(AP154;154:77)(AP155;155:269)(AP156;156:78)(AP157;157:270)(AP158;158:79)(AP159;159:271)(AP160;160:80)(AP161;161:272)(AP162;162:81)(AP163;163:273)(AP164;164:82)(AP165;165:274)(AP166;166:83)(AP167;167:275)(AP168;168:84)(AP169;169:276)(AP170;170:85)(AP171;171:277)(AP172;172:86)(AP173;173:278)(AP174;174:87)(AP175;175:279)(AP176;176:88)(AP177;177:280)(AP178;178:89)(AP179;179:281)(AP180;180:90)(AP181;181:282)(AP182;182:91)(AP183;183:283)(AP184;184:92)(AP185;185:284)(AP186;186:93)(AP187;187:285)(AP188;188:94)(AP189;189:286)(AP190;190:95)(AP191;191:287)(AP192;192:96)(AP193;193:288)(AP194;194:97)(AP195;195:289)(AP196;196:98)(AP197;197:290)(AP198;198:99)(AP199;199:291)(AP200;200:100)(AP201;201:292)(AP202;202:101)(AP203;203:293)(AP204;204:102)(AP205;205:294)(AP206;206:103)(AP207;207:295)(AP208;208:104)(AP209;209:296)(AP210;210:105)(AP211;211:297)(AP212;212:106)(AP213;213:298)(AP214;214:107)(AP215;215:299)(AP216;216:108)(AP217;217:300)(AP218;218:109)(AP219;219:301)(AP220;220:110)(AP221;221:302)(AP222;222:111)(AP223;223:303)(AP224;224:112)(AP225;225:304)(AP226;226:113)(AP227;227:305)(AP228;228:114)(AP229;229:306)(AP230;230:115)(AP231;231:307)(AP232;232:116)(AP233;233:308)(AP234;234:117)(AP235;235:309)(AP236;236:118)(AP237;237:310)(AP238;238:119)(AP239;239:311)(AP240;240:120)(AP241;241:312)(AP242;242:121)(AP243;243:313)(AP244;244:122)(AP245;245:314)(AP246;246:123)(AP247;247:315)(AP248;248:124)(AP249;249:316)(AP250;250:125)(AP251;251:317)(AP252;252:126)(AP253;253:318)(AP254;254:127)(AP255;255:319)(AP256;256:128)(AP257;257:320)(AP258;258:129)(AP259;259:321)(AP260;260:130)(AP261;261:322)(AP262;262:131)(AP263;263:323)(AP264;264:132)(AP265;265:324)(AP266;266:133)(AP267;267:325)(AP268;268:134)(AP269;269:326)(AP270;270:135)(AP271;271:327)(AP272;272:136)(AP273;273:328)(AP274;274:137)(AP275;275:329)(AP276;276:138)(AP277;277:330)(AP278;278:139)(AP279;279:331)(AP280;280:140)(AP281;281:332)(AP282;282:141)(AP283;283:333)(AP284;284:142)(AP285;285:334)(AP286;286:143)(AP287;287:335)(AP288;288:144)(AP289;289:336)(AP290;290:145)(AP291;291:337)(AP292;292:146)(AP293;293:338)(AP294;294:147)(AP295;295:339)(AP296;296:148)(AP297;297:340)(AP298;298:149)(AP299;299:341)(AP300;300:150)(AP301;301:342)(AP302;302:151)(AP303;303:343)(AP304;304:152)(AP305;305:344)(AP306;306:153)(AP307;307:345)(AP308;308:154)(AP309;309:346)(AP310;310:155)(AP311;311:347)(AP312;312:156)(AP313;313:348)(AP314;314:157)(AP315;315:349)(AP316;316:158)(AP317;317:350)(AP318;318:159)(AP319;319:351)(AP320;320:160)(AP321;321:352)(AP322;322:161)(AP323;323:353)(AP324;324:162)(AP325;325:354)(AP326;326:163)(AP327;327:355)(AP328;328:164)(AP329;329:356)(AP330;330:165)(AP331;331:357)(AP332;332:166)(AP333;333:358)(AP334;334:167)(AP335;335:359)(AP336;336:168)(AP337;337:360)(AP338;338:169)(AP339;339:361)(AP340;340:170)(AP341;341:362)(AP342;342:171)(AP343;343:363)(AP344;344:172)(AP345;345:364)(AP346;346:173)(AP347;347:365)(AP348;348:174)(AP349;349:366)(AP350;350:175)(AP351;351:367)(AP352;352:176)(AP353;353:368)(AP354;354:177)(AP355;355:369)(AP356;356:178)(AP357;357:370)(AP358;358:179)(AP359;359:371)(AP360;360:180)(AP361;361:372)(AP362;362:181)(AP363;363:373)(AP364;364:182)(AP365;365:374)(AP366;366:183)(AP367;367:375)(AP368;368:184)(AP369;369:376)(AP370;370:185)(AP371;371:377)(AP372;372:186)(AP373;373:378)(AP374;374:187)(AP375;375:379)(AP376;376:188)(AP377;377:380)(AP378;378:189)(AP379;379:381)(AP380;380:190)(AP381;381:382)(AP382;382:191)(AP383;383:383)(SY0;768:768)'
        # linear (West)
        meta['imroTbl'] = '(0,384)(0 0 1 1000 250 1)(1 0 1 1000 250 1)(2 0 1 1000 250 1)(3 0 1 1000 250 1)(4 0 1 1000 250 1)(5 0 1 1000 250 1)(6 0 1 1000 250 1)(7 0 1 1000 250 1)(8 0 1 1000 250 1)(9 0 1 1000 250 1)(10 0 1 1000 250 1)(11 0 1 1000 250 1)(12 0 1 1000 250 1)(13 0 1 1000 250 1)(14 0 1 1000 250 1)(15 0 1 1000 250 1)(16 0 1 1000 250 1)(17 0 1 1000 250 1)(18 0 1 1000 250 1)(19 0 1 1000 250 1)(20 0 1 1000 250 1)(21 0 1 1000 250 1)(22 0 1 1000 250 1)(23 0 1 1000 250 1)(24 0 1 1000 250 1)(25 0 1 1000 250 1)(26 0 1 1000 250 1)(27 0 1 1000 250 1)(28 0 1 1000 250 1)(29 0 1 1000 250 1)(30 0 1 1000 250 1)(31 0 1 1000 250 1)(32 0 1 1000 250 1)(33 0 1 1000 250 1)(34 0 1 1000 250 1)(35 0 1 1000 250 1)(36 0 1 1000 250 1)(37 0 1 1000 250 1)(38 0 1 1000 250 1)(39 0 1 1000 250 1)(40 0 1 1000 250 1)(41 0 1 1000 250 1)(42 0 1 1000 250 1)(43 0 1 1000 250 1)(44 0 1 1000 250 1)(45 0 1 1000 250 1)(46 0 1 1000 250 1)(47 0 1 1000 250 1)(48 0 1 1000 250 1)(49 0 1 1000 250 1)(50 0 1 1000 250 1)(51 0 1 1000 250 1)(52 0 1 1000 250 1)(53 0 1 1000 250 1)(54 0 1 1000 250 1)(55 0 1 1000 250 1)(56 0 1 1000 250 1)(57 0 1 1000 250 1)(58 0 1 1000 250 1)(59 0 1 1000 250 1)(60 0 1 1000 250 1)(61 0 1 1000 250 1)(62 0 1 1000 250 1)(63 0 1 1000 250 1)(64 0 1 1000 250 1)(65 0 1 1000 250 1)(66 0 1 1000 250 1)(67 0 1 1000 250 1)(68 0 1 1000 250 1)(69 0 1 1000 250 1)(70 0 1 1000 250 1)(71 0 1 1000 250 1)(72 0 1 1000 250 1)(73 0 1 1000 250 1)(74 0 1 1000 250 1)(75 0 1 1000 250 1)(76 0 1 1000 250 1)(77 0 1 1000 250 1)(78 0 1 1000 250 1)(79 0 1 1000 250 1)(80 0 1 1000 250 1)(81 0 1 1000 250 1)(82 0 1 1000 250 1)(83 0 1 1000 250 1)(84 0 1 1000 250 1)(85 0 1 1000 250 1)(86 0 1 1000 250 1)(87 0 1 1000 250 1)(88 0 1 1000 250 1)(89 0 1 1000 250 1)(90 0 1 1000 250 1)(91 0 1 1000 250 1)(92 0 1 1000 250 1)(93 0 1 1000 250 1)(94 0 1 1000 250 1)(95 0 1 1000 250 1)(96 0 1 1000 250 1)(97 0 1 1000 250 1)(98 0 1 1000 250 1)(99 0 1 1000 250 1)(100 0 1 1000 250 1)(101 0 1 1000 250 1)(102 0 1 1000 250 1)(103 0 1 1000 250 1)(104 0 1 1000 250 1)(105 0 1 1000 250 1)(106 0 1 1000 250 1)(107 0 1 1000 250 1)(108 0 1 1000 250 1)(109 0 1 1000 250 1)(110 0 1 1000 250 1)(111 0 1 1000 250 1)(112 0 1 1000 250 1)(113 0 1 1000 250 1)(114 0 1 1000 250 1)(115 0 1 1000 250 1)(116 0 1 1000 250 1)(117 0 1 1000 250 1)(118 0 1 1000 250 1)(119 0 1 1000 250 1)(120 0 1 1000 250 1)(121 0 1 1000 250 1)(122 0 1 1000 250 1)(123 0 1 1000 250 1)(124 0 1 1000 250 1)(125 0 1 1000 250 1)(126 0 1 1000 250 1)(127 0 1 1000 250 1)(128 0 1 1000 250 1)(129 0 1 1000 250 1)(130 0 1 1000 250 1)(131 0 1 1000 250 1)(132 0 1 1000 250 1)(133 0 1 1000 250 1)(134 0 1 1000 250 1)(135 0 1 1000 250 1)(136 0 1 1000 250 1)(137 0 1 1000 250 1)(138 0 1 1000 250 1)(139 0 1 1000 250 1)(140 0 1 1000 250 1)(141 0 1 1000 250 1)(142 0 1 1000 250 1)(143 0 1 1000 250 1)(144 0 1 1000 250 1)(145 0 1 1000 250 1)(146 0 1 1000 250 1)(147 0 1 1000 250 1)(148 0 1 1000 250 1)(149 0 1 1000 250 1)(150 0 1 1000 250 1)(151 0 1 1000 250 1)(152 0 1 1000 250 1)(153 0 1 1000 250 1)(154 0 1 1000 250 1)(155 0 1 1000 250 1)(156 0 1 1000 250 1)(157 0 1 1000 250 1)(158 0 1 1000 250 1)(159 0 1 1000 250 1)(160 0 1 1000 250 1)(161 0 1 1000 250 1)(162 0 1 1000 250 1)(163 0 1 1000 250 1)(164 0 1 1000 250 1)(165 0 1 1000 250 1)(166 0 1 1000 250 1)(167 0 1 1000 250 1)(168 0 1 1000 250 1)(169 0 1 1000 250 1)(170 0 1 1000 250 1)(171 0 1 1000 250 1)(172 0 1 1000 250 1)(173 0 1 1000 250 1)(174 0 1 1000 250 1)(175 0 1 1000 250 1)(176 0 1 1000 250 1)(177 0 1 1000 250 1)(178 0 1 1000 250 1)(179 0 1 1000 250 1)(180 0 1 1000 250 1)(181 0 1 1000 250 1)(182 0 1 1000 250 1)(183 0 1 1000 250 1)(184 0 1 1000 250 1)(185 0 1 1000 250 1)(186 0 1 1000 250 1)(187 0 1 1000 250 1)(188 0 1 1000 250 1)(189 0 1 1000 250 1)(190 0 1 1000 250 1)(191 0 1 1000 250 1)(192 0 1 1000 250 1)(193 0 1 1000 250 1)(194 0 1 1000 250 1)(195 0 1 1000 250 1)(196 0 1 1000 250 1)(197 0 1 1000 250 1)(198 0 1 1000 250 1)(199 0 1 1000 250 1)(200 0 1 1000 250 1)(201 0 1 1000 250 1)(202 0 1 1000 250 1)(203 0 1 1000 250 1)(204 0 1 1000 250 1)(205 0 1 1000 250 1)(206 0 1 1000 250 1)(207 0 1 1000 250 1)(208 0 1 1000 250 1)(209 0 1 1000 250 1)(210 0 1 1000 250 1)(211 0 1 1000 250 1)(212 0 1 1000 250 1)(213 0 1 1000 250 1)(214 0 1 1000 250 1)(215 0 1 1000 250 1)(216 0 1 1000 250 1)(217 0 1 1000 250 1)(218 0 1 1000 250 1)(219 0 1 1000 250 1)(220 0 1 1000 250 1)(221 0 1 1000 250 1)(222 0 1 1000 250 1)(223 0 1 1000 250 1)(224 0 1 1000 250 1)(225 0 1 1000 250 1)(226 0 1 1000 250 1)(227 0 1 1000 250 1)(228 0 1 1000 250 1)(229 0 1 1000 250 1)(230 0 1 1000 250 1)(231 0 1 1000 250 1)(232 0 1 1000 250 1)(233 0 1 1000 250 1)(234 0 1 1000 250 1)(235 0 1 1000 250 1)(236 0 1 1000 250 1)(237 0 1 1000 250 1)(238 0 1 1000 250 1)(239 0 1 1000 250 1)(240 0 1 1000 250 1)(241 0 1 1000 250 1)(242 0 1 1000 250 1)(243 0 1 1000 250 1)(244 0 1 1000 250 1)(245 0 1 1000 250 1)(246 0 1 1000 250 1)(247 0 1 1000 250 1)(248 0 1 1000 250 1)(249 0 1 1000 250 1)(250 0 1 1000 250 1)(251 0 1 1000 250 1)(252 0 1 1000 250 1)(253 0 1 1000 250 1)(254 0 1 1000 250 1)(255 0 1 1000 250 1)(256 0 1 1000 250 1)(257 0 1 1000 250 1)(258 0 1 1000 250 1)(259 0 1 1000 250 1)(260 0 1 1000 250 1)(261 0 1 1000 250 1)(262 0 1 1000 250 1)(263 0 1 1000 250 1)(264 0 1 1000 250 1)(265 0 1 1000 250 1)(266 0 1 1000 250 1)(267 0 1 1000 250 1)(268 0 1 1000 250 1)(269 0 1 1000 250 1)(270 0 1 1000 250 1)(271 0 1 1000 250 1)(272 0 1 1000 250 1)(273 0 1 1000 250 1)(274 0 1 1000 250 1)(275 0 1 1000 250 1)(276 0 1 1000 250 1)(277 0 1 1000 250 1)(278 0 1 1000 250 1)(279 0 1 1000 250 1)(280 0 1 1000 250 1)(281 0 1 1000 250 1)(282 0 1 1000 250 1)(283 0 1 1000 250 1)(284 0 1 1000 250 1)(285 0 1 1000 250 1)(286 0 1 1000 250 1)(287 0 1 1000 250 1)(288 0 1 1000 250 1)(289 0 1 1000 250 1)(290 0 1 1000 250 1)(291 0 1 1000 250 1)(292 0 1 1000 250 1)(293 0 1 1000 250 1)(294 0 1 1000 250 1)(295 0 1 1000 250 1)(296 0 1 1000 250 1)(297 0 1 1000 250 1)(298 0 1 1000 250 1)(299 0 1 1000 250 1)(300 0 1 1000 250 1)(301 0 1 1000 250 1)(302 0 1 1000 250 1)(303 0 1 1000 250 1)(304 0 1 1000 250 1)(305 0 1 1000 250 1)(306 0 1 1000 250 1)(307 0 1 1000 250 1)(308 0 1 1000 250 1)(309 0 1 1000 250 1)(310 0 1 1000 250 1)(311 0 1 1000 250 1)(312 0 1 1000 250 1)(313 0 1 1000 250 1)(314 0 1 1000 250 1)(315 0 1 1000 250 1)(316 0 1 1000 250 1)(317 0 1 1000 250 1)(318 0 1 1000 250 1)(319 0 1 1000 250 1)(320 0 1 1000 250 1)(321 0 1 1000 250 1)(322 0 1 1000 250 1)(323 0 1 1000 250 1)(324 0 1 1000 250 1)(325 0 1 1000 250 1)(326 0 1 1000 250 1)(327 0 1 1000 250 1)(328 0 1 1000 250 1)(329 0 1 1000 250 1)(330 0 1 1000 250 1)(331 0 1 1000 250 1)(332 0 1 1000 250 1)(333 0 1 1000 250 1)(334 0 1 1000 250 1)(335 0 1 1000 250 1)(336 0 1 1000 250 1)(337 0 1 1000 250 1)(338 0 1 1000 250 1)(339 0 1 1000 250 1)(340 0 1 1000 250 1)(341 0 1 1000 250 1)(342 0 1 1000 250 1)(343 0 1 1000 250 1)(344 0 1 1000 250 1)(345 0 1 1000 250 1)(346 0 1 1000 250 1)(347 0 1 1000 250 1)(348 0 1 1000 250 1)(349 0 1 1000 250 1)(350 0 1 1000 250 1)(351 0 1 1000 250 1)(352 0 1 1000 250 1)(353 0 1 1000 250 1)(354 0 1 1000 250 1)(355 0 1 1000 250 1)(356 0 1 1000 250 1)(357 0 1 1000 250 1)(358 0 1 1000 250 1)(359 0 1 1000 250 1)(360 0 1 1000 250 1)(361 0 1 1000 250 1)(362 0 1 1000 250 1)(363 0 1 1000 250 1)(364 0 1 1000 250 1)(365 0 1 1000 250 1)(366 0 1 1000 250 1)(367 0 1 1000 250 1)(368 0 1 1000 250 1)(369 0 1 1000 250 1)(370 0 1 1000 250 1)(371 0 1 1000 250 1)(372 0 1 1000 250 1)(373 0 1 1000 250 1)(374 0 1 1000 250 1)(375 0 1 1000 250 1)(376 0 1 1000 250 1)(377 0 1 1000 250 1)(378 0 1 1000 250 1)(379 0 1 1000 250 1)(380 0 1 1000 250 1)(381 0 1 1000 250 1)(382 0 1 1000 250 1)(383 0 1 1000 250 1)'
        meta['snsChanMap'] = '(384,384,1)(AP0;0:0)(AP1;1:1)(AP2;2:2)(AP3;3:3)(AP4;4:4)(AP5;5:5)(AP6;6:6)(AP7;7:7)(AP8;8:8)(AP9;9:9)(AP10;10:10)(AP11;11:11)(AP12;12:12)(AP13;13:13)(AP14;14:14)(AP15;15:15)(AP16;16:16)(AP17;17:17)(AP18;18:18)(AP19;19:19)(AP20;20:20)(AP21;21:21)(AP22;22:22)(AP23;23:23)(AP24;24:24)(AP25;25:25)(AP26;26:26)(AP27;27:27)(AP28;28:28)(AP29;29:29)(AP30;30:30)(AP31;31:31)(AP32;32:32)(AP33;33:33)(AP34;34:34)(AP35;35:35)(AP36;36:36)(AP37;37:37)(AP38;38:38)(AP39;39:39)(AP40;40:40)(AP41;41:41)(AP42;42:42)(AP43;43:43)(AP44;44:44)(AP45;45:45)(AP46;46:46)(AP47;47:47)(AP48;48:48)(AP49;49:49)(AP50;50:50)(AP51;51:51)(AP52;52:52)(AP53;53:53)(AP54;54:54)(AP55;55:55)(AP56;56:56)(AP57;57:57)(AP58;58:58)(AP59;59:59)(AP60;60:60)(AP61;61:61)(AP62;62:62)(AP63;63:63)(AP64;64:64)(AP65;65:65)(AP66;66:66)(AP67;67:67)(AP68;68:68)(AP69;69:69)(AP70;70:70)(AP71;71:71)(AP72;72:72)(AP73;73:73)(AP74;74:74)(AP75;75:75)(AP76;76:76)(AP77;77:77)(AP78;78:78)(AP79;79:79)(AP80;80:80)(AP81;81:81)(AP82;82:82)(AP83;83:83)(AP84;84:84)(AP85;85:85)(AP86;86:86)(AP87;87:87)(AP88;88:88)(AP89;89:89)(AP90;90:90)(AP91;91:91)(AP92;92:92)(AP93;93:93)(AP94;94:94)(AP95;95:95)(AP96;96:96)(AP97;97:97)(AP98;98:98)(AP99;99:99)(AP100;100:100)(AP101;101:101)(AP102;102:102)(AP103;103:103)(AP104;104:104)(AP105;105:105)(AP106;106:106)(AP107;107:107)(AP108;108:108)(AP109;109:109)(AP110;110:110)(AP111;111:111)(AP112;112:112)(AP113;113:113)(AP114;114:114)(AP115;115:115)(AP116;116:116)(AP117;117:117)(AP118;118:118)(AP119;119:119)(AP120;120:120)(AP121;121:121)(AP122;122:122)(AP123;123:123)(AP124;124:124)(AP125;125:125)(AP126;126:126)(AP127;127:127)(AP128;128:128)(AP129;129:129)(AP130;130:130)(AP131;131:131)(AP132;132:132)(AP133;133:133)(AP134;134:134)(AP135;135:135)(AP136;136:136)(AP137;137:137)(AP138;138:138)(AP139;139:139)(AP140;140:140)(AP141;141:141)(AP142;142:142)(AP143;143:143)(AP144;144:144)(AP145;145:145)(AP146;146:146)(AP147;147:147)(AP148;148:148)(AP149;149:149)(AP150;150:150)(AP151;151:151)(AP152;152:152)(AP153;153:153)(AP154;154:154)(AP155;155:155)(AP156;156:156)(AP157;157:157)(AP158;158:158)(AP159;159:159)(AP160;160:160)(AP161;161:161)(AP162;162:162)(AP163;163:163)(AP164;164:164)(AP165;165:165)(AP166;166:166)(AP167;167:167)(AP168;168:168)(AP169;169:169)(AP170;170:170)(AP171;171:171)(AP172;172:172)(AP173;173:173)(AP174;174:174)(AP175;175:175)(AP176;176:176)(AP177;177:177)(AP178;178:178)(AP179;179:179)(AP180;180:180)(AP181;181:181)(AP182;182:182)(AP183;183:183)(AP184;184:184)(AP185;185:185)(AP186;186:186)(AP187;187:187)(AP188;188:188)(AP189;189:189)(AP190;190:190)(AP191;191:191)(AP192;192:192)(AP193;193:193)(AP194;194:194)(AP195;195:195)(AP196;196:196)(AP197;197:197)(AP198;198:198)(AP199;199:199)(AP200;200:200)(AP201;201:201)(AP202;202:202)(AP203;203:203)(AP204;204:204)(AP205;205:205)(AP206;206:206)(AP207;207:207)(AP208;208:208)(AP209;209:209)(AP210;210:210)(AP211;211:211)(AP212;212:212)(AP213;213:213)(AP214;214:214)(AP215;215:215)(AP216;216:216)(AP217;217:217)(AP218;218:218)(AP219;219:219)(AP220;220:220)(AP221;221:221)(AP222;222:222)(AP223;223:223)(AP224;224:224)(AP225;225:225)(AP226;226:226)(AP227;227:227)(AP228;228:228)(AP229;229:229)(AP230;230:230)(AP231;231:231)(AP232;232:232)(AP233;233:233)(AP234;234:234)(AP235;235:235)(AP236;236:236)(AP237;237:237)(AP238;238:238)(AP239;239:239)(AP240;240:240)(AP241;241:241)(AP242;242:242)(AP243;243:243)(AP244;244:244)(AP245;245:245)(AP246;246:246)(AP247;247:247)(AP248;248:248)(AP249;249:249)(AP250;250:250)(AP251;251:251)(AP252;252:252)(AP253;253:253)(AP254;254:254)(AP255;255:255)(AP256;256:256)(AP257;257:257)(AP258;258:258)(AP259;259:259)(AP260;260:260)(AP261;261:261)(AP262;262:262)(AP263;263:263)(AP264;264:264)(AP265;265:265)(AP266;266:266)(AP267;267:267)(AP268;268:268)(AP269;269:269)(AP270;270:270)(AP271;271:271)(AP272;272:272)(AP273;273:273)(AP274;274:274)(AP275;275:275)(AP276;276:276)(AP277;277:277)(AP278;278:278)(AP279;279:279)(AP280;280:280)(AP281;281:281)(AP282;282:282)(AP283;283:283)(AP284;284:284)(AP285;285:285)(AP286;286:286)(AP287;287:287)(AP288;288:288)(AP289;289:289)(AP290;290:290)(AP291;291:291)(AP292;292:292)(AP293;293:293)(AP294;294:294)(AP295;295:295)(AP296;296:296)(AP297;297:297)(AP298;298:298)(AP299;299:299)(AP300;300:300)(AP301;301:301)(AP302;302:302)(AP303;303:303)(AP304;304:304)(AP305;305:305)(AP306;306:306)(AP307;307:307)(AP308;308:308)(AP309;309:309)(AP310;310:310)(AP311;311:311)(AP312;312:312)(AP313;313:313)(AP314;314:314)(AP315;315:315)(AP316;316:316)(AP317;317:317)(AP318;318:318)(AP319;319:319)(AP320;320:320)(AP321;321:321)(AP322;322:322)(AP323;323:323)(AP324;324:324)(AP325;325:325)(AP326;326:326)(AP327;327:327)(AP328;328:328)(AP329;329:329)(AP330;330:330)(AP331;331:331)(AP332;332:332)(AP333;333:333)(AP334;334:334)(AP335;335:335)(AP336;336:336)(AP337;337:337)(AP338;338:338)(AP339;339:339)(AP340;340:340)(AP341;341:341)(AP342;342:342)(AP343;343:343)(AP344;344:344)(AP345;345:345)(AP346;346:346)(AP347;347:347)(AP348;348:348)(AP349;349:349)(AP350;350:350)(AP351;351:351)(AP352;352:352)(AP353;353:353)(AP354;354:354)(AP355;355:355)(AP356;356:356)(AP357;357:357)(AP358;358:358)(AP359;359:359)(AP360;360:360)(AP361;361:361)(AP362;362:362)(AP363;363:363)(AP364;364:364)(AP365;365:365)(AP366;366:366)(AP367;367:367)(AP368;368:368)(AP369;369:369)(AP370;370:370)(AP371;371:371)(AP372;372:372)(AP373;373:373)(AP374;374:374)(AP375;375:375)(AP376;376:376)(AP377;377:377)(AP378;378:378)(AP379;379:379)(AP380;380:380)(AP381;381:381)(AP382;382:382)(AP383;383:383)(SY0;768:768)'
        meta['imAiRangeMin'] = '-0.6'
        meta['imAiRangeMax'] = '0.6'
        meta['imMaxInt'] = '512'
        
            
    chanmap = read_snsChanMap(meta)
    imroTbl = read_imroTbl(meta)

    if len(np.unique(imroTbl[:,1])) == 1 and chanmap[1,1] == 192:
        print('checkerboard')
        chanmap[:,1] = chanmap[:,0]
    elif len(np.unique(imroTbl[:,1])) > 1 and chanmap[1,1] == 192:
        print('long')
    elif (chanmap[:,0] == chanmap[:,1]).all():
        print('linear')     

    return chanmap, imroTbl
    