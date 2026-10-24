#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORTS
from data_analysis_tools_mkTurk.utils_meta import *
from data_analysis_tools_mkTurk.utils_mkturk import *
import pathlib as Path
# sys.path
# sys.path.append(r"c:\users\hiyun\anaconda3\lib\site-packages")
import numpy as np 
import os
import json
import pickle
from natsort import os_sorted


# In[ ]:


# SET PATHS, RECORDING INFO

import socket 
host = socket.gethostname()
if 'rc.zi.columbia.edu' in host: 
    engram_path = Path.Path('/mnt/smb/locker/issa-locker')
elif 'DESKTOP' in host:
    engram_path = Path.Path('Z:/')
elif 'Younah' in host:
    engram_path = Path.Path('/Volumes/issa-locker/')

base_data_path = engram_path /'Data'
base_save_out_path = engram_path / 'users'/ 'Dan' / 'ephys'
monkey = 'West'
date = '20241011'

data_path = get_recording_path(base_data_path, monkey, date)

n_recordings = len(data_path)
print(n_recordings, ' recordings found')
if n_recordings > 1:
    print('foo')
    data_ind = input('multiple recording sessions. Please select a number between 0 and' + str(len(data_path)-1))  # waiting for user input
    data_path = Path.Path(data_path[int(data_ind)])
else:
    data_path = Path.Path(data_path[0])
penetration = data_path.relative_to(base_data_path/monkey).as_posix().split('/')[0]
print(data_path)

print('\nData path found: '+ str(data_path.exists()))

# meta_iter = data_path.glob('*ap.meta')
# bin_iter = data_path.glob('*ap.bin')   
# meta_path = next(meta_iter)
# bin_path = next(bin_iter) 


# In[ ]:





# In[ ]:


# get trig 
trig_path = data_path / 'imec_trig'
print(trig_path)
trig_on, trig_off = np.load(trig_path / 'trig_ind.npy')
print(len(trig_on))


# In[ ]:


# behavior files in the data path 

behav_file_list_orig = os_sorted(Path.Path(base_data_path, monkey, penetration).glob('*.json'))
print('Number of behavior files in the data path: ', len(behav_file_list_orig)) 
for i, b_f in enumerate(behav_file_list_orig):
    m = json.load(open(b_f, 'rb'))
    n_trials_mk_prepared = len(m['TRIALEVENTS']['Sample']['0'])
    if len(m['TRIALEVENTS']['TSequenceActualClip']) >  0:
        n_trials_mk_shown = len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
    else:
        n_trials_mk_shown = 0 
    behav_file_list_orig[i] = b_f.as_posix()
    print(b_f.stem, n_trials_mk_prepared, n_trials_mk_shown)


# In[ ]:


if len(trig_on) > 1 and len(behav_file_list_orig) > 0:
    save_out_root_path = base_save_out_path/ monkey
    save_out_path = Path.Path(save_out_root_path, penetration)

    if not os.path.exists(save_out_path):
        os.mkdir(save_out_path)
        print('creating save out folder')
    else:
        print('path exists')

    if n_recordings> 1:
        save_out_path = Path.Path(save_out_root_path, penetration, data_path.parts[len(data_path.parts)-2])

    print(save_out_path)
    os.makedirs(save_out_path, exist_ok=True)
else:
    print('Skip this recording session as there''s no trigger saved or behavior files not in the path')


# In[ ]:


def get_filecode(filecode_ind, len_filecode,out_list):
    if len(filecode_ind) > len_filecode and len(filecode_ind) % len_filecode == 0:
        # chunk by 6 
        for i in range(0, len(filecode_ind), len_filecode):  
            out_list.append(filecode_ind[i:i+len_filecode])
    elif len(filecode_ind) == 6:
        out_list.append(filecode_ind)
    elif len(filecode_ind) > len_filecode: # if the detected 
        out_list.append(filecode_ind[len(filecode_ind)-len_filecode:len(filecode_ind)])
    return out_list


# In[ ]:


from itertools import groupby
from operator import itemgetter

Fs = 30000
len_filecode = 6
max_trig_dur =300# digit (9 +1) * 10ms # change this value for earlier files. After 10/04/2023, 150 seems fine
trig_dur = (trig_off - trig_on) /Fs * 1000 # ms
ind = np.where(trig_dur <=max_trig_dur)[0] 

filecodes_ind_imec_possible= []
for k, g in groupby(enumerate(ind), lambda ix : ix[0] -ix[1]):
    filecode_ind = list(map(itemgetter(1), g))
    print(filecode_ind, len(filecode_ind))
    filecodes_ind_imec_possible = get_filecode(filecode_ind,len_filecode, filecodes_ind_imec_possible)
print( f'{len(filecodes_ind_imec_possible)}' + ' possible filecodes found')


# In[ ]:


filecodes_imec = []
scs_ind_imec = []
n_scs_imec = []
filecodes_ind_imec = []

for i, filecode_ind in enumerate(filecodes_ind_imec_possible):
    assert len(filecode_ind) == len_filecode, f'filecode length is not {len_filecode}'

    filecode_dur = trig_dur[filecode_ind]
    # sc_ind corresponds to indices of sample command triggers followed by a filecode 
    if i == len(filecodes_ind_imec_possible) -1: 
        sc_ind = np.arange(filecode_ind[len_filecode-1]+1, len(trig_dur)) # sample commands followed by the last filecode within a session
    else:
        sc_ind = np.arange(filecode_ind[len_filecode-1]+1, filecodes_ind_imec_possible[i+1][0])

    # number of sample commands or number of initated trials
    n_scs = len(sc_ind)

    # converting the digital filecode to timestamps which should match the name of the behavior file 
    f_convert =[round(f/10-1) for f in filecode_dur]
    f_convert = [0 if x<0 else x for x in f_convert]

    filecode = str(f_convert[0]) + str(f_convert[1]) + '_' + \
    str(f_convert[2]) + str(f_convert[3]) + '_' + \
    str(f_convert[4]) + str(f_convert[5]) 

    print(i, 'start ind: ', filecode_ind[0],'filecode: ', filecode, '# of sample commands: ', n_scs)
    filecodes_imec.append(filecode)
    scs_ind_imec.append(sc_ind)
    n_scs_imec.append(n_scs)
    filecodes_ind_imec.append(filecode_ind)

filecodes_ind_imec = np.array(filecodes_ind_imec)
scs_ind_imec = np.array(scs_ind_imec, dtype=object)
n_scs_imec = np.array(n_scs_imec)
filecodes_imec = np.array(filecodes_imec)


# In[ ]:


# there are some cases where the first sample command is fired before the filecode. 
# In this case, there is a very short latency between the onest of the filecode and the offset of the first sample command 
# it's likely that this first sample command was grouped to the previous file as the last sample command 

for idx,(filecode, filecode_ind, scs_ind) in enumerate(zip(filecodes_imec, filecodes_ind_imec,scs_ind_imec)):

    if idx > 0 and n_scs_imec[idx-1] > 0:
        t_diff = (trig_on[filecode_ind[0]] - trig_off[scs_ind_imec[idx-1][len(scs_ind_imec[idx-1])-1]]) /Fs * 1000
        print(idx, t_diff)
        if t_diff < 200: # less than 100ms or some small number
            print(filecode)
            prev_list= scs_ind_imec[idx-1]
            scs_ind = np.insert(scs_ind, 0,prev_list[len(prev_list)-1])
            scs_ind_imec[idx] = scs_ind
            n_scs_imec[idx] = len(scs_ind)
            scs_ind_imec[idx-1] = prev_list[0:len(prev_list)-1]
            n_scs_imec[idx-1] = n_scs_imec[idx-1] -1


# In[ ]:


for i,(filecode, filecode_ind, scs_ind, n_scs) in enumerate(zip(filecodes_imec, filecodes_ind_imec, scs_ind_imec, n_scs_imec)):
    print(i, 'filecode: ', filecode, ' # of sample commands: ', n_scs, len(scs_ind))


# In[ ]:


# Matching behavior file to imec filecode 

behav_file_list = []
behav_file_list_idx = [-1]
filecodes_imec_to_analyze = []
scs_ind_imec_to_analyze = []
n_scs_imec_to_analyze = []
filecodes_ind_imec_to_analyze = []
for i, (f, n_scs, scs_ind, filecodes_ind) in enumerate(zip(filecodes_imec, n_scs_imec,scs_ind_imec,filecodes_ind_imec)):
    print('\n')
    print(i, f)
    print('# of scs: ', n_scs)
    
    behav_file = os_sorted(Path.Path(base_data_path, monkey, penetration).glob('*' + f + '*.json'))
    if len(behav_file) > 0:
        m = json.load(open(behav_file[0].as_posix(),'rb'))
        if len(m['TRIALEVENTS']['TSequenceActualClip']) >  0:
            n_trials_mk = len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
        else:
            n_trials_mk = 0 

        print('matching behavior file found')
        print('matched with' , behav_file[0].as_posix())
        print('# of mkturk trials: ', n_trials_mk)
        print('RewardStage : ', m['TASK']['RewardStage'])

        behav_file_list.append(behav_file[0].as_posix())
        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == behav_file[0].as_posix())[0])
        filecodes_imec_to_analyze.append(f)
        scs_ind_imec_to_analyze.append(scs_ind)
        n_scs_imec_to_analyze.append(n_scs)
        filecodes_ind_imec_to_analyze.append(filecodes_ind)
    else:
        print(f'No corresponding behavior file that matches with filecode {f} from imec' )

        # if there exists a mkturk file that matches almost all of the filecodes
        for b_f_idx, b_f in enumerate(behav_file_list_orig):
            if b_f not in behav_file_list and b_f_idx > max(behav_file_list_idx):
                m = json.load(open(b_f,'rb'))
                if len(m['TRIALEVENTS']['TSequenceActualClip']) >  0:
                    n_trials_mk = len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
                else:
                    n_trials_mk = 0 

                file_time = Path.Path(b_f).stem.split('T')[1].split('_' + monkey)[0]
                file_time_txt = file_time.split('_')
                file_time_hour = file_time_txt[0]
                file_time_minute = file_time_txt[1]
                file_time_second = file_time_txt[2]

                f_txt = f.split('_')
                f_hour = f_txt[0]
                f_minute = f_txt[1]
                f_second = f_txt[2]

                hour_diff = abs(int(file_time_hour) - int(f_hour))
                minute_diff = abs(int(file_time_minute) - int(f_minute))
                second_diff = abs(int(file_time_second) - int(f_second))
                
                diff_all = np.array((hour_diff,minute_diff,second_diff))

                str_match =0

                for t_str_count in range(8):
                    if file_time[t_str_count] == f[t_str_count]:
                        str_match += 1

                if sum(diff_all) <=3 or len(np.where(diff_all == 0)[0]) == 2 or str_match >=6:
                #if str_match >=6: # you can relax this threshold by making this number smaller
                    print('filecode is defective but matches most of the datetime string in the behavior file')
                    print('matched with' , b_f)
                    print('# of mkturk trials: ', n_trials_mk)
                    print('RewardStage : ', m['TASK']['RewardStage'])
                    if n_scs == 0 and m['TASK']['RewardStage'] == 0:
                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_imec_to_analyze.append(f)
                        scs_ind_imec_to_analyze.append(scs_ind)
                        n_scs_imec_to_analyze.append(n_scs)
                        filecodes_ind_imec_to_analyze.append(filecodes_ind)
                        break 
                    elif abs(n_scs - n_trials_mk) < 7:
                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_imec_to_analyze.append(f)
                        scs_ind_imec_to_analyze.append(scs_ind)
                        n_scs_imec_to_analyze.append(n_scs)
                        filecodes_ind_imec_to_analyze.append(filecodes_ind)
                        break 
                    else:
                        print('# of sc and # of mk trials don''t match up')
                        break
                else:

                    if n_trials_mk >= n_scs -2 and n_trials_mk <= n_scs + 2 and n_scs !=0: # this might happen for earlier files 

                        print('no string matched but the number of sample command triggers seem to match the number of mkturk trials')
                        print('# of mkturk trials: ', n_trials_mk)
                        print('matched with' , b_f)
                        print('RewardStage : ', m['TASK']['RewardStage'])

                        behav_file_list.append(b_f)
                        behav_file_list_idx.append(np.where(np.array(behav_file_list_orig) == b_f)[0])
                        filecodes_imec_to_analyze.append(f)
                        scs_ind_imec_to_analyze.append(scs_ind)
                        n_scs_imec_to_analyze.append(n_scs)
                        filecodes_ind_imec_to_analyze.append(filecodes_ind)

                        break


# In[ ]:


assert len(behav_file_list) == len(n_scs_imec_to_analyze)


# In[ ]:


print(len(behav_file_list_orig) - len(behav_file_list), ' behavior files are unmatched \n')
unmatched_behav_file = list(set(behav_file_list_orig) - set(behav_file_list))

for b_f in unmatched_behav_file:
    print(b_f)
    m = json.load(open(b_f,'rb'))
    print('RewardStage : ', m['TASK']['RewardStage'], '\n')
    if len(m['TRIALEVENTS']['TSequenceActualClip']) >  0:
        n_trials_mk = len(m['TRIALEVENTS']['TSequenceActualClip']['0'])
    else:
        n_trials_mk = 0 

    print('# of trials: ', n_trials_mk)


# In[ ]:


# You can either try to find triggers that belong to this remaining behavior file 
#  or skip it if it's a calibration file 

# of trigs unaccounted for 
print(len(trig_on) - (np.sum(n_scs_imec_to_analyze) + len_filecode * len(filecodes_imec_to_analyze)))


# In[ ]:


n_stims_sess = 0 
n_trials_sess = 0
data_dict_all = dict()
for idx, (behav_file, scs_ind, n_scs,filecode_ind) in enumerate(zip(behav_file_list,scs_ind_imec_to_analyze, n_scs_imec_to_analyze,filecodes_ind_imec_to_analyze)):
    print(behav_file)
    # skip calibration files
    m = json.load(open(behav_file,'rb'))

    if m['TASK']['RewardStage'] == 0:
        print('calibration file. removing it from further analysis')
    else:
        if len(m['TASK']['ImageBagsSample']) != len( m['SCENES']['SampleScenes']):
            print('ImageBagSample and number of scenefiles don''t match. skip this behavior file')
            continue

        data_dict  = create_data_mat(behav_file)
        print(m['SCENEMETA']['SampleImageSetDir'])

        sc_off_mk =np.array(m['TRIALEVENTS']['SampleCommandOffReturnTime'],dtype=float)
        sc_on_mk = np.array(m['TRIALEVENTS']['SampleCommandReturnTime'],dtype=float)

        if 'NRSVPMax' in m['TASK'].keys():
            n_rsvp = max(m['TASK']['NRSVP'], m['TASK']['NRSVPMax'])
        else:
            n_rsvp = m['TASK']['NRSVP'] 

        trig_on_time = trig_on[scs_ind]/Fs 
        trig_off_time = trig_off[scs_ind]/Fs 
        trig_dur = trig_off_time - trig_on_time

        if max(len(sc_on_mk),len(sc_off_mk)) <=10:
            print('two few trials. Removing it from further analysis')
            continue
        
        if len(sc_on_mk) != len(sc_off_mk):
            print('mkturk behavior file has mismatched sample command on and off ')
            if sc_on_mk[0] < sc_off_mk[0] and len(sc_on_mk) > len(sc_off_mk):
                sc_off_mk = np.concatenate((sc_off_mk,np.nan * np.ones(len(sc_on_mk) - len(sc_off_mk))))
                
        sc_dur_mk =sc_off_mk - sc_on_mk
        n_trials_mk = len(sc_dur_mk)

        print('# of trials from mkturk file :', n_trials_mk, '\n'
            '# of imec triggers : ', n_scs)
        
        if n_trials_mk <= 10 and n_scs <= 10:
            print('two few trials. Removing it from further analysis')
        else:   
            # compare number of triggers from imec file and n_trials_mk
            # if they mismatch, they usually mismatch by 1
            mean = np.mean(trig_dur)
            sd = np.std(trig_dur)
            print('mean trig dur: ', mean, 'sd: ', sd)
            
            if n_trials_mk == n_scs: 
                print('# of mkturk trials matches # of imec trigs')
                print(np.nansum(np.abs(sc_dur_mk/1000 - trig_dur)))  # this number should be small but sometimes mkturk SampleCommandOffReturnTime is weird
                #print(trig_dur, sc_dur_mk)
            
            if n_trials_mk > n_scs: # more mkturk trials than imec sample commands
                print('more mkturk trials than imec trigs')
                n_diff= n_trials_mk - n_scs
                t_diff_first = np.nansum(np.abs(sc_dur_mk[0:n_scs]/1000 - trig_dur))
                t_diff_last = np.nansum(np.abs(sc_dur_mk[n_diff:n_trials_mk]/1000-trig_dur))
                print(f'first {n_diff} mkturk trials : ', sc_dur_mk[0:n_diff], f' last {n_diff} mkturk trials: ', sc_dur_mk[n_scs:n_trials_mk])
                print(f'aligned to the first {n_scs} mkturk trials: ', t_diff_first, f' aligned to the last {n_scs} mkturk trials: ', t_diff_last)
                
                if t_diff_first > 5 and t_diff_last > 5:
                    print('difference is too big on both ends. Check trig_dur and sc_dur_mk. If trig_dur is fine, proceed')
                    #print('imec trig : ', trig_dur, 'mkturk time : ', sc_dur_mk/1000)
                    print(len(np.where(sc_dur_mk <0)[0]), ' negative sc durations in mkturk file') 

                if t_diff_first < t_diff_last:
                    # adds nan value at the end of imec trigger
                    print('adds nan values at the end of imec trigger')
                    trig_on_time = np.concatenate((trig_on_time,np.nan* np.ones(n_diff)))
                    trig_off_time = np.concatenate((trig_off_time,np.nan* np.ones(n_diff)))
                else:
                    # adds nan value at the beginning of imec trigger
                    print('adds nan values at the beginning of imec trigger')
                    trig_on_time = np.concatenate((np.nan* np.ones(n_diff), trig_on_time))
                    trig_off_time = np.concatenate((np.nan* np.ones(n_diff), trig_off_time))

                n_scs = len(trig_on_time)
                n_scs_imec[idx] = n_scs

            elif n_trials_mk < n_scs: # more imec sample commands than mkturk trials
                print('more imec trigs than mkturk trials')
                n_diff=  n_scs - n_trials_mk  
                t_diff_first = np.nansum(np.abs(sc_dur_mk/1000 - trig_dur[0:n_trials_mk]))
                t_diff_last = np.nansum(np.abs(sc_dur_mk/1000-trig_dur[n_diff:n_scs]))
                print(f'first {n_diff} trig :', trig_dur[0:n_diff], f'last {n_diff} trig : ', trig_dur[n_trials_mk:n_scs])
                print(f'aligned to the first {n_trials_mk} imec trigs:', t_diff_first,f'aligned to the last {n_trials_mk} imec trigs:', t_diff_last)
                if t_diff_first > 5 and t_diff_last > 5: # difference is more than 5 seconds 
                    print('difference is too big on both ends. Check trig_dur and sc_dur_mk')
                    #print('imec trig : ', trig_dur, 'mkturk time : ', sc_dur_mk/1000)
                    
                    # sometimes mkturk sc is negative, and most of times this is why t_diff_first and t_diff_last is large 
                    print(len(np.where(sc_dur_mk <0)[0]), ' negative sc durations in mkturk file') 
                    ##### IMPORTANT! Sometimes there are 0s in imec trigger. See if removing these 0s help
                    print(len(np.where(trig_dur==0)[0]), ' 0s found in imec trigger')
                    if len(np.where(trig_dur==0)[0]) == n_diff: 
                        print(np.nansum(np.abs(sc_dur_mk/1000 - trig_dur[trig_dur!=0])), ' diff after removing 0s from imec trigger')

                    #proceed_bool = input('proceed?')
                if len(np.where(trig_dur==0)[0]) == n_diff: 
                    print('removing 0s from imec trigger')
                    scs_ind = scs_ind[trig_dur !=0]
                    trig_on_time = trig_on_time[trig_dur !=0]
                    trig_off_time = trig_off_time[trig_dur !=0]
                    trig_dur = trig_dur[trig_dur!=0]
                else:
                    if  t_diff_last > t_diff_first:
                        scs_ind = scs_ind[0:n_trials_mk]
                        trig_on_time = trig_on_time[0:n_trials_mk]
                        trig_off_time = trig_off_time[0:n_trials_mk]
                        # remove trials that are excess n_mk
                        print(f'removing {n_scs-n_trials_mk} sample commands from the end')
                    else:
                        scs_ind = scs_ind[n_diff:n_scs]
                        trig_on_time = trig_on_time[n_diff:n_scs]
                        trig_off_time = trig_off_time[n_diff:n_scs]
                        # remove trials that are excess n_mk
                        print(f'removing {n_scs-n_trials_mk} sample commands from the beginning')
                
                scs_ind_imec[idx]= scs_ind
                n_scs = len(scs_ind)
                n_scs_imec[idx] = n_scs
            
            assert n_trials_mk == n_scs == len(trig_on_time) == len(trig_off_time)
            trig_on_time = np.repeat(trig_on_time,n_rsvp)
            trig_off_time = np.repeat(trig_off_time, n_rsvp)

            for n_stim in data_dict:
                data_dict[n_stim]['imec_trig_on'] = trig_on_time[n_stim]
                data_dict[n_stim]['imec_trig_off'] = trig_off_time[n_stim]

            # add short stim info 
            for n_stim in data_dict:
                data_dict[n_stim]['stim_info_short'] = gen_short_scene_info(data_dict[n_stim]['stim_info'])
            
            # save out data_dict

            save_out_file_name = 'data_dict_' + Path.Path(behav_file).stem
            #pickle.dump(data_dict, open(save_out_path / save_out_file_name, 'wb'), protocol = 2)

            data_dict_new = dict.fromkeys(range(n_stims_sess,len(data_dict)+n_stims_sess))
            for n_stim in data_dict_new:
                data_dict_new[n_stim] = data_dict[n_stim - n_stims_sess]
                data_dict_new[n_stim]['trial_num'] = data_dict[n_stim-n_stims_sess]['trial_num']  + n_trials_sess

            data_dict_all = {**data_dict_all, **data_dict_new}
            n_stims_sess += len(data_dict)
            n_trials_sess += n_trials_mk
            print(n_stims_sess)

print('total # of stimulus presentations prepared in this session: ', n_stims_sess)


# In[ ]:


for i in range(len(data_dict_all)):
    print(i, data_dict_all[i]['rsvp_num'])


# In[ ]:


save_out_file_name = 'data_dict_' + data_path.name
pickle.dump(data_dict_all, open(save_out_path / save_out_file_name, 'wb'), protocol = 2)


# In[ ]:


del data_dict
del data_dict_all


# In[ ]:


data_dict = pickle.load(open(save_out_path / save_out_file_name, 'rb'))


# In[ ]:


# Get unique stim info 


# get unique stims in the current session
unique_stim = []
stim_all = []
stim_t_all = [] # (start, end) #
stim_t_mk_all = []
stim_present_bool = []
stim_rsvp_num = []
stim_trial_num = []
reward_bool = []
stim_scenefile = []
stim_dur_all = []
stim_iti_dur_all =[]
for n_stim in data_dict:
    stim_all.append(data_dict[n_stim]['stim_info_short'])
    stim_rsvp_num.append(data_dict[n_stim]['rsvp_num'])
    stim_trial_num.append(data_dict[n_stim]['trial_num'])
    reward_bool.append(data_dict[n_stim]['reward'])
    
    t_on_mk = data_dict[n_stim]['imec_trig_on'] + data_dict[n_stim]['t_mk']/1000 

    
    if type(data_dict[n_stim]['ph_t_rise']) == float: # if the photodiode is availalbe. Photodiode value is saved at the onset of first stimulus of a trial
        t_on_ph = data_dict[n_stim]['imec_trig_on'] + data_dict[n_stim]['ph_t_rise']/1000 + np.unique(np.array(data_dict[n_stim]['stim_info'].loc[:,'dur'].tolist()))[0]/1000 * data_dict[n_stim]['rsvp_num'] \
                + data_dict[n_stim]['iti_dur']/1000 * data_dict[n_stim]['rsvp_num']
    
    else: 
        t_on_ph = np.nan

    if data_dict[n_stim]['t_mk'] == -1:
        stim_present_bool.append(0)
    else:
        stim_present_bool.append(1)

    stim_t_mk_all.append(t_on_mk)
    stim_t_all.append(t_on_ph) # 100 ms before the stimulus onset to end of stimulus + iti

    stim_scenefile.append(data_dict[n_stim]['scenefile'])

    if data_dict[n_stim]['stim_info_short'] not in unique_stim:
        unique_stim.append(data_dict[n_stim]['stim_info_short'])

    stim_dur_all.append(np.unique(np.array(data_dict[n_stim]['stim_info'].loc[:,'dur'].tolist()))[0]/1000)
    stim_iti_dur_all.append(data_dict[n_stim]['iti_dur'])

stim_all = np.array(stim_all)
stim_t_all = np.array(stim_t_all)
stim_t_mk_all = np.array(stim_t_mk_all)
stim_present_bool = np.array(stim_present_bool)
stim_rsvp_num = np.array(stim_rsvp_num)
reward_bool = np.array(reward_bool)
stim_scenefile = np.array(stim_scenefile)
stim_dur_all = np.array(stim_dur_all)
stim_trial_num = np.array(stim_trial_num)
stim_iti_dur_all= np.array(stim_iti_dur_all)


# In[ ]:


stim_info_sess = dict.fromkeys(unique_stim)


# In[ ]:


for stim in unique_stim:
    stim_info_sess[stim] = dict.fromkeys(['stim_ind', 't_on', 't_on_mk', 'dur', 'iti_dur','present_bool', 'rsvp_num', 'reward_bool', 'scenefile'])
    stim_info_sess[stim]['stim_ind'] = np.where(np.array(stim_all) == stim)[0]
    stim_info_sess[stim]['t_on_mk'] = stim_t_mk_all[stim_info_sess[stim]['stim_ind']] 
    stim_info_sess[stim]['t_on'] = stim_t_all[stim_info_sess[stim]['stim_ind']] #photodiode this should be default 
    stim_info_sess[stim]['dur'] = stim_dur_all[stim_info_sess[stim]['stim_ind']]
    stim_info_sess[stim]['iti_dur'] = stim_iti_dur_all[stim_info_sess[stim]['stim_ind']]/1000
    stim_info_sess[stim]['present_bool']= stim_present_bool[stim_info_sess[stim]['stim_ind']]
    stim_info_sess[stim]['rsvp_num'] = stim_rsvp_num[stim_info_sess[stim]['stim_ind']]
    stim_info_sess[stim]['trial_num'] = stim_trial_num[stim_info_sess[stim]['stim_ind']]
    stim_info_sess[stim]['reward_bool'] = reward_bool[stim_info_sess[stim]['stim_ind']]
    stim_info_sess[stim]['scenefile'] = stim_scenefile[stim_info_sess[stim]['stim_ind']]


# In[ ]:


print ('# of unique stimulus prepared during the session: ', len(unique_stim))


# In[ ]:


# save the stim info
print(save_out_path)
pickle.dump(stim_info_sess, open(save_out_path / 'stim_info_sess','wb'), protocol = 2 )


# In[ ]:


# write out all stims 

with open(save_out_path / 'stim_list.json', 'w') as f:
    json.dump(unique_stim, f)


# In[ ]:




