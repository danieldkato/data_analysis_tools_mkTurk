# data_analysis_tools_mkTurk

functions for processing and analyzing mkTurk datafiles and electrophysiology data

- A session is a behavioral/neural recording for a particular date for a particular subject. A session can contain multiple mkTurk behavior data files, and multiple electrophysiology data files.
  
- **get_data_dict_from_mkturk.ipynb** aligns neural data to behavior data by searching for filecodes in the recorded triggers and matching with the names of behavior datafiles. This generates a pickled dictionary called **data_dict_XX** which is indexed by trial/rsvp instances within a session. The particular stimulus for that trial/rsvp is recorded as a dataframe with the first column containing all elements within a scene. You can access this dataframe by `data_dict[<i>]['stim_info']`.
<img width="1068" alt="example_stim_info" src="https://github.com/younahjeon/data_analysis_tools_mkTurk/assets/38961502/91b0b6aa-5705-4083-8cb9-e89509704faf">

- **get_data_dict_from_mkturk.ipynb** also produces a dictionary called **stim_info_sess** which groups all the presentations for a single stimulus within a session. The keys of this dictionary correspond to `data_dict[<i>]['short_stim_info']`
  
- In **stim_info_sess**, each stimulus has the following info
    - 'stim_ind': index for the stimulus within a datafile
    - 't_on_mk': onset time of the stimulus calculated from mkTurk. Software timing
    - 't_on': onset time of the stimulus adjusted by photodiode. Hardware timing
    - 'dur': stimulus duration
    - 'iti_dur': grey screen duration after the stimulus
    - 'present_bool': boolean for whether the stimulus was on screen or not, could be that the monkey wasn't fixating, aborting a trial preemptively
    - 'rsvp_num' : the nth position within an rsvp sequence. Should be the same as 'trial_num' if the task was not rsvp
    - 'trial_num'
    - 'reward_bool': whether the trial was rewarded or not
    - 'scenefile': which scenefile the stimulus belongs to. Sometimes the same stimulus can be called in multiple scenefiles


## Pipeline: from multiunit data to psth per channel 
  After you have multiunit and trigger data ready,
1) run **get_data_dict_from_mkturk.ipynb**
2) upload both **analyze_bystim.py** and **analyze_bystim.sh** to the GPU cluster
3) edit **analyze_bystim.sh** with appropriate date and monkey
4) run **analyze_bystim.sh** 
