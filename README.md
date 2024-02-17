# data_analysis_tools_mkTurk

functions for processing and analyzing mkTurk datafiles and electrophysiology data

- A session is a behavioral/neural recording for a particular date for a particular subject. A session can contain multiple mkTurk behavior data files, and multiple electrophysiology data files.
  
- get_data_dict_from_mkturk.ipynb aligns neural data to behavior data by searching for filecodes in the recorded triggers and match with names of behavior datafiles. This generates a pickled dictionary called stim_info_sess which groups all the presentations for a single stimulus within a session
  
- Each stimulus has the following info
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
