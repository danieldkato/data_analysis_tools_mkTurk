# useful functions for code
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from utils_mkturk import gen_scene_df, gen_short_scene_info
from make_engram_path import ENGRAM_PATH

def setup_paths():
    """Setup data paths based on hostname"""

    base_data_path = ENGRAM_PATH / 'Data'
    base_save_out_path = ENGRAM_PATH / 'users/Younah/ephys'

    return ENGRAM_PATH, base_data_path, base_save_out_path

@dataclass
class Config:
    """Configuration parameters for the analysis"""
    smooth_bin: int = 1
    binwidth_psth: float = 0.01
    t_before: float = 0.1
    t_after: float = 0.1
    stim_dur: float = 0.3
    visual_threshold_sd: float = 1.5
    visual_min_consecutive: int = 4
    visual_min_firing_rate: float = 10.0
    
    def __post_init__(self):
        # Process time bins
        t_early_raw = np.array([40, 100])
        t_late_raw = np.array([100, 200])

        self.t_early_bin = (t_early_raw / (self.smooth_bin * 10) + 10).astype(int)
        self.t_late_bin = (t_late_raw / (self.smooth_bin * 10) + 10).astype(int)
        

def check_consecutive(n: int, lst: list) -> bool:
    """Check if list contains n consecutive integers"""
    subs = [lst[i:i+n] for i in range(len(lst)) if len(lst[i:i+n]) == n]
    return any([(np.diff(sub) == 1).all() for sub in subs])

def identify_visual_channels(mean_psth_data: np.ndarray, config: Config, psth_bins: np.ndarray, chans_broken: list, 
                           brain_boundary: list) -> np.ndarray:
    """Identify visual channels based on response criteria"""
    n_chans, n_stims = mean_psth_data.shape[:2]
    
    visual_bool = np.zeros((n_chans, n_stims), dtype=bool)
    fr_bool = np.zeros((n_chans, n_stims), dtype=bool)
    
    baseline_end = int(np.where(psth_bins == 0)[0])
    
    for ch in range(n_chans):
        for stim in range(n_stims):
            # Baseline-corrected PSTH
            psth = mean_psth_data[ch, stim, :] - np.nanmean(mean_psth_data[ch, stim, :baseline_end])
            # Response period
            response_window = psth[config.t_early_bin[0]:config.t_late_bin[1]]
            mean_fr = np.nanmean(response_window)
            sd_fr = np.nanstd(response_window)
            
            if sd_fr > 0:
                # Check for consecutive bins above threshold
                above_threshold = np.where(response_window > config.visual_threshold_sd * sd_fr)[0]
                visual_bool[ch, stim] = check_consecutive(config.visual_min_consecutive, above_threshold)
                # Check firing rate threshold
                if visual_bool[ch, stim] and mean_fr * 100 > config.visual_min_firing_rate:
                    fr_bool[ch, stim] = True
    
    # Find channels that are visual for at least one stimulus
    ch_visual = np.where(np.sum(fr_bool, axis=1) > 0)[0]
    
    # Remove broken channels and channels beyond brain boundary
    ch_visual = np.sort(np.array(list(set(ch_visual).difference(chans_broken))))
    if brain_boundary:
        ch_visual = ch_visual[ch_visual <= brain_boundary[0]]
    
    return ch_visual

def get_time_centers(config, stim_dur: float) -> tuple[np.ndarray, np.ndarray]:
        """Get time bin centers for plotting."""
        n_bins = int(np.ceil((stim_dur + config.t_before + config.t_after) 
                            / config.binwidth_psth))
        psth_bins = np.linspace(-config.t_before, 
                               stim_dur + config.t_after, n_bins + 1)
        bincents = psth_bins[:-1] + config.binwidth_psth / 2

        return psth_bins, bincents

def load_meta_and_scene_df(base_data_path: Path, monkey: str, csv_name: str, scenefile_name: str) -> tuple:
    """Load metadata and scene dataframe"""
    meta = pd.read_csv((base_data_path / monkey /  'stim_info').as_posix() + f'/{csv_name}', header = None, names= ['category'])
    scene_df = gen_scene_df((base_data_path / monkey / 'scenefiles_update' / scenefile_name).as_posix())
    return meta, scene_df

def get_stim_info(meta:pd.DataFrame, scene_df: pd.DataFrame) -> tuple:
    """Extract stimulus info and categories"""
    stims = list(map(gen_short_scene_info,scene_df))
    cat_ind = [int(s.split('/')[1].split('_')[0]) for s in stims]
    # find the category for that stimulus
    ids_class = (meta['category'][cat_ind].to_list())
    return np.array(stims), np.array(ids_class)

def compute_fr(psth: dict, psth_meta: dict, stim_ids: list, bl_correct: bool = True) -> dict:
    """ Compute firing rates for different time windows"""
    config = Config()
    n_stim = len(stim_ids)
    fr = {'early': np.full(n_stim, np.nan),
            'late': np.full(n_stim, np.nan),
            'all': np.full(n_stim, np.nan)}

    for i, stim_id in enumerate(stim_ids):
        psth_data = psth[stim_id]
        psth_bins, _ = get_time_centers(config, psth_meta[stim_id]['stim_dur'])

        fr_bl = np.nanmean(psth_data[:, :int(np.where(psth_bins == 0)[0])])
        if bl_correct:
            fr['early'][i] = np.nanmean(psth_data[:, config.t_early_bin[0]:config.t_early_bin[1]]) - fr_bl
            fr['late'][i] = np.nanmean(psth_data[:, config.t_late_bin[0]:config.t_late_bin[1]]) - fr_bl
            fr['all'][i] = np.nanmean(psth_data[:, config.t_early_bin[0]:config.t_late_bin[1]]) - fr_bl
        else:
            fr['early'][i] = np.nanmean(psth_data[:, config.t_early_bin[0]:config.t_early_bin[1]])
            fr['late'][i] = np.nanmean(psth_data[:, config.t_late_bin[0]:config.t_late_bin[1]])
            fr['all'][i] = np.nanmean(psth_data[:, config.t_early_bin[0]:config.t_late_bin[1]])

    return fr


def organize_psth_by_category(psth: dict[str, np.ndarray], psth_meta: dict[str, dict], stim_ids: list[str], categories: list[str], category_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """ Organize PSTH data by category"""
    category_data = {cat: [] for cat in category_map}
    stim_ids_by_category = {cat: [] for cat in category_map}
    trial_counts = {cat: 0 for cat in category_map}
    
    for stim_id, cat in zip(stim_ids, categories):
        psth_data = psth[stim_id]
        print(psth_data)
        n_trials = psth_meta[stim_id]['n_trials']

        # stack arrays for each category

        for CAT in category_data:
            if cat in category_map[CAT]:
                category_data[CAT].append(psth_data)
                stim_ids_by_category[CAT].append(stim_id)
                trial_counts[CAT] += n_trials
        
    # Stack arrays for each category
    for CAT in category_data:
        if category_data[CAT]:
            category_data[CAT] = np.vstack(category_data[CAT])
        else:
            category_data[CAT] = np.array([])

    return category_data, stim_ids_by_category, trial_counts

def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safely divide arrays, handling division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result[denominator == 0] = np.nan
        return result

def compute_selectivity_index(target_psth, nontarget_psth):
    target_mean_psth = np.nanmean(target_psth * 100, axis=0)
    nontarget_mean_psth = np.nanmean(nontarget_psth * 100, axis=0)

    psth_sum = target_mean_psth + nontarget_mean_psth
    psth_diff = target_mean_psth - nontarget_mean_psth

    SI = safe_divide(psth_diff, psth_sum)
    return SI

def compute_selectivity_indices(psth_face, psth_body, psth_object, psth_scene):
    # face selectivity
    nonface_psth = np.vstack((psth_body, psth_object, psth_scene))
    FSI = compute_selectivity_index(psth_face, nonface_psth)

    # body selectivity
    nonbody_psth = np.vstack((psth_face, psth_object, psth_scene))
    BSI = compute_selectivity_index(psth_body, nonbody_psth)

    # scene selectivity
    nonscene_psth = np.vstack((psth_face, psth_object, psth_body))
    SSI = compute_selectivity_index(psth_scene, nonscene_psth)

    # object selectivity
    nonobject_psth = np.vstack((psth_face, psth_scene, psth_body))
    OSI = compute_selectivity_index(psth_object, nonobject_psth)
    
    CSI = dict()
    CSI['face'] = FSI
    CSI['body'] = BSI
    CSI['place']= SSI
    CSI['object'] = OSI

    return FSI, BSI, SSI, OSI, CSI 
    
class Plotter:
    def __init__(self):
        self.data = []

    def _format_psth_plot(self, ax,stim_dur: float, title: str):
        ax.set_xlabel('Time (s)', size= 15)
        ax.set_ylabel('spikes/sec', size = 15)
        ax.legend(bbox_to_anchor = (1,1))
        ax.set_title(title)
        ymin, ymax  = ax.get_ylim()
        ax.fill_between(np.linspace(0,stim_dur),ymin,ymax,color = 'y', alpha = 0.2)

class PerChannelPlotter:
    def __init__(self, config: Config):
        self.config = config

    def _format_psth_plot(self, ax,stim_dur: float, title: str):
        ax.set_xlabel('Time (s)', size= 15)
        ax.set_ylabel('spikes/sec', size = 15)
        ax.legend(bbox_to_anchor = (1,1))
        ax.set_title(title)
        ymin, ymax  = ax.get_ylim()
        ax.axvspan(0, stim_dur, color = 'y', alpha = 0.2)

    def _plot_category_psths(self, category_data: dict[str, np.ndarray], trial_counts: dict[str, int], stim_dur: float, n_chan: int):

        _, bincents = get_time_centers(self.config, stim_dur)

        fig, ax = plt.subplots()
        for cat in category_data:
            if 'stim' not in cat:
                mean_psth = np.nanmean(category_data[cat] *100,axis = 0) 
                sem_psth = np.nanstd(category_data[cat]*100,axis = 0) / np.sqrt(trial_counts[cat])
                plt.plot(bincents,mean_psth,label = cat + '_' + str(trial_counts[cat]))
                ax.fill_between(bincents, mean_psth-sem_psth, 
                                mean_psth+sem_psth,alpha=0.2)
        
        title = 'ch{:0>3d}'.format(n_chan)
        self._format_psth_plot(ax, stim_dur,title)

        return fig 
 

    def _plot_face_vs_nonface(self, category_data: dict[str, np.ndarray], trial_counts: dict[str, int], stim_dur: float, n_chan: int):
        """Plot face vs non-face PSTHs."""

        _, bincents = get_time_centers(self.config, stim_dur)

        fig, ax = plt.subplots()
        face_mean_psth = np.nanmean(category_data['face']*100,axis = 0)
        sem_psth = np.nanstd(category_data['face']*100,axis = 0) / np.sqrt(trial_counts['face'])
        plt.plot(bincents,face_mean_psth,label = 'face' + '_' + str(trial_counts['face']), color = 'tab:blue')
        ax.fill_between(bincents, face_mean_psth-sem_psth, 
                        face_mean_psth+sem_psth,alpha=0.2)
        
        nonface_category = [cat for cat in category_data if 'face' not in cat and 'stim' not in cat]
        psth_nonface = np.vstack([category_data[cat] for cat in nonface_category])
        n_nonface_trials = sum([trial_counts[cat] for cat in nonface_category]) 
        nonface_mean_psth = np.nanmean(psth_nonface*100,axis = 0)
        sem_psth = np.nanstd(psth_nonface*100,axis = 0) / np.sqrt(n_nonface_trials)
        plt.plot(bincents,nonface_mean_psth,label = 'nonface' + '_' + str(n_nonface_trials), color = 'tab:orange')
        ax.fill_between(bincents, nonface_mean_psth-sem_psth,   
                        nonface_mean_psth+sem_psth,alpha=0.2)
        
        title = 'ch{:0>3d}'.format(n_chan) + f'\n # face trials = {trial_counts["face"]}  # non-face trials ={n_nonface_trials}'
        self._format_psth_plot(ax, stim_dur, title)

        return fig 
    


# def load_meta_and_scene_df(csv_name: str, scenefile_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Load metadata and scene dataframe"""
#     meta = pd.read_csv((base_data_path / monkey /  'stim_info').as_posix() + f'/{csv_name}', header = None, names= ['category'])
#     scene_df = gen_scene_df((base_data_path / monkey / 'scenefiles_update' / scenefile_name).as_posix())
#     return meta, scene_df

# def get_stim_info(meta:pd.DataFrame, scene_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
#     """Extract stimulus info and categories"""
#     stims = list(map(gen_short_scene_info,scene_df))
#     cat_ind = [int(s.split('/')[1].split('_')[0]) for s in stims]
#     # find the category for that stimulus
#     ids_class = (meta['category'][cat_ind].to_list())
#     return np.array(stims), np.array(ids_class)

# scenefile_csv_combo = pd.DataFrame(columns = ['scenefile', 'csv'], data=[('20240404_SearchStimuli3_FBOP28_300ms_sz_0_6', 'SearchStimuli3.csv', ),
#                     ('20240404_SearchStimuli3_FBOP30_300ms', 'SearchStimuli3.csv'),
#                     ('20240117_SearchStimuli2_FBOP40_300ms', 'SearchStimuli2.csv'),
#                     ('20231117_IssaDiCarlo_FBOP40_300ms', 'FBOP.csv'),
#                     ('20240117_SearchStimuli_FBOP40_300ms', 'SearchStimuli.csv'),
#                     ('20240530_IssaDiCarlo_FBOP20_300ms', 'FBOP.csv')])


# # create a dataframe for SearchStimuli stimulus name and ids_class

# df = pd.DataFrame(columns= ['stims', 'id_class'])
# for ind, row in scenefile_csv_combo.iterrows():
#     scenefile = row['scenefile'] + '.json'
#     csv = row['csv']
#     meta, scene_df = load_meta_and_scene_df(csv, scenefile)
#     stims, ids_class = get_stim_info(meta, scene_df)
#     df_new = pd.DataFrame({'stims': stims, 'id_class': ids_class})
#     df = pd.concat([df, df_new], ignore_index=True)