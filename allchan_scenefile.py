from natsort import os_sorted
import pickle
from pathlib import Path
import os
import sys
import numpy as np
from .utils_ephys import get_psth_byscenefile_allchans, gen_heatmap_byscenefile
from .utils_meta import init_dirs, get_chanmap
from .make_engram_path import BASE_DATA_PATH, BASE_SAVE_OUT_PATH


def allchan_scenefile(monkey: str, date: str):
    """
    Generate PSTH heatmaps for all channels organized by scene file.
    
    This function processes electrophysiology data for a given monkey and date, organizing
    the analysis by scene files across all recording channels. It initializes data paths,
    loads channel mapping and stimulus information, then generates PSTH (Peri-Stimulus Time Histogram)
    data and corresponding heatmaps for each session.
    
    Args:
        monkey: Identifier for the monkey subject (e.g., 'monkey1', 'monkey2').
        date: Recording date in string format.
    
    Notes:
        - Requires stim_info_sess files to exist in the save_out_path directories
        - Will attempt to load existing chanmap.npy or generate it from imroTbl if not found
        - Skips sessions without stim_info_sess unless it's the last session (exits with error)
        - Uses global ENGRAM_PATH for base data and output directories
        - Output files are saved to user-specific ephys directory under ENGRAM_PATH
    
    Raises:
        SystemExit: If stim_info_sess file doesn't exist for the last session in data_path_list.
    """

    base_data_path = BASE_DATA_PATH
    base_save_out_path = BASE_SAVE_OUT_PATH

    print(date)

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

    for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        data_path = Path(data_path)
        save_out_path = Path(save_out_path)
        plot_save_out_path = Path(plot_save_out_path)
        print(data_path)
        try: 
            stim_info_path= os_sorted(save_out_path.glob('stim_info_sess'))[0]
        except:
            if n == len(data_path_list) -1:
                sys.exit('stim info path doesn''t exist')
            else:
                continue

        print(stim_info_path)
        print('\nstim info sess found: ' + str(stim_info_path.exists()))

        try: 
            chanmap = np.load(save_out_path / 'chanmap.npy')
        except:
            chanmap, imroTbl = get_chanmap(data_path)

        get_psth_byscenefile_allchans(save_out_path)
            
        gen_heatmap_byscenefile(save_out_path, plot_save_out_path,chanmap)

if __name__ == "__main__":
    monkey = sys.argv[2]
    date = str(sys.argv[3])
    allchan_scenefile(monkey, date)