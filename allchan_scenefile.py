from natsort import os_sorted
import pickle
import pathlib as Path
import os
import sys
import numpy as np
from utils_ephys import * 
from utils_meta import * 
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta

engram_path = Path.Path('/mnt/smb/locker/issa-locker')

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

monkey = sys.argv[2]
date = str(sys.argv[3])
print(date)

data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
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
