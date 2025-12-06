import numpy as np
from scipy.io import loadmat
import os
import pathlib as Path
import re
#from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta 

try:
    import cupy as cp
except ImportError:
    pass
try:
    import cupy._core as cp_core
except ImportError:
    try:
        import cupy.core as cp_core
    except ImportError:
        pass

def get_trig_pulses_dig(data):
    trig_is_on = np.array(np.where(data == 1)).squeeze()
    if trig_is_on.size > 0: 
        trig_goff = np.where(np.diff(trig_is_on)!=1)[0]
        trig_gon = np.concatenate(([0],trig_goff+1))
        trig_goff = np.concatenate((trig_goff,[trig_is_on.shape[0]-1]))
        on_ind = trig_is_on[trig_gon]
        off_ind = trig_is_on[trig_goff]
    else: 
        on_ind = np.array([])
        off_ind = np.array([])
    return on_ind, off_ind

def get_trig_pulses_analog(data,thresh):
    trig_is_on = np.array(np.where(data >= thresh)).squeeze()
    if trig_is_on.size > 0: 
        trig_goff = np.where(np.diff(trig_is_on)!=1)[0]
        trig_gon = np.concatenate(([0],trig_goff+1))
        trig_goff = np.concatenate((trig_goff,[trig_is_on.shape[0]-1]))
        on_ind = trig_is_on[trig_gon]
        off_ind = trig_is_on[trig_goff]
    else: 
        on_ind = np.array([])
        off_ind = np.array([])
    return on_ind, off_ind


def correct_pulses(on_inds, off_inds): 
    merge_arr = np.sort(np.concatenate((on_inds,off_inds)))
    np.diff(merge_arr)
    idx = np.where(np.diff(merge_arr) == 1)[0]
    print('correct pulses: {:d} merges required'.format(len(idx)))
    new_merge = np.delete(merge_arr, [idx, idx+1])
    new_merge = new_merge.reshape(int(new_merge.shape[0]/2),2).T
    on_inds_out = new_merge[0,:]
    off_inds_out = new_merge[1,:]
    
    return on_inds_out, off_inds_out

def convertMatToDict(filepath):
    data_mat = loadmat(filepath)
    data_dict = {}
    for mkeys in data_mat:
        if mkeys != '__header__' and mkeys != '__version__' and mkeys != '__globals__':
            #print(mkeys)
            data_dict[mkeys] = {}
            for sub_key in data_mat[mkeys].dtype.names:
                data_dict[mkeys][sub_key] = data_mat[mkeys][sub_key][0][0]
                
                if data_dict[mkeys][sub_key].shape == (1,1):
                    data_dict[mkeys][sub_key] = data_dict[mkeys][sub_key][0]
    
    return data_dict

def get_file_paths(folderPath):
    # returns paths for ap, lf, daq binary data 
    
    for f in os.listdir(folderPath):
        if os.path.isfile(os.path.join(folderPath,f)) and 'nidq.bin' in f:
            daqbinPath = os.path.join(folderPath,f)
            print('daqbinPath: ', daqbinPath)
            daqbinFullPath = Path(daqbinPath)

        elif os.path.isfile(os.path.join(folderPath,f)) and 'imec0.ap.bin' in f:
            apbinPath = os.path.join(folderPath,f)
            print('apbinPath: ', apbinPath)
            apbinFullPath = Path(apbinPath)
                    
        elif os.path.isfile(os.path.join(folderPath,f)) and 'imec0.lf.bin' in f:
            lfbinPath = os.path.join(folderPath,f)
            print('lfbinPath: ', lfbinPath)
            lfbinFullPath = Path(lfbinPath)

    
    return apbinFullPath, lfbinFullPath, daqbinFullPath

def free_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def make_bashfiles(shell_script_name, script_name, monkey, date, shell_script_dir=None):
        
    foldername = 'bashfiles_' + date

    os.makedirs(foldername,exist_ok= True)

    if shell_script_dir is not None:
        shell_script_fullpath = os.path.join(shell_script_dir, shell_script_name)	
    else:
        shell_script_fullpath = shell_script_name 
    with open(shell_script_fullpath,'r') as f:
        template = f.read()

    new_script = template.replace('date=', f'date={date}')
    new_script = new_script.replace('monkey=', f'monkey={monkey}')

    if 'py' in script_name:
        f_name = script_name.split('.')[0]
        new_script = new_script.replace('func', f'{script_name}')

    else:
        f_name = shell_script_name.split('.')[0]

    file_name = f'{foldername}/{f_name}_{date}.sh'

    print(file_name)
    with open(file_name, 'w') as rsh:      
        rsh.write(new_script)
