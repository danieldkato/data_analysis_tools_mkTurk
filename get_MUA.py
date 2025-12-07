# %%
#IMPORTS
import sys
import matplotlib.pyplot as plt
from utils_meta import get_recording_path
from pathlib import Path
import socket 
host = socket.gethostname()
import sys
sys.path
sys.path.append(r"c:\users\hiyun\anaconda3\lib\site-packages")
# if 'rc.zi.columbia.edu' in host: 
#     sys.path.append('/mnt/smb/locker/issa-locker/users/Ryan/code/python/Python_SpikeGLXtools')
#     sys.path.append('/share/issa/users/rjm2225/.conda/envs/spike-analysis-env/lib/python3.7/site-packages')
# else:   
#     sys.path.append(r'C:\Users\Alex\Documents\Python Scripts\Python_SpikeGLXtools')

#sys.path.append(r'C:\Users\Alex\Documents\Python Scripts\Python_SpikeGLXtools')
#sys.path.append('/mnt/smb/locker/issa-locker/users/Ryan/code/python/Python_SpikeGLXtools')
#sys.path.append('/share/issa/users/rjm2225/.conda/envs/spike-analysis-env/lib/python3.7/site-packages')
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta, SampRate, ChanGainsIM,ChannelCountsIM, ChannelCountsNI, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
import numpy as np 
import os
from pprint import pprint
import time 
#!pip install --user cupy
from scipy import signal
try:
    import cupy as cp
    import cupyx.scipy as csp
except ImportError:
    pass

import gc
import subprocess
gc.collect(generation=2) 

from natsort import os_sorted
import glob


# %%
# SET PATHS, RECORDING INFO

import socket 
host = socket.gethostname()

#engram_path = Path('Z:/')
engram_path = Path(os.path.join('/', 'mnt', 'smb', 'locker', 'issa-locker'))

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'
monkey = sys.argv[1]
date = sys.argv[2]

folder_path = next((base_data_path/monkey).glob('*' + date + '*'))
data_path_list = get_recording_path(base_data_path,monkey,date)
for data_path in data_path_list:
    data_path = Path(data_path)
    print('\nData path found: '+ str(data_path.exists()))

    meta_iter = data_path.glob('*ap.meta')
    bin_iter = data_path.glob('*ap.bin')   
    meta_path = next(meta_iter)
    bin_path = next(bin_iter) 

    # %%
    # READ META FILE 
    meta = readMeta(bin_path)
    if ('fileTimeSecs' in meta.keys()): 
        Fs = float(meta['imSampRate'])
        filesize = float(meta['fileSizeBytes'])
        file_len_sec = float(meta['fileTimeSecs'])
        file_len_samps = int(file_len_sec * Fs)
        n_chans = int(meta['nSavedChans'])-1
    else: 
        print(f'meta file time not saved. recovering this info from OS file size:')
        n_chans = 384
        Fs = 30000.0
        # get filesize from the os due to frozen SpikeGLX
        filesize_os = os.path.getsize(bin_path)
        file_len_pred = filesize_os/(Fs*2*(n_chans+1)) # 385 channels, 2 bytes per sample
        print(f'file size: {filesize_os}')
        print(f'file len (sec): {file_len_pred}')
        file_len_sec = file_len_pred
        filesize = filesize_os
        meta['fileSizeBytes'] = filesize
        meta['nSavedChans'] = 385
        meta['imroTbl'] = '(0,384)(0 0 1 500 250 0)(1 0 1 500 250 0)(2 0 1 500 250 0)(3 0 1 500 250 0)(4 0 1 500 250 0)(5 0 1 500 250 0)(6 0 1 500 250 0)(7 0 1 500 250 0)(8 0 1 500 250 0)(9 0 1 500 250 0)(10 0 1 500 250 0)(11 0 1 500 250 0)(12 0 1 500 250 0)(13 0 1 500 250 0)(14 0 1 500 250 0)(15 0 1 500 250 0)(16 0 1 500 250 0)(17 0 1 500 250 0)(18 0 1 500 250 0)(19 0 1 500 250 0)(20 0 1 500 250 0)(21 0 1 500 250 0)(22 0 1 500 250 0)(23 0 1 500 250 0)(24 0 1 500 250 0)(25 0 1 500 250 0)(26 0 1 500 250 0)(27 0 1 500 250 0)(28 0 1 500 250 0)(29 0 1 500 250 0)(30 0 1 500 250 0)(31 0 1 500 250 0)(32 0 1 500 250 0)(33 0 1 500 250 0)(34 0 1 500 250 0)(35 0 1 500 250 0)(36 0 1 500 250 0)(37 0 1 500 250 0)(38 0 1 500 250 0)(39 0 1 500 250 0)(40 0 1 500 250 0)(41 0 1 500 250 0)(42 0 1 500 250 0)(43 0 1 500 250 0)(44 0 1 500 250 0)(45 0 1 500 250 0)(46 0 1 500 250 0)(47 0 1 500 250 0)(48 0 1 500 250 0)(49 0 1 500 250 0)(50 0 1 500 250 0)(51 0 1 500 250 0)(52 0 1 500 250 0)(53 0 1 500 250 0)(54 0 1 500 250 0)(55 0 1 500 250 0)(56 0 1 500 250 0)(57 0 1 500 250 0)(58 0 1 500 250 0)(59 0 1 500 250 0)(60 0 1 500 250 0)(61 0 1 500 250 0)(62 0 1 500 250 0)(63 0 1 500 250 0)(64 0 1 500 250 0)(65 0 1 500 250 0)(66 0 1 500 250 0)(67 0 1 500 250 0)(68 0 1 500 250 0)(69 0 1 500 250 0)(70 0 1 500 250 0)(71 0 1 500 250 0)(72 0 1 500 250 0)(73 0 1 500 250 0)(74 0 1 500 250 0)(75 0 1 500 250 0)(76 0 1 500 250 0)(77 0 1 500 250 0)(78 0 1 500 250 0)(79 0 1 500 250 0)(80 0 1 500 250 0)(81 0 1 500 250 0)(82 0 1 500 250 0)(83 0 1 500 250 0)(84 0 1 500 250 0)(85 0 1 500 250 0)(86 0 1 500 250 0)(87 0 1 500 250 0)(88 0 1 500 250 0)(89 0 1 500 250 0)(90 0 1 500 250 0)(91 0 1 500 250 0)(92 0 1 500 250 0)(93 0 1 500 250 0)(94 0 1 500 250 0)(95 0 1 500 250 0)(96 0 1 500 250 0)(97 0 1 500 250 0)(98 0 1 500 250 0)(99 0 1 500 250 0)(100 0 1 500 250 0)(101 0 1 500 250 0)(102 0 1 500 250 0)(103 0 1 500 250 0)(104 0 1 500 250 0)(105 0 1 500 250 0)(106 0 1 500 250 0)(107 0 1 500 250 0)(108 0 1 500 250 0)(109 0 1 500 250 0)(110 0 1 500 250 0)(111 0 1 500 250 0)(112 0 1 500 250 0)(113 0 1 500 250 0)(114 0 1 500 250 0)(115 0 1 500 250 0)(116 0 1 500 250 0)(117 0 1 500 250 0)(118 0 1 500 250 0)(119 0 1 500 250 0)(120 0 1 500 250 0)(121 0 1 500 250 0)(122 0 1 500 250 0)(123 0 1 500 250 0)(124 0 1 500 250 0)(125 0 1 500 250 0)(126 0 1 500 250 0)(127 0 1 500 250 0)(128 0 1 500 250 0)(129 0 1 500 250 0)(130 0 1 500 250 0)(131 0 1 500 250 0)(132 0 1 500 250 0)(133 0 1 500 250 0)(134 0 1 500 250 0)(135 0 1 500 250 0)(136 0 1 500 250 0)(137 0 1 500 250 0)(138 0 1 500 250 0)(139 0 1 500 250 0)(140 0 1 500 250 0)(141 0 1 500 250 0)(142 0 1 500 250 0)(143 0 1 500 250 0)(144 0 1 500 250 0)(145 0 1 500 250 0)(146 0 1 500 250 0)(147 0 1 500 250 0)(148 0 1 500 250 0)(149 0 1 500 250 0)(150 0 1 500 250 0)(151 0 1 500 250 0)(152 0 1 500 250 0)(153 0 1 500 250 0)(154 0 1 500 250 0)(155 0 1 500 250 0)(156 0 1 500 250 0)(157 0 1 500 250 0)(158 0 1 500 250 0)(159 0 1 500 250 0)(160 0 1 500 250 0)(161 0 1 500 250 0)(162 0 1 500 250 0)(163 0 1 500 250 0)(164 0 1 500 250 0)(165 0 1 500 250 0)(166 0 1 500 250 0)(167 0 1 500 250 0)(168 0 1 500 250 0)(169 0 1 500 250 0)(170 0 1 500 250 0)(171 0 1 500 250 0)(172 0 1 500 250 0)(173 0 1 500 250 0)(174 0 1 500 250 0)(175 0 1 500 250 0)(176 0 1 500 250 0)(177 0 1 500 250 0)(178 0 1 500 250 0)(179 0 1 500 250 0)(180 0 1 500 250 0)(181 0 1 500 250 0)(182 0 1 500 250 0)(183 0 1 500 250 0)(184 0 1 500 250 0)(185 0 1 500 250 0)(186 0 1 500 250 0)(187 0 1 500 250 0)(188 0 1 500 250 0)(189 0 1 500 250 0)(190 0 1 500 250 0)(191 0 1 500 250 0)(192 0 1 500 250 0)(193 0 1 500 250 0)(194 0 1 500 250 0)(195 0 1 500 250 0)(196 0 1 500 250 0)(197 0 1 500 250 0)(198 0 1 500 250 0)(199 0 1 500 250 0)(200 0 1 500 250 0)(201 0 1 500 250 0)(202 0 1 500 250 0)(203 0 1 500 250 0)(204 0 1 500 250 0)(205 0 1 500 250 0)(206 0 1 500 250 0)(207 0 1 500 250 0)(208 0 1 500 250 0)(209 0 1 500 250 0)(210 0 1 500 250 0)(211 0 1 500 250 0)(212 0 1 500 250 0)(213 0 1 500 250 0)(214 0 1 500 250 0)(215 0 1 500 250 0)(216 0 1 500 250 0)(217 0 1 500 250 0)(218 0 1 500 250 0)(219 0 1 500 250 0)(220 0 1 500 250 0)(221 0 1 500 250 0)(222 0 1 500 250 0)(223 0 1 500 250 0)(224 0 1 500 250 0)(225 0 1 500 250 0)(226 0 1 500 250 0)(227 0 1 500 250 0)(228 0 1 500 250 0)(229 0 1 500 250 0)(230 0 1 500 250 0)(231 0 1 500 250 0)(232 0 1 500 250 0)(233 0 1 500 250 0)(234 0 1 500 250 0)(235 0 1 500 250 0)(236 0 1 500 250 0)(237 0 1 500 250 0)(238 0 1 500 250 0)(239 0 1 500 250 0)(240 0 1 500 250 0)(241 0 1 500 250 0)(242 0 1 500 250 0)(243 0 1 500 250 0)(244 0 1 500 250 0)(245 0 1 500 250 0)(246 0 1 500 250 0)(247 0 1 500 250 0)(248 0 1 500 250 0)(249 0 1 500 250 0)(250 0 1 500 250 0)(251 0 1 500 250 0)(252 0 1 500 250 0)(253 0 1 500 250 0)(254 0 1 500 250 0)(255 0 1 500 250 0)(256 0 1 500 250 0)(257 0 1 500 250 0)(258 0 1 500 250 0)(259 0 1 500 250 0)(260 0 1 500 250 0)(261 0 1 500 250 0)(262 0 1 500 250 0)(263 0 1 500 250 0)(264 0 1 500 250 0)(265 0 1 500 250 0)(266 0 1 500 250 0)(267 0 1 500 250 0)(268 0 1 500 250 0)(269 0 1 500 250 0)(270 0 1 500 250 0)(271 0 1 500 250 0)(272 0 1 500 250 0)(273 0 1 500 250 0)(274 0 1 500 250 0)(275 0 1 500 250 0)(276 0 1 500 250 0)(277 0 1 500 250 0)(278 0 1 500 250 0)(279 0 1 500 250 0)(280 0 1 500 250 0)(281 0 1 500 250 0)(282 0 1 500 250 0)(283 0 1 500 250 0)(284 0 1 500 250 0)(285 0 1 500 250 0)(286 0 1 500 250 0)(287 0 1 500 250 0)(288 0 1 500 250 0)(289 0 1 500 250 0)(290 0 1 500 250 0)(291 0 1 500 250 0)(292 0 1 500 250 0)(293 0 1 500 250 0)(294 0 1 500 250 0)(295 0 1 500 250 0)(296 0 1 500 250 0)(297 0 1 500 250 0)(298 0 1 500 250 0)(299 0 1 500 250 0)(300 0 1 500 250 0)(301 0 1 500 250 0)(302 0 1 500 250 0)(303 0 1 500 250 0)(304 0 1 500 250 0)(305 0 1 500 250 0)(306 0 1 500 250 0)(307 0 1 500 250 0)(308 0 1 500 250 0)(309 0 1 500 250 0)(310 0 1 500 250 0)(311 0 1 500 250 0)(312 0 1 500 250 0)(313 0 1 500 250 0)(314 0 1 500 250 0)(315 0 1 500 250 0)(316 0 1 500 250 0)(317 0 1 500 250 0)(318 0 1 500 250 0)(319 0 1 500 250 0)(320 0 1 500 250 0)(321 0 1 500 250 0)(322 0 1 500 250 0)(323 0 1 500 250 0)(324 0 1 500 250 0)(325 0 1 500 250 0)(326 0 1 500 250 0)(327 0 1 500 250 0)(328 0 1 500 250 0)(329 0 1 500 250 0)(330 0 1 500 250 0)(331 0 1 500 250 0)(332 0 1 500 250 0)(333 0 1 500 250 0)(334 0 1 500 250 0)(335 0 1 500 250 0)(336 0 1 500 250 0)(337 0 1 500 250 0)(338 0 1 500 250 0)(339 0 1 500 250 0)(340 0 1 500 250 0)(341 0 1 500 250 0)(342 0 1 500 250 0)(343 0 1 500 250 0)(344 0 1 500 250 0)(345 0 1 500 250 0)(346 0 1 500 250 0)(347 0 1 500 250 0)(348 0 1 500 250 0)(349 0 1 500 250 0)(350 0 1 500 250 0)(351 0 1 500 250 0)(352 0 1 500 250 0)(353 0 1 500 250 0)(354 0 1 500 250 0)(355 0 1 500 250 0)(356 0 1 500 250 0)(357 0 1 500 250 0)(358 0 1 500 250 0)(359 0 1 500 250 0)(360 0 1 500 250 0)(361 0 1 500 250 0)(362 0 1 500 250 0)(363 0 1 500 250 0)(364 0 1 500 250 0)(365 0 1 500 250 0)(366 0 1 500 250 0)(367 0 1 500 250 0)(368 0 1 500 250 0)(369 0 1 500 250 0)(370 0 1 500 250 0)(371 0 1 500 250 0)(372 0 1 500 250 0)(373 0 1 500 250 0)(374 0 1 500 250 0)(375 0 1 500 250 0)(376 0 1 500 250 0)(377 0 1 500 250 0)(378 0 1 500 250 0)(379 0 1 500 250 0)(380 0 1 500 250 0)(381 0 1 500 250 0)(382 0 1 500 250 0)(383 0 1 500 250 0)'
        meta['imAiRangeMin'] = '-0.6'
        meta['imAiRangeMax'] = '0.6'
        meta['imMaxInt'] = '512'


    # %%
    # raw ap data 
    raw_data = makeMemMapRaw(bin_path, meta)

    # %%
    getTrig = True

    # %%
    # Get trigger line from imec

    if getTrig:

        dLineList = [6]
        # Get channel index of requested digital word dwReq
        dwReq = 0
        if meta['typeThis'] == 'imec':
            if 'snsApLfSy' not in list(meta.keys()):
                digCh = 384
            else:
                AP, LF, SY = ChannelCountsIM(meta)
                if SY == 0:
                    print("No imec sync channel saved.")
                    digArray = np.zeros((0), 'uint8')
                else:
                    digCh = AP + LF + dwReq
        else:
            MN, MA, XA, DW = ChannelCountsNI(meta)
            if dwReq > DW-1:
                print("Maximum digital word in file = %d" % (DW-1))
                digArray = np.zeros((0), 'uint8')
            else:
                digCh = MN + MA + XA + dwReq

        print(digCh)

        trig_dir = data_path / Path('imec_trig')
        trig_dir.mkdir(exist_ok=False)

    # %%


    # %%
    # high-pass filter
    high_pass_Hz = 300
    sos = signal.butter(3, high_pass_Hz, btype='high', fs=Fs, output='sos')

    # READ SOME DATA
    t_win = 20
    win_len_samps = int(t_win*Fs)
    file_len_samps = int(file_len_sec * Fs)

    n_chunks = int(np.ceil(file_len_sec/t_win))
    print(n_chunks)

    # %%
    # MU spike parameters: 

    # spike_thresh = -3 # standard deviations
    # reject_thresh = 20 # for rejection of noise spikes
    # min_dist = 20 # min number of samples between peaks (corresponds to 6.6 microsec)
    leftsamps = 15
    rightsamps = 30
    sw_len = leftsamps + rightsamps
    # total is 1.5 msec 

    # positive std as well - YJ 2023.11.13 
    # change
    spike_thresh = 4
    reject_thresh = 20 # for rejection of noise spikes
    min_dist = 20 # min number of samples between peaks (corresponds to 6.6 microsec)

    # %%
    MUA_dir = data_path / Path('MUA_4SD')
    MUA_dir.mkdir(exist_ok=False)

    # %%
    # open files for all chs
    wf_fhand = [open(MUA_dir / 'spike_wfs_ch{:0>3d}_tmp.npy'.format(ch),'wb') for ch in range(n_chans)]
    # save amplitude and time for both negative and positive peaks and save stds per channel - 2023.11.09 YJ
    # saved in a tuple (pos amp, neg amp) / (pos st, neg st)
    pk_fhand =  [open(MUA_dir / 'spike_pks_ch{:0>3d}_tmp.npy'.format(ch), 'wb') for ch in range(n_chans)]
    st_fhand = [open(MUA_dir / 'spike_ts_ch{:0>3d}_tmp.npy'.format(ch), 'wb') for ch in range(n_chans)]
    sd_fhand = [open(MUA_dir / 'spike_sds_ch{:0>3d}_tmp.npy'.format(ch), 'wb') for ch in range(n_chans)]
    # also save signlabel

    sl_fhand =  [open(MUA_dir / 'spike_sls_ch{:0>3d}_tmp.npy'.format(ch), 'wb') for ch in range(n_chans)]

    # %%
    try: 
        mempool = cp.get_default_memory_pool()
        print(mempool.used_bytes())
        print('cupy will be used')
    except:
        pass

    try: 
        device_name = torch.cuda.get_device_name()
        print(device_name)
        print(torch.cuda.mem_get_info())
        print('torch will be used')
    except:
        pass
    # %%
    total_t = 0

    for chunk in range(n_chunks):
        
        # CLEAN UP FOR SANITY
        if 'select_data' in locals(): 
            del select_data
        
        if 'conv_data' in locals(): 
            del conv_data
        
        if 'data_array' in locals(): 
            del data_array
        
        if 'car_data' in locals():
            del car_data
            
        if 'data_filt' in locals():
            del data_filt

        try: 
            mempool.free_all_blocks()
        except: 
            pass
        
        # BEGIN!
        print(f'iter {chunk+1} of {n_chunks}')
        t_stch = time.perf_counter()
        
        first_samp = (chunk*win_len_samps)
        last_samp = min([first_samp + win_len_samps, file_len_samps])
        try: 
            select_data = cp.array(np.array(raw_data[:n_chans, first_samp:last_samp]))
            print('using cupy')
        except NameError:
            try: 
                select_data = torch.asarray(np.array(raw_data[:n_chans, first_samp:last_samp]))
                print('using torch')
            except:
                select_data = np.array(raw_data[:n_chans, first_samp:last_samp])
        
        print(f'first samp is {first_samp}, last samp is {last_samp}')    
        chunk_start_sec = float(first_samp/Fs)
        print(f'start time is {chunk_start_sec}')
        
        # trig file

        if getTrig:
            digArray = raw_data[digCh,first_samp:last_samp].squeeze()
            ind = np.where(digArray != 0)[0] + first_samp
            save_file_name = f'trig_ind_{first_samp}_{last_samp}'
            np.save(Path(trig_dir, save_file_name), ind)
        
        # APPLY GAIN CORRECTION
        #print('gain correction')
        APgain,_ = ChanGainsIM(meta)
        #t1 = time.time()
        fI2V = float(meta['imAiRangeMax'])/int(meta['imMaxInt'])
        conv_all = fI2V / APgain
        try: 
            conv_all = cp.array(conv_all)
            conv_data = cp.multiply(conv_all, select_data.T)
        except NameError:
            try: 
                conv_all = torch.asarray(conv_all)
                conv_data = torch.multiply(conv_all, select_data.T)
            except:
                conv_data = np.multiply(conv_all, select_data.T)
        #print(conv_all.shape)
        #print(select_data.shape)

        conv_data = conv_data.T
        #t2 = time.time()
        #print(f'took {t2-t1}s')

        # clean up
        del select_data 
        del conv_all
        gc.collect(generation=2)
        
        # subtract channel-wise mean (zero channels)
        #print('mean and CAR subtraction')
        #t1 = time.time()
        try: 
            conv_data = cp.subtract(conv_data.T, cp.mean(conv_data, axis=1))
        except NameError:
            try: 
                conv_data = torch.subtract(conv_data.T,torch.mean(conv_data,axis = 1))
            except:
                conv_data = np.subtract(conv_data.T,np.mean(conv_data,axis = 1))


        conv_data = conv_data.T
        # CAR
        try: 
            car_data = cp.subtract(conv_data, cp.mean(conv_data, axis=0))
        except NameError:
            try:
                car_data = torch.subtract(conv_data, torch.mean(conv_data, axis=0))
            except:
                car_data = np.subtract(conv_data, np.mean(conv_data, axis = 0))

        del conv_data
        #t2 = time.time()
        #print(f'took {t2-t1}s')
        #print(data_array.shape)

        # high pass filter 
        #print('filtering')
        try: 
            data_filt = signal.sosfiltfilt(sos, cp.asnumpy(car_data), axis=1)
        except NameError:
            try:
                data_filt = signal.sosfiltfilt(sos, car_data.detach().cpu().numpy(), axis=1)
            except:
                data_filt = signal.sosfiltfilt(sos,car_data,axis = 1)

        del car_data 
        
        # MUA extraction 
        stds_bych = np.std(data_filt, axis=1)

        for ch, idx in enumerate(range(n_chans)): 
            #print(f'ch{ch} chunk {chunk}')
            ch_thresh = stds_bych[idx] * spike_thresh
            ch_thresh_rej = stds_bych[idx] * reject_thresh
            data_chunk = data_filt[idx,:]
            chunk_len = len(data_chunk)

            # [pks,_] = signal.find_peaks(-data_chunk, height=-ch_thresh, distance=min_dist)
            # sw_starts = pks-leftsamps
            # sw_ends = pks+rightsamps
            # spike_wfs_curr = np.empty([len(pks), sw_len])
            # spike_ts_curr = pks/Fs + chunk_start_sec

            # 2023.11.23.YJ 
            # negative peaks
            [pks_neg,_] = signal.find_peaks(-data_chunk, height = ch_thresh, distance = min_dist)
            # positive peaks
            [pks_pos, _] = signal.find_peaks(data_chunk, height = ch_thresh, distance = min_dist)

            pks = np.sort(np.concatenate((pks_neg, pks_pos)))

            # remove duplicate events 

            if len(pks) > 0:
                pks = pks[np.concatenate(([True],np.diff(pks)>sw_len))]
                
            sw_starts = pks-leftsamps
            sw_ends = pks+rightsamps
            
            spike_wfs_curr = np.empty([len(pks), sw_len])
            spike_pks_curr = np.empty([len(pks),3]) # store negative and positive peak of a waveform and a standard deviation used 
            spike_ts_curr = np.empty([len(pks),2])
            spike_sl_curr = np.empty(len(pks))
            
            #spike_ts_curr = pks/Fs + chunk_start_sec

            for spkidx, pk in enumerate(pks): 
                sw_start = max([1,sw_starts[spkidx]])
                sw_end = min([sw_ends[spkidx], chunk_len])
                wf_tmp = data_chunk[sw_start:sw_end]
                # zero pad if WF occurs at start or end
                if sw_starts[spkidx]<1:
                    pad = np.zeros([1, -sw_starts[spkidx]+1])
                    wf_tmp = np.append(wf_tmp, pad)
                if sw_ends[spkidx]>chunk_len: 
                    pad = np.zeros([1, sw_ends[spkidx]-chunk_len])
                    wf_tmp = np.append(wf_tmp, pad)

                spike_wfs_curr[spkidx,:] = wf_tmp
                
                # find the other peak within this window
                pks_amp = data_chunk[pk]
                pks_ts = pk/Fs + chunk_start_sec
                if pks_amp < 0 : # negative peak
                    # find the positive peak
                    ind = np.argmax(wf_tmp) 
                    spike_sl_curr[spkidx] = 1
        
                elif pks_amp > 0 : # positive peak
                    ind = np.argmin(wf_tmp)
                    spike_sl_curr[spkidx] = 0
                
                pks_opps_amp = wf_tmp[ind]
                pks_opps_ts = (ind - leftsamps) / Fs + pks_ts
                
                spike_pks_curr[spkidx,:] = (pks_amp, pks_opps_amp, stds_bych[idx])
                spike_ts_curr[spkidx,:] = (pks_ts, pks_opps_ts)

            # write out data to NPY files 
            np.save(wf_fhand[idx], spike_wfs_curr)

            np.save(st_fhand[idx], spike_ts_curr)
            np.save(pk_fhand[idx], spike_pks_curr)
            np.save(sd_fhand[idx], stds_bych[idx])
            np.save(sl_fhand[idx], spike_sl_curr)
            
            del spike_wfs_curr
            del spike_ts_curr
            del spike_pks_curr
            del spike_sl_curr
            
        t_endch = time.perf_counter()
        
        print(f'chunk took {t_endch-t_stch} sec')
        total_t += t_endch - t_stch
        
    print('total took', total_t)

    # %%
    for ch in range(n_chans): 
        wf_fhand[ch].close()
        st_fhand[ch].close()
        pk_fhand[ch].close()
        sd_fhand[ch].close()
        sl_fhand[ch].close()

    # %%
    # save to npz file 

    n_chans = 384
    for ch in range(n_chans): 
        print(f'ch{ch} of {n_chans}')
        # wf_files = print(next(wfs_iter))
        # st_files = print(next(sts_iter))
        # pk_files = print(next(pks_iter))
        
        wfs_iter = MUA_dir.glob('spike_wfs_ch{:03d}_tmp.npy'.format(ch))
        f_wf = open(next(wfs_iter), 'rb')
        #wfs_tmp = [np.load(f_wf) for _ in range(n_chunks)]
        wfs_tmp = []
        for _ in range(n_chunks):
            try:
                wfs_tmp.append(np.load(f_wf))
            except:
                print(_)
                pass
        wfs_all = np.concatenate(wfs_tmp)
        
        sts_iter = MUA_dir.glob('spike_ts_ch{:03d}_tmp.npy'.format(ch))
        f_ts = open(next(sts_iter), 'rb')
        #ts_tmp = [np.load(f_ts) for _ in range(n_chunks)]
        ts_tmp = []
        for _ in range(n_chunks):
            try:
                ts_tmp.append(np.load(f_ts))
            except:
                print(_)
                pass
        ts_all = np.concatenate(ts_tmp)

        pks_iter = MUA_dir.glob('spike_pks_ch{:0>3d}_tmp.npy'.format(ch))
        f_pks = open(next(pks_iter), 'rb')
        #pks_tmp = [np.load(f_pks) for _ in range(n_chunks)]
        pks_tmp = []
        for _ in range(n_chunks):
            try:
                pks_tmp.append(np.load(f_pks))
            except:
                print(_)
                pass
        pks_all = np.concatenate(pks_tmp)

        sds_iter = MUA_dir.glob('spike_sds_ch{:0>3d}_tmp.npy'.format(ch))
        f_sds = open(next(sds_iter), 'rb')
        
        sds_tmp = []
        for _ in range(n_chunks):
            try:
                sds_tmp.append(np.load(f_sds)[np.newaxis])
            except:
                print(_)
                pass
        sds_all = np.concatenate(sds_tmp)

        sls_iter = MUA_dir.glob('spike_sls_ch{:03d}_tmp.npy'.format(ch))
        f_sl = open(next(sls_iter), 'rb')
        sls_tmp = []
        for _ in range(n_chunks):
            try:
                sls_tmp.append(np.load(f_sl))
            except:
                print(_)
                pass
        sls_all = np.concatenate(sls_tmp)

    #     np.savez(MUA_dir / 'ch{:03d}'.format(ch), wfs=wfs_all, spike_ts=ts_all, spike_pks=pks_all\
    #             spike_thresh=spike_thresh, recording=recording, n_chunks_for_processing = n_chunks, \
    #             chunk_len_sec = t_win, high_pass_Hz = high_pass_Hz)

        np.save(MUA_dir / 'ch{:03d}_pks'.format(ch),pks_all)
        np.save(MUA_dir / 'ch{:03d}_ts'.format(ch),ts_all)
        np.save(MUA_dir / 'ch{:03d}_wfs'.format(ch),wfs_all)
        np.save(MUA_dir / 'ch{:03d}_sds'.format(ch),sds_all)
        np.save(MUA_dir / 'ch{:03d}_sls'.format(ch),sls_all)
        np.savez(MUA_dir / 'ch{:03d}_meta'.format(ch),spike_thresh=spike_thresh, recording=data_path, n_chunks_for_processing = n_chunks, \
                chunk_len_sec = t_win, high_pass_Hz = high_pass_Hz)
        
        f_wf.close()
        f_ts.close()
        f_pks.close()
        f_sds.close()
        f_sl.close()

    # %%
    # PERMANENTLY DELETE THE TMP FILES - MAKE SURE EVERYTHING WORKED BEFORE YOU DO THIS!!!
    n_chans = 384
    #for ch in range(384):
    wfs_iter = MUA_dir.glob('spike_wfs_ch*_tmp.npy'.format(ch))
    sts_iter = MUA_dir.glob('spike_ts_ch*_tmp.npy'.format(ch))
    pks_iter = MUA_dir.glob('spike_pks_ch*_tmp.npy'.format(ch))
    sds_iter = MUA_dir.glob('spike_sds_ch*_tmp.npy'.format(ch))
    sls_iter = MUA_dir.glob('spike_sls_ch*_tmp.npy'.format(ch))
    for wf in wfs_iter: 
        os.remove(wf)
        #subprocess.call(['rm', '-f', wf])
    for st in sts_iter: 
        os.remove(st)
        #subprocess.call(['rm', '-f', st])
    for pk in pks_iter:
        os.remove(pk)

    for sd in sds_iter:
        os.remove(sd)

    for sl in sls_iter:
        os.remove(sl)

    # %%
    trig_files = os_sorted(glob.glob(trig_dir.as_posix() + '/trig_ind_*'))

    assert len(trig_files) == n_chunks

    trig_is_on = []
    if len(trig_files) > 0:
        for file in trig_files:
            # check the order of files 
            trig_is_on.extend(np.load(file))
    else:
        print('no chunked trig indices files found')

    # %%
    trig_is_on = np.array(trig_is_on)
    trig_goff = np.where(np.diff(trig_is_on)!=1)[0]
    trig_gon = np.concatenate(([0],trig_goff+1))
    trig_goff = np.concatenate((trig_goff,[trig_is_on.shape[0]-1]))
    on_ind = trig_is_on[trig_gon]
    off_ind = trig_is_on[trig_goff]

    # %%
    save_out_name = 'trig_ind'
    np.save(Path(trig_dir,save_out_name),(on_ind,off_ind))

    # %%
    # remove chunked trig npy files 
    # MAKE sure you have trig_ind.npy file saved

    for file in trig_files:
        os.remove(file)

    # %%



