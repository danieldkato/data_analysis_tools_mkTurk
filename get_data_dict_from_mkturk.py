import pickle
import sys
import os
import numpy as np
#import pathlib as Path
from pathlib import Path
from natsort import os_sorted
from .utils_mkturk import * 
from .utils_meta import * 
from sys import platform
from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta, SampRate, ChannelCountsIM,ChannelCountsNI, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
from itertools import groupby
from operator import itemgetter


import socket 
host = socket.gethostname()

#engram_path = Path('Z:/')
engram_path = Path(os.path.join('/', 'mnt', 'smb', 'locker', 'issa-locker'))
base_data_path = engram_path  / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

monkey = sys.argv[2]
date = sys.argv[3]
def get_filecode(filecode_ind, len_filecode,out_list):
        if len(filecode_ind) > len_filecode and len(filecode_ind) % len_filecode == 0:
            # chunk by 6 
            for i in range(0, len(filecode_ind), len_filecode):  
                out_list.append(filecode_ind[i:i+len_filecode])
        elif len(filecode_ind) == 6:
            out_list.append(filecode_ind)
        elif len(filecode_ind) > len_filecode: 
            out_list.append(filecode_ind[len(filecode_ind)-len_filecode:len(filecode_ind)])
        return out_list


data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)
for n,(data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
    print(save_out_path)
    print('\nsave out path found: '+ str(save_out_path.exists()))
    if not save_out_path.exists():
        os.makedirs(save_out_path, exist_ok= True)
        
    penetration = data_path.relative_to(base_data_path/monkey).as_posix().split('/')[0]

    trig_path = data_path / 'imec_trig'

    if not data_path.exists():
        if n == len(data_path_list)-1:
            sys.exit('data path doesn''t exist')
        else:
            continue
    elif not trig_path.exists():
        if n == len(data_path_list) - 1:
            sys.exit('trig path doesn''t exist')
        else:
            continue

    try:
        trig_on, trig_off = np.load(trig_path / 'trig_ind.npy')
    except:
        continue

    if len(trig_on)  ==0  and len(trig_off) == 0:
        sys.exit('trig file is empty')

    n_chans = 384
    Fs = 30000.0
    # meta_iter = data_path.glob('*ap.meta')
    # bin_iter = data_path.glob('*ap.bin')   
    # meta_path = next(meta_iter)
    # bin_path = next(bin_iter) 

    # # %%
    # # READ META FILE 
    # meta = readMeta(bin_path)
    # if ('fileTimeSecs' in meta.keys()): 
    #     filesize = float(meta['fileSizeBytes'])
    #     file_len_sec = float(meta['fileTimeSecs'])
    #     n_chans = int(meta['nSavedChans'])-1
    #     Fs = float(meta['imSampRate'])
    # else: 
    #     print(f'meta file time not saved. recovering this info from OS file size:')
    #     # get filesize from the os due to frozen SpikeGLX
    #     filesize_os = os.path.getsize(bin_path)
    #     n_chans = 384
    #     Fs = 30000.0
    #     file_len_pred = filesize_os/(Fs*2*(n_chans+1)) # 385 channels, 2 bytes per sample
    #     print(f'file size: {filesize_os}')
    #     print(f'file len (sec): {file_len_pred}')
    #     file_len_sec = file_len_pred
    #     filesize = filesize_os
    #     meta['fileSizeBytes'] = filesize
    #     meta['nSavedChans'] = 385
    #     meta['imroTbl'] = '(0,384)(0 0 1 500 250 0)(1 0 1 500 250 0)(2 0 1 500 250 0)(3 0 1 500 250 0)(4 0 1 500 250 0)(5 0 1 500 250 0)(6 0 1 500 250 0)(7 0 1 500 250 0)(8 0 1 500 250 0)(9 0 1 500 250 0)(10 0 1 500 250 0)(11 0 1 500 250 0)(12 0 1 500 250 0)(13 0 1 500 250 0)(14 0 1 500 250 0)(15 0 1 500 250 0)(16 0 1 500 250 0)(17 0 1 500 250 0)(18 0 1 500 250 0)(19 0 1 500 250 0)(20 0 1 500 250 0)(21 0 1 500 250 0)(22 0 1 500 250 0)(23 0 1 500 250 0)(24 0 1 500 250 0)(25 0 1 500 250 0)(26 0 1 500 250 0)(27 0 1 500 250 0)(28 0 1 500 250 0)(29 0 1 500 250 0)(30 0 1 500 250 0)(31 0 1 500 250 0)(32 0 1 500 250 0)(33 0 1 500 250 0)(34 0 1 500 250 0)(35 0 1 500 250 0)(36 0 1 500 250 0)(37 0 1 500 250 0)(38 0 1 500 250 0)(39 0 1 500 250 0)(40 0 1 500 250 0)(41 0 1 500 250 0)(42 0 1 500 250 0)(43 0 1 500 250 0)(44 0 1 500 250 0)(45 0 1 500 250 0)(46 0 1 500 250 0)(47 0 1 500 250 0)(48 0 1 500 250 0)(49 0 1 500 250 0)(50 0 1 500 250 0)(51 0 1 500 250 0)(52 0 1 500 250 0)(53 0 1 500 250 0)(54 0 1 500 250 0)(55 0 1 500 250 0)(56 0 1 500 250 0)(57 0 1 500 250 0)(58 0 1 500 250 0)(59 0 1 500 250 0)(60 0 1 500 250 0)(61 0 1 500 250 0)(62 0 1 500 250 0)(63 0 1 500 250 0)(64 0 1 500 250 0)(65 0 1 500 250 0)(66 0 1 500 250 0)(67 0 1 500 250 0)(68 0 1 500 250 0)(69 0 1 500 250 0)(70 0 1 500 250 0)(71 0 1 500 250 0)(72 0 1 500 250 0)(73 0 1 500 250 0)(74 0 1 500 250 0)(75 0 1 500 250 0)(76 0 1 500 250 0)(77 0 1 500 250 0)(78 0 1 500 250 0)(79 0 1 500 250 0)(80 0 1 500 250 0)(81 0 1 500 250 0)(82 0 1 500 250 0)(83 0 1 500 250 0)(84 0 1 500 250 0)(85 0 1 500 250 0)(86 0 1 500 250 0)(87 0 1 500 250 0)(88 0 1 500 250 0)(89 0 1 500 250 0)(90 0 1 500 250 0)(91 0 1 500 250 0)(92 0 1 500 250 0)(93 0 1 500 250 0)(94 0 1 500 250 0)(95 0 1 500 250 0)(96 0 1 500 250 0)(97 0 1 500 250 0)(98 0 1 500 250 0)(99 0 1 500 250 0)(100 0 1 500 250 0)(101 0 1 500 250 0)(102 0 1 500 250 0)(103 0 1 500 250 0)(104 0 1 500 250 0)(105 0 1 500 250 0)(106 0 1 500 250 0)(107 0 1 500 250 0)(108 0 1 500 250 0)(109 0 1 500 250 0)(110 0 1 500 250 0)(111 0 1 500 250 0)(112 0 1 500 250 0)(113 0 1 500 250 0)(114 0 1 500 250 0)(115 0 1 500 250 0)(116 0 1 500 250 0)(117 0 1 500 250 0)(118 0 1 500 250 0)(119 0 1 500 250 0)(120 0 1 500 250 0)(121 0 1 500 250 0)(122 0 1 500 250 0)(123 0 1 500 250 0)(124 0 1 500 250 0)(125 0 1 500 250 0)(126 0 1 500 250 0)(127 0 1 500 250 0)(128 0 1 500 250 0)(129 0 1 500 250 0)(130 0 1 500 250 0)(131 0 1 500 250 0)(132 0 1 500 250 0)(133 0 1 500 250 0)(134 0 1 500 250 0)(135 0 1 500 250 0)(136 0 1 500 250 0)(137 0 1 500 250 0)(138 0 1 500 250 0)(139 0 1 500 250 0)(140 0 1 500 250 0)(141 0 1 500 250 0)(142 0 1 500 250 0)(143 0 1 500 250 0)(144 0 1 500 250 0)(145 0 1 500 250 0)(146 0 1 500 250 0)(147 0 1 500 250 0)(148 0 1 500 250 0)(149 0 1 500 250 0)(150 0 1 500 250 0)(151 0 1 500 250 0)(152 0 1 500 250 0)(153 0 1 500 250 0)(154 0 1 500 250 0)(155 0 1 500 250 0)(156 0 1 500 250 0)(157 0 1 500 250 0)(158 0 1 500 250 0)(159 0 1 500 250 0)(160 0 1 500 250 0)(161 0 1 500 250 0)(162 0 1 500 250 0)(163 0 1 500 250 0)(164 0 1 500 250 0)(165 0 1 500 250 0)(166 0 1 500 250 0)(167 0 1 500 250 0)(168 0 1 500 250 0)(169 0 1 500 250 0)(170 0 1 500 250 0)(171 0 1 500 250 0)(172 0 1 500 250 0)(173 0 1 500 250 0)(174 0 1 500 250 0)(175 0 1 500 250 0)(176 0 1 500 250 0)(177 0 1 500 250 0)(178 0 1 500 250 0)(179 0 1 500 250 0)(180 0 1 500 250 0)(181 0 1 500 250 0)(182 0 1 500 250 0)(183 0 1 500 250 0)(184 0 1 500 250 0)(185 0 1 500 250 0)(186 0 1 500 250 0)(187 0 1 500 250 0)(188 0 1 500 250 0)(189 0 1 500 250 0)(190 0 1 500 250 0)(191 0 1 500 250 0)(192 0 1 500 250 0)(193 0 1 500 250 0)(194 0 1 500 250 0)(195 0 1 500 250 0)(196 0 1 500 250 0)(197 0 1 500 250 0)(198 0 1 500 250 0)(199 0 1 500 250 0)(200 0 1 500 250 0)(201 0 1 500 250 0)(202 0 1 500 250 0)(203 0 1 500 250 0)(204 0 1 500 250 0)(205 0 1 500 250 0)(206 0 1 500 250 0)(207 0 1 500 250 0)(208 0 1 500 250 0)(209 0 1 500 250 0)(210 0 1 500 250 0)(211 0 1 500 250 0)(212 0 1 500 250 0)(213 0 1 500 250 0)(214 0 1 500 250 0)(215 0 1 500 250 0)(216 0 1 500 250 0)(217 0 1 500 250 0)(218 0 1 500 250 0)(219 0 1 500 250 0)(220 0 1 500 250 0)(221 0 1 500 250 0)(222 0 1 500 250 0)(223 0 1 500 250 0)(224 0 1 500 250 0)(225 0 1 500 250 0)(226 0 1 500 250 0)(227 0 1 500 250 0)(228 0 1 500 250 0)(229 0 1 500 250 0)(230 0 1 500 250 0)(231 0 1 500 250 0)(232 0 1 500 250 0)(233 0 1 500 250 0)(234 0 1 500 250 0)(235 0 1 500 250 0)(236 0 1 500 250 0)(237 0 1 500 250 0)(238 0 1 500 250 0)(239 0 1 500 250 0)(240 0 1 500 250 0)(241 0 1 500 250 0)(242 0 1 500 250 0)(243 0 1 500 250 0)(244 0 1 500 250 0)(245 0 1 500 250 0)(246 0 1 500 250 0)(247 0 1 500 250 0)(248 0 1 500 250 0)(249 0 1 500 250 0)(250 0 1 500 250 0)(251 0 1 500 250 0)(252 0 1 500 250 0)(253 0 1 500 250 0)(254 0 1 500 250 0)(255 0 1 500 250 0)(256 0 1 500 250 0)(257 0 1 500 250 0)(258 0 1 500 250 0)(259 0 1 500 250 0)(260 0 1 500 250 0)(261 0 1 500 250 0)(262 0 1 500 250 0)(263 0 1 500 250 0)(264 0 1 500 250 0)(265 0 1 500 250 0)(266 0 1 500 250 0)(267 0 1 500 250 0)(268 0 1 500 250 0)(269 0 1 500 250 0)(270 0 1 500 250 0)(271 0 1 500 250 0)(272 0 1 500 250 0)(273 0 1 500 250 0)(274 0 1 500 250 0)(275 0 1 500 250 0)(276 0 1 500 250 0)(277 0 1 500 250 0)(278 0 1 500 250 0)(279 0 1 500 250 0)(280 0 1 500 250 0)(281 0 1 500 250 0)(282 0 1 500 250 0)(283 0 1 500 250 0)(284 0 1 500 250 0)(285 0 1 500 250 0)(286 0 1 500 250 0)(287 0 1 500 250 0)(288 0 1 500 250 0)(289 0 1 500 250 0)(290 0 1 500 250 0)(291 0 1 500 250 0)(292 0 1 500 250 0)(293 0 1 500 250 0)(294 0 1 500 250 0)(295 0 1 500 250 0)(296 0 1 500 250 0)(297 0 1 500 250 0)(298 0 1 500 250 0)(299 0 1 500 250 0)(300 0 1 500 250 0)(301 0 1 500 250 0)(302 0 1 500 250 0)(303 0 1 500 250 0)(304 0 1 500 250 0)(305 0 1 500 250 0)(306 0 1 500 250 0)(307 0 1 500 250 0)(308 0 1 500 250 0)(309 0 1 500 250 0)(310 0 1 500 250 0)(311 0 1 500 250 0)(312 0 1 500 250 0)(313 0 1 500 250 0)(314 0 1 500 250 0)(315 0 1 500 250 0)(316 0 1 500 250 0)(317 0 1 500 250 0)(318 0 1 500 250 0)(319 0 1 500 250 0)(320 0 1 500 250 0)(321 0 1 500 250 0)(322 0 1 500 250 0)(323 0 1 500 250 0)(324 0 1 500 250 0)(325 0 1 500 250 0)(326 0 1 500 250 0)(327 0 1 500 250 0)(328 0 1 500 250 0)(329 0 1 500 250 0)(330 0 1 500 250 0)(331 0 1 500 250 0)(332 0 1 500 250 0)(333 0 1 500 250 0)(334 0 1 500 250 0)(335 0 1 500 250 0)(336 0 1 500 250 0)(337 0 1 500 250 0)(338 0 1 500 250 0)(339 0 1 500 250 0)(340 0 1 500 250 0)(341 0 1 500 250 0)(342 0 1 500 250 0)(343 0 1 500 250 0)(344 0 1 500 250 0)(345 0 1 500 250 0)(346 0 1 500 250 0)(347 0 1 500 250 0)(348 0 1 500 250 0)(349 0 1 500 250 0)(350 0 1 500 250 0)(351 0 1 500 250 0)(352 0 1 500 250 0)(353 0 1 500 250 0)(354 0 1 500 250 0)(355 0 1 500 250 0)(356 0 1 500 250 0)(357 0 1 500 250 0)(358 0 1 500 250 0)(359 0 1 500 250 0)(360 0 1 500 250 0)(361 0 1 500 250 0)(362 0 1 500 250 0)(363 0 1 500 250 0)(364 0 1 500 250 0)(365 0 1 500 250 0)(366 0 1 500 250 0)(367 0 1 500 250 0)(368 0 1 500 250 0)(369 0 1 500 250 0)(370 0 1 500 250 0)(371 0 1 500 250 0)(372 0 1 500 250 0)(373 0 1 500 250 0)(374 0 1 500 250 0)(375 0 1 500 250 0)(376 0 1 500 250 0)(377 0 1 500 250 0)(378 0 1 500 250 0)(379 0 1 500 250 0)(380 0 1 500 250 0)(381 0 1 500 250 0)(382 0 1 500 250 0)(383 0 1 500 250 0)'
    #     meta['imAiRangeMin'] = '-0.6'
    #     meta['imAiRangeMax'] = '0.6'
    #     meta['imMaxInt'] = '512'



# %%
# behavior files in the data path 

    behav_file_list_orig = os_sorted(Path(base_data_path, monkey, penetration).glob('*.json'))
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

    # %%
    # %%
    
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

    # %%
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
    scs_ind_imec = np.array(scs_ind_imec,dtype = 'object')
    n_scs_imec = np.array(n_scs_imec)
    filecodes_imec = np.array(filecodes_imec)

    # %%
    # there are some cases where the first sample command is fired before the filecode. 
    # In this case, there is a very short latency between the onest of the filecode and the offset of the first sample command 
    # In the above code, it's likely that the first sample command was grouped to the previous file as the last sample command 

    for idx,(filecode, filecode_ind, scs_ind) in enumerate(zip(filecodes_imec, filecodes_ind_imec,scs_ind_imec)):

        # t_diff = (trig_on[scs_ind[0]] - trig_off[filecode_ind[len(filecode_ind)-1]]) /Fs * 1000
        # print('diff between offset of filecode and onset of the first sample command ', idx, t_diff)
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

    # %%
    for i,(filecode, filecode_ind, scs_ind, n_scs) in enumerate(zip(filecodes_imec, filecodes_ind_imec, scs_ind_imec, n_scs_imec)):
        print(i, 'filecode: ', filecode, ' # of sample commands: ', n_scs, len(scs_ind))

    # %%
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
        # skip filecodes with zero sample commands (likely some weird short pulses picked up or it's 
        behav_file = os_sorted(Path(base_data_path, monkey, penetration).glob('*' + f + '*.json'))
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

                    file_time = Path(b_f).stem.split('T')[1].split('_' + monkey)[0]
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
                        elif abs(n_scs - n_trials_mk) < 5:
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

    # %%
    assert len(behav_file_list) == len(n_scs_imec_to_analyze)

    # %%
    print(len(behav_file_list_orig) - len(behav_file_list), ' behavior files are unmatched \n')
    unmatched_behav_file = list(set(behav_file_list_orig) - set(behav_file_list))

    # %%
    # You can either try to find triggers that belong to this remaining behavior file 
    #  or skip it if it's a calibration file 

    # of trigs unaccounted for 
    print(len(trig_on) - (np.sum(n_scs_imec_to_analyze) + len_filecode * len(filecodes_imec_to_analyze)))

    # %%
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
            print(m['TASK']['ImageBagsSample'])

            try: 
                sc_off_mk =np.array(m['TRIALEVENTS']['SampleCommandOffReturnTime'],dtype=float)
            except:
                print('no SampleCommandOffReturnTime in the behavior file')
                sc_off_mk = [np.nan]

            try:
                sc_on_mk = np.array(m['TRIALEVENTS']['SampleCommandReturnTime'],dtype=float)
            except:
                print('no SampleCommandReturnTime in the behavior file')
                sc_on_mk = [np.nan]

            if 'NRSVPMax' in m['TASK'].keys():
                n_rsvp = max(m['TASK']['NRSVP'], m['TASK']['NRSVPMax'])
            else:
                n_rsvp = m['TASK']['NRSVP'] 

            trig_on_time = trig_on[scs_ind]/Fs 
            trig_off_time = trig_off[scs_ind]/Fs 
            trig_dur = trig_off_time - trig_on_time

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

                save_out_file_name = 'data_dict_' + Path(behav_file).stem
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

    # %%
    save_out_file_name = 'data_dict_' + data_path.name
    pickle.dump(data_dict_all, open(save_out_path / save_out_file_name, 'wb'), protocol = 2)

    # %%
 
    del data_dict
    del data_dict_all

    # %%
    data_dict = pickle.load(open(save_out_path / save_out_file_name, 'rb'))

    # %%
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

    # %%
    stim_info_sess = dict.fromkeys(unique_stim)


    # %%
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

    # %%
    print ('# of unique stimulus prepared during the session: ', len(unique_stim))

    # %%
    # save the stim info
    print(save_out_path)
    pickle.dump(stim_info_sess, open(save_out_path / 'stim_info_sess','wb'), protocol = 2 )

    # %%
    # write out all stims 

    with open(save_out_path / 'stim_list.json', 'w') as f:
        json.dump(unique_stim, f)