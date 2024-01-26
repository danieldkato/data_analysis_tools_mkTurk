import pandas as pd
import json
import math
import numpy as np
import pathlib as Path 

def getLongestArray(x):
    # getting number of trials from 
    n = 0;
    if (type(x) != dict):
        if (type(x) == list):
            n_new = len(x);
        else: 
            n_new = 0;

        return n_new;
    #if not an enumerable object
    else:
        for keys in x:
            if (keys != "baseVertexInd"):
                if (type(x[keys]) == list):
                    n_new = len(x[keys]);
                #IF array
                elif (type(x[keys]) == dict):
                    n_new = getLongestArray(x[keys]);
                #ELSE !array
                else:
                    n_new = 0;

                if (n_new > n):
                    n = n_new;
                #IF
            #IF array of raw vertexinds
        #FOR keys
    #IF object
    return n

def gen_scene_df(scenefile):

    if isinstance(scenefile, str) or isinstance(scenefile, Path.Path): # if the user provided the path to the scenefile
        scenefile = json.load(open(scenefile,'rb'))

    try: 
        n_scenes = scenefile['nimages']
    except:
        n_scenes = getLongestArray(scenefile)
    
    columns = ['size', 'posX', 'posY', 'posZ', 'targetX', 'targetY', 'targetZ', 'rotX', 'rotY', 'rotZ', 'visibility', 'filterVal', 'meta', 'dur']
    # columns are properties of objects

    # rows are objects in a scene, including lights and background if they exist 
    if 'CAMERAS' in list(scenefile.keys()):
        all_cam_ids = list(scenefile['CAMERAS'].keys())
    else:
        all_cam_ids = []

    if 'OBJECTS' in list(scenefile.keys()):
        all_obj_ids = list(scenefile['OBJECTS'].keys())
    else:
        all_obj_ids = []

    if 'LIGHTS' in list(scenefile.keys()):
        all_light_ids = list(scenefile['LIGHTS'].keys())
    else:
        all_light_ids = []

    if 'IMAGEFILTERS' in list(scenefile.keys()):
        all_imgf_ids = ['img_' + imgf for imgf in scenefile['IMAGEFILTERS'].keys()]
    else:
        all_imgf_ids = []

    if 'OBJECTFILTERS' in list(scenefile.keys()):
        all_objf_ids = ['obj_' + objf for objf in scenefile['OBJECTFILTERS'].keys()]
    else:
        all_objf_ids = []

    bkg_new = []
    if 'IMAGES' in list(scenefile.keys()):
        for bkg_s in scenefile['IMAGES']['imageidx']:
            if type(bkg_s) != int:
                if all(v is None for v in bkg_s) or all(v == '' for v in bkg_s):
                    bkg_new.append(-1)
                else:
                    bkg_new.append(bkg_s[0])
            else:
                bkg_new.append(bkg_s)
        all_bkg_ids = ['bkg' + str(b) for b in np.unique(np.sort(bkg_new)) if b != -1 ]
    else:
        all_bkg_ids =[]

    scene_df_all = []

    for n_s in range(n_scenes):
        # CAMERAS
        cam_df = pd.DataFrame(index = all_cam_ids,columns = columns)
        for cam in all_cam_ids:
            cam_df = pd.DataFrame(index = all_cam_ids,columns = columns)
            # posititon
            posX = scenefile["CAMERAS"][cam]["position"]['x']

            try: 
                posX = posX[n_s]
            except:
                posX = posX[0]

            posY = scenefile["CAMERAS"][cam]["position"]['y']

            try: 
                posY = posY[n_s]
            except:
                posY = posY[0]

            posZ = scenefile["CAMERAS"][cam]["position"]['z']

            try: 
                posZ = posZ[n_s]
            except:
                posZ = posZ[0]

            cam_df.loc[cam]['posX'] = posX
            cam_df.loc[cam]['posY'] = posY
            cam_df.loc[cam]['posZ'] = posZ

            # target
            try: 
                targetX = scenefile["CAMERAS"][cam]["targetTHREEJS"]['x']
            except:
                targetX = scenefile["CAMERAS"][cam]["targetInches"]['x']

            try:
                targetX = targetX[n_s]
            except:
                targetX = targetX[0]

            try:
                targetY = scenefile["CAMERAS"][cam]["targetTHREEJS"]['y']
            except: 
                targetY = scenefile["CAMERAS"][cam]["targetInches"]['y']

            try: 
                targetY = targetY[n_s]
            except:
                targetY = targetY[0]

            try: 
                targetZ = scenefile["CAMERAS"][cam]["targetTHREEJS"]['z']
            except:
                targetZ = scenefile["CAMERAS"][cam]["targetInches"]['z']

            try:
                targetZ = targetZ[n_s]
            except:
                targetZ = targetZ[0]

            cam_df.loc[cam]['targetX'] = targetX
            cam_df.loc[cam]['targetY'] = targetY
            cam_df.loc[cam]['targetZ'] = targetZ

             # visibility
            try:
                cam_df.loc[cam]['visibility'] = scenefile['CAMERAS'][cam]['visible'][n_s]
            except:
                cam_df.loc[cam]['visibility'] = scenefile['CAMERAS'][cam]['visible'][0]
                

        # OBJECTS
        obj_df = pd.DataFrame(index = all_obj_ids, columns = columns)
        
        for obj in all_obj_ids:
            obj_df.loc[obj]['type'] = 'object'
            # size
            try: 
                size = scenefile["OBJECTS"][obj]["sizeTHREEJS"]
            except:
                size = scenefile["OBJECTS"][obj]["sizeInches"]

            try:
                size = size[n_s]
            except:
                size = size[0]

            obj_df.loc[obj]['size'] = size

            # posititon
            try:
                posX = scenefile["OBJECTS"][obj]["positionTHREEJS"]['x']
            except:
                posX = scenefile["OBJECTS"][obj]["positionInches"]['x']

            try: 
                posX = posX[n_s]
            except:
                posX = posX[0]

            try: 
                posY = scenefile["OBJECTS"][obj]["positionTHREEJS"]['y']
            except:
                posY = scenefile["OBJECTS"][obj]["positionInches"]['y']

            try: 
                posY = posY[n_s]
            except:
                posY = posY[0]

            try:
                posZ = scenefile["OBJECTS"][obj]["positionTHREEJS"]['z']
            except:
                posZ = scenefile["OBJECTS"][obj]["positionInches"]['z']

            try: 
                posZ = posZ[n_s]
            except:
                posZ = posZ[0]


            obj_df.loc[obj]['posX'] = posX
            obj_df.loc[obj]['posY'] = posY
            obj_df.loc[obj]['posZ'] = posZ

            # rotation

            rotX = scenefile["OBJECTS"][obj]["rotationDegrees"]['x']

            try: 
                rotX = rotX[n_s]
            except:
                rotX = rotX[0]                

            rotY = scenefile["OBJECTS"][obj]["rotationDegrees"]['y']

            try:
                rotY = rotY[n_s]
            except:
                rotY = rotY[0]

            rotZ = scenefile["OBJECTS"][obj]["rotationDegrees"]['z']

            try:
                rotZ = rotZ[n_s]
            except:
                rotZ = rotZ[0]

            obj_df.loc[obj]['rotX'] = rotX
            obj_df.loc[obj]['rotY'] = rotY
            obj_df.loc[obj]['rotZ'] = rotZ

            # visibility
            try:
                obj_df.loc[obj]['visibility'] = scenefile['OBJECTS'][obj]['visible'][n_s]
            except:
                obj_df.loc[obj]['visibility'] = scenefile['OBJECTS'][obj]['visible'][0]
            # meta

            obj_df.loc[obj]['meta'] = scenefile['OBJECTS'][obj]['meshpath']
        
        # LIGHT
        light_df = pd.DataFrame(index = all_light_ids, columns = columns)
        
        for light in all_light_ids:
            
            # posititon
            posX = scenefile["LIGHTS"][light]["position"]['x']

            try: 
                posX = posX[n_s]
            except:
                posX = posX[0]

            posY = scenefile["LIGHTS"][light]["position"]['y']

            try: 
                posY = posY[n_s]
            except:
                posY = posY[0]

            posZ = scenefile["LIGHTS"][light]["position"]['z']

            try:
                posZ = posZ[n_s]
            except:
                posZ = posZ[0]

            light_df.loc[light]['posX'] = posX
            light_df.loc[light]['posY'] = posY
            light_df.loc[light]['posZ'] = posZ

            # visibility 
            try:
                light_df.loc[light]['visibility'] = scenefile['LIGHTS'][light]['visible'][n_s]
            except:
                light_df.loc[light]['visibility'] = scenefile['LIGHTS'][light]['visible'][0]

            
        # BACKGROUND
        
        bkg_df = pd.DataFrame(index = all_bkg_ids, columns = columns)

        if len(all_bkg_ids) != 0: # background is present in the scenefile
            for bkg_id in all_bkg_ids:
                bkg_df.loc[bkg_id]['type'] = 'background'
            if n_s <len(bkg_new):
                if bkg_new[n_s] != -1:
                    bkg_df.loc['bkg' + str(bkg_new[n_s])]['visibility'] = 1
                    bkg_df.loc['bkg' + str(bkg_new[n_s])]['meta'] = scenefile['IMAGES']['imagebag']
                    try:
                        bkg_df.loc['bkg' + str(bkg_new[n_s])]['size'] = scenefile['IMAGES']['sizeTHREEJS'][0] 
                    except:
                        bkg_df.loc['bkg' + str(bkg_new[n_s])]['size'] = scenefile['IMAGES']['sizeInches'][0] 

        if len(all_bkg_ids) == 1: # background applies to all scenes 
            if bkg_new[0] != -1:
                bkg_df.loc['bkg' + str(bkg_new[0])]['visibility'] = 1
                bkg_df.loc['bkg' + str(bkg_new[0])]['meta'] = scenefile['IMAGES']['imagebag']
                try:
                    bkg_df.loc['bkg' + str(bkg_new[0])]['size'] = scenefile['IMAGES']['sizeTHREEJS'][0] 
                except:
                    bkg_df.loc['bkg' + str(bkg_new[0])]['size'] = scenefile['IMAGES']['sizeInches'][0] 


        # IMAGEFILTERS 
        imgf_df = pd.DataFrame(index = all_imgf_ids, columns = columns)
        for imgf in all_imgf_ids:
            imgf_df.loc[imgf]['type'] = 'imgFilter'
            if len(scenefile['IMAGEFILTERS'][imgf.split('_')[1]]) > 0:
                try: 
                    imgf_df.loc[imgf]['filterVal']= scenefile['IMAGEFILTERS'][imgf.split('_')[1]][n_s]
                except:
                    imgf_df.loc[imgf]['filterVal']= scenefile['IMAGEFILTERS'][imgf.split('_')[1]][0]                

        # OBJECTFILTERS
        objf_df = pd.DataFrame(index = all_objf_ids, columns = columns)
        for objf in all_objf_ids:
            objf_df.loc[objf]['type'] = 'objFilter'
            if len(scenefile['OBJECTFILTERS'][objf.split('_')[1]]) > 0:
                try:
                    objf_df.loc[objf]['filterVal']= scenefile['OBJECTFILTERS'][objf.split('_')[1]][n_s]
                except:
                    objf_df.loc[objf]['filterVal']= scenefile['OBJECTFILTERS'][objf.split('_')[1]][0]

                
        all_df = pd.concat((cam_df, obj_df,light_df,bkg_df, imgf_df,objf_df))

    
        # scene duration
        if type(scenefile['durationMS']) == int:
            all_df.loc[:,'dur'] = scenefile['durationMS']
        elif len(scenefile['durationMS']) == 1:
            all_df.loc[:,'dur'] = scenefile['durationMS'][0]
        else:
            all_df.loc[:,'dur'] = scenefile['durationMS'][n_s]

        scene_df_all.append(all_df)

        
    return scene_df_all

def create_data_mat(file):
    # creates a data dictionary
    
    params = ['behav_file', 'scenefile', 'trial_num', 'rsvp_num', 'stim_id', 'stim_info', 't_mk', 'sc_t_mk', 'ph_t_rise', 'reward', 'reward_dur', 'punish_dur', 'iti_dur', 'prestim_dur']
    m = json.load(open(file,'rb'))

    n_rsvp = m['TASK']['NRSVP']
    n_trials_prepared= len(m['TRIALEVENTS']['Sample']['0'])

    n_stims = n_rsvp * n_trials_prepared

    scene_df_all = []
    scenefile_all = []
    for scenefile_path, scenefile in zip(m['TASK']['ImageBagsSample'], m['SCENES']['SampleScenes']):
        scene_df = gen_scene_df(scenefile)
        n_scenes = len(scene_df)
        scene_df_all.extend(scene_df)
        scenefile_all.extend([scenefile_path] * n_scenes)

    data_dict = dict.fromkeys(range(n_stims),[])

    for n in range(n_stims):
        data_dict[n] = dict.fromkeys(params, [])
        data_dict[n]['behav_file'] = Path.Path(file).stem
        data_dict[n]['trial_num'] = math.floor(n/n_rsvp)
        data_dict[n]['rsvp_num'] = n % n_rsvp
        data_dict[n]['stim_id'] = int(m['TRIALEVENTS']['Sample'][str(data_dict[n]['rsvp_num'])][data_dict[n]['trial_num']])
        data_dict[n]['scenefile'] = scenefile_all[data_dict[n]['stim_id']]
        data_dict[n]['stim_info'] = scene_df_all[data_dict[n]['stim_id']]
        try: 
            data_dict[n]['t_mk'] = m['TRIALEVENTS']['TSequenceActualClip'][str(data_dict[n]['rsvp_num'])][data_dict[n]['trial_num']]
        except:
            data_dict[n]['t_mk'] = []

        try: 
            data_dict[n]['sc_t_mk'] = m['TRIALEVENTS']['SampleCommandReturnTime'][data_dict[n]['trial_num']]
        except:
            data_dict[n]['sc_t_mk'] = []

        try:
            data_dict[n]['ph_t_rise'] = m['TIMEEVENTS']['DAQ']['ph']['trise'][data_dict[n]['trial_num']]
        except:
            data_dict[n]['ph_t_rise'] = []

        data_dict[n]['reward'] = m['TRIALEVENTS']['NReward'][data_dict[n]['trial_num']]
        data_dict[n]['reward_dur'] = m['TASK']['RewardDuration']
        data_dict[n]['punish_dur'] = m['TASK']['PunishTimeOut']
        data_dict[n]['iti_dur'] = m['TASK']['InterTrialInterval']
        data_dict[n]['prestim_dur'] =m['TASK']['SamplePRE']

    return data_dict

def gen_short_scene_info(scene_df):
    # generates a short string to describe a scene

    visible_objs = scene_df[scene_df['visibility'] == 1].index
    # short info
    # object , background then camera
    scene_df_short = ''
    for obj in visible_objs:
        if scene_df.loc[obj]['type'] == 'object':
            str_ = scene_df.loc[obj]['meta'].split('/')
            obj_name = str_[len(str_)-2] + '/' + str_[len(str_)-1]
            if scene_df_short != '':
                obj_name = '_' + obj_name 
                
            scene_df_short = scene_df_short + obj_name + '_sz_' + str(scene_df.loc[obj]['size']) +\
            '_posX_' + str(scene_df.loc[obj]['posX']) + '_posY_' + str(scene_df.loc[obj]['posY']) + \
            '_posZ_' +  str(scene_df.loc[obj]['posZ']) + '_rotX_' + str(scene_df.loc[obj]['rotX']) + \
            '_rotY_' + str(scene_df.loc[obj]['rotY']) + '_rotZ_' + str(scene_df.loc[obj]['rotZ']) 
    for obj in visible_objs:
        if scene_df.loc[obj]['type'] == 'background':
            str_ = scene_df.loc[obj]['meta'].split('/')
            obj_name = str_[len(str_)-2] + '/' + str_[len(str_)-1]
            if scene_df_short != '':
                obj_name = '_' + obj_name
            scene_df_short = scene_df_short + obj_name + obj.split('bkg')[1] + \
            '_sz_' + str(scene_df.loc[obj]['size']) 

    for obj in visible_objs:
        if scene_df.loc[obj]['type'] == 'light':
            obj_name = obj
            if scene_df_short != '':
                obj_name = '_' + obj_name
            scene_df_short = scene_df_short + obj_name +\
            '_posX_' + str(scene_df.loc[obj]['posX']) + '_posY_' + str(scene_df.loc[obj]['posY']) + \
            '_posZ_' +  str(scene_df.loc[obj]['posZ'])

    for obj in visible_objs:
        if scene_df.loc[obj]['type'] == 'camera':
            obj_name = obj
            if scene_df_short != '':
                obj_name = '_' + obj_name
            scene_df_short = scene_df_short + obj_name +\
            '_posX_' + str(scene_df.loc[obj]['posX']) + '_posY_' + str(scene_df.loc[obj]['posY']) + \
            '_posZ_' +  str(scene_df.loc[obj]['posZ']) + '_targetX_' + str(scene_df.loc[obj]['targetX']) + \
            '_targetY_' + str(scene_df.loc[obj]['targetY']) + '_targetZ_' + str(scene_df.loc[obj]['targetZ']) 

    imgFilter = scene_df[(scene_df['filterVal'].notna()) & (scene_df['filterVal'] != "")].index
    for imgF in imgFilter:
        imgF_name = imgF
        if scene_df_short != '':
            imgF_name = '_' + imgF
        scene_df_short = scene_df_short + imgF_name + '_' + str(scene_df.loc[imgF]['filterVal'])

    return scene_df_short

