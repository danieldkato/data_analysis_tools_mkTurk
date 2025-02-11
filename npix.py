import os
import sys
import pathlib
import re
import numpy as np
import pandas as pd
from numpy.matlib import repmat
from data_analysis_tools_mkTurk.utils_meta import *

def generate_imro_table(length='short', parity='columnar', short_bank=0, n=384, typ=0,
    refID=0, ap_gain=500, lf_gain=250, ap_highpass=True, output_directory=None):
    """
    Generate IMRO table specifying which bank each neuropixel channel should be
    active on. 
    
    Note this function currently only produces correctly-formatted output for 
    probe-type 0; see https://billkarsh.github.io/SpikeGLX/help/imroTables/ for 
    detail. 

    Parameters
    ----------
    length : 'short'|'long', optional
        If 'short', all channels will be active on the same bank. If 'long', 
        at least two banks will include at least one active channe. The default 
        is 'short'.
    
    parity : 'columnar'|'alternating_sides', optional
        How to distribute active channels between banks if using long configuration. 
        If set to 'columnar', all channels on one side of each active bank will
        be active. If set to 'alternating_sides', active channels within a bank
        will alternate sides. Only used if `length` is set to 'long'. The default 
        is 'columnar'.
        
    short_bank : int, optional
        Which bank all channels should be active on if configuration is `length`
        is set to 'short'. The default is 0.
        
    n : int, optional
        Overall number of channels. The default is 384.

    typ : int, optional
        Probe type. The default is 0.

    refID : int, optional
        Reference ID index. 1 corresponds to tip reference. The default is 0.

    ap_gain : int, optional
        AP band gain. The default is 500.

    lf_gain : int, optional
        LF band gain. The default is 250.

    ap_highpass : bool, optional
        Whether highpass filter is applied on AP band. The default is True.

    output_directory : str, optional
        Where to save generated text file. If None, will be saved in current
        directory.

    Returns
    -------
    tuples : list
        List of n+1 tuples, where n is the number f channels on the probe. 0-th 
        tuple specifies probe-type and number of channels, while each other tuple 
        specifies the configuration for a single channel. Again see 
        https://billkarsh.github.io/SpikeGLX/help/imroTables/ for detail on 
        how each tuple is formatted. 

    """ 
    
    indices = np.arange(n) # Array of all channel indices
    ref_array = refID*np.ones(n, dtype=int)
    apgain_array = ap_gain*np.ones(n, dtype=int)
    lfgain_array = lf_gain*np.ones(n, dtype=int)
    highpass_array = ap_highpass*np.ones(n, dtype=int)    
    
    # If length is 'short', then all channels are in the same bank:
    if length == 'short':
        banks = short_bank*np.ones(n)
        parity = None
        additional_info = '_bank{}'.format(str(short_bank))
        
    # If length is 'long', then several different options for how to distribute
    # channels across different banks:
    elif length == 'long':
        
        banks = np.zeros(n)
        additional_info = '_' + parity
        
        # If all sites within a bank should be on same side of probe:
        if parity == 'columnar':
            banks[np.arange(0,n,2)+1] = 1
        
        elif parity == 'alternating_sides':
            banks[np.arange(0,n,4)+1] = 1
            banks[np.arange(0,n,4)+2] = 1
            
    banks = banks.astype(int)
    
    ch_tuples = list(zip(indices, banks, ref_array, apgain_array, lfgain_array, highpass_array))
    tuples = [(typ, n)] + ch_tuples
    
    # Define current output directory as default if necessary:
    if output_directory == None:
        output_directory = os.getcwd()

    # Create output directory if necessary:
    if not os.path.exists(output_directory):
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
    fname = '{}{}_ref{}'.format(length, additional_info, refID)
    fpath = os.path.join(output_directory, fname)
    
    # Raise warning if overwriting existing file:
    if os.path.exists(fpath):
        response = input('Warning: text file {} already exists. Enter ''y'' to overwrite, any other key to abort.\n'.format(fpath))
        if not response.lower() == 'y':
            print('Aborting execution.\n')
            return
        
    # Write file:
    f = open(fpath, 'w')
    for t in tuples:
        f.write(str(t))
    f.write('\n')
    f.close()
    
    return tuples

    

def get_site_coords(base_data_path, monkey, date, config='short', spacing=15, tip_length=175):
    """
    Compute coordinates of neuropixels probe recording site. 

    Parameters
    ----------

    Returns
    -------

    """
    
    spacing = spacing/1000 # convert to mm
    
    # Get 0-coordinates:
    zero_coords = get_coords_sess(base_data_path, monkey, date)
    depth_adjusted = (zero_coords.depth - tip_length)/1000 # Adjust for probe tip length
    
    # Compute unit vector pointing in direction of probe in ML-DV plane: 
    ehat = np.expand_dims(np.array([0,-1,0]), axis=1) # AP, ML, DV
    
    # Convert degrees to radians, angle rel. horiz.        
    angle_vert =  90 - zero_coords.Ang # Convert to relative to horizontal
    angle_vert = 2*np.pi * angle_vert/360  # Convert degrees to radians
    angle_horiz = 2*np.pi * zero_coords.HAng/360 # Convert degrees to radians
    
    # Define rotation in ML-DV plane
    Rvert = [[1,0,0],
         [0, np.cos(angle_vert), -np.sin(angle_vert)],
         [0, np.sin(angle_vert), np.cos(angle_vert)]]
    
    # Define rotation in AP-ML plane
    Rhoriz = [[np.cos(angle_horiz),-np.sin(angle_horiz),0],
     [np.sin(angle_horiz), np.cos(angle_horiz), 0],
    [0, 0, 1]]

    # Apply rotations:    
    bhat = np.matmul(Rhoriz, ehat)
    bhat = np.matmul(Rvert, ehat) # Unit vector pointing in direction of probe
    
    # Apply scale and offset:
    offset = np.expand_dims(np.array([zero_coords.AP, zero_coords.ML, zero_coords.DV]), axis=1)
    distal_coords =  offset + bhat * depth_adjusted 
    
    # Get IMRO table:
    glx_meta_path = get_sess_metadata_path(base_data_path, monkey, date)
    if 'win' in sys.platform:
        glx_meta_path = '\\\\?\\' + glx_meta_path
    imro_tbl = extract_imro_table(glx_meta_path)
    
    # Get number of channels:
    n_chans = imro_tbl[0][1]
    Chs = np.arange(n_chans)  
    
    # Get bank assignments for each channel:
    Banks = np.array([x[1] for x in imro_tbl[1:]])
    
    # Compute distance of each recording site from (adjusted) tip:
    bank_length = n_chans/2*spacing 
    Distances = bank_length*Banks + spacing*(Chs - Chs % 2)# < Array of recording site distances from (adjusted) tip

    # Compute 3D coordinates of each recoding site:
    B = repmat(bhat.T, n_chans, 1)
    D = np.transpose(repmat(Distances, 3, 1))
    F = repmat(distal_coords.T, n_chans, 1)
    Coords = F - np.multiply(D, B)
    
    # Save as pandas dataframe:
    coords_df = pd.DataFrame(columns=['channel', 'ap', 'dv', 'ml', 'depth'])
    coords_df['channel'] = Chs
    coords_df['ap'] = Coords[:,0]
    coords_df['dv'] = Coords[:,1]
    coords_df['ml'] = Coords[:,2]    
    coords_df['depth'] = depth_adjusted - D
    
    return coords_df



def extract_imro_table(metadata_path):
    """
    Extract IMRO table specifying which channels are active on which banks from
    ephys recording metadata file.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file.

    Returns
    -------
    tuples : list
        List of tuples. The 0-th tuple is a pair describing the probe-type and
        overall number of channels. Each subsequent tuple describes the 
        configuration of a single channel, although the specific format depends
        on the probe-type. See https://billkarsh.github.io/SpikeGLX/help/imroTables/ 
        for detail. 

    """
    
    f = open(metadata_path,'r')
    lines = f.readlines()
    f.close()
    
    # Find lines defining IMRO table:
    imroTbl_lines = [x for x in lines if 'imroTbl' in x]
    
    # Raise error if number of lines enoding IMRO table is not exactly equal to 1:
    if len(imroTbl_lines) == 0:
        raise AssertionError('No IMRO table detected in input file {}.'.format(metadata_path))
    elif len(imroTbl_lines) > 1:
        raise AssertionError('More than one IMRO table detected in input file {}.'.format(metadata_path))
        
    # Parse line defining IMRO table into a list of strings, each encoding a tuple
    imroTbl_str = imroTbl_lines[0]
    r = re.search('\(', imroTbl_str).start() # Find index of first open parentheses
    imroTbl_str = imroTbl_str[r:] # Retain first open parens and everything after
    tuple_strs = imroTbl_str.split('(')
    tuple_strs = tuple_strs[1:] # Cut off extraneous empty string at beginning
    tuples = []
    
    # Convert strings to tuples for individual channels: 
    for tx, t in enumerate(tuple_strs):
        
        # Chop off final end parens and anything afterwards (e.g. newline):
        tail_start = re.search('\)', t).start()
        head = t[0:tail_start]
        
        if tx == 0:
            delimiter = ','
        else: 
            delimiter = ' '        
        
        # Split into strings encoding individual numbers and convert to int:
        nums = head.split(delimiter)
        nums = [int(x) for x in nums]
        
        curr_tuple = tuple(nums)
        tuples.append(curr_tuple)
        
    return tuples



def get_sess_metadata_path(base_data_path, monkey, date):
    
    data_path = get_recording_path(base_data_path, monkey, date, depth = 4)[0]
    fnames = os.listdir(data_path)
    metafiles = [x for x in fnames if '.ap.meta' in x]
    mfile = metafiles[0]
    session_metadata_path = os.path.join(data_path, mfile)
    return session_metadata_path



def partition_adjacent_channels(coords_df, bin_size=4):
    
    # Sort sites by depth:
    coords_df = coords_df.sort_values('depth', ascending=False) # ch0 is the deepest, so sorting by depth with ascending=True would reverse channel order, which we don't want
    
    # Define bin edge indices:
    n_chan = len(coords_df.channel)
    inds = np.arange(0, n_chan+bin_size, bin_size)
    
    # Get channel ranges for each bin:
    channels = np.array(coords_df.channel)
    binned_ch_inds = [channels[inds[x]:inds[x+1]] for x in np.arange(len(inds)-1)]
    
    return binned_ch_inds


def get_session_chs(date, region=None):
    
    # Hack-y, relies on extremely hack-y date_2_chs() function; try to find a 
    # better way of doing this eventually
    
    if region == 'all':
        return np.arange(384)
    
    session_dict = date_2_chs(date)
    if session_dict is not None:
        
        channels = []
        regions = list(session_dict.keys())
        
        if region is None:
            for region in regions:
                
                if session_dict[region] is not None:
                    start_ch = session_dict[region][0]
                    stop_ch = session_dict[region][1]
                    curr_area_channels = list(np.arange(start_ch, stop_ch))
                    channels = channels + curr_area_channels
                else:   
                    continue
                
        # If requesting a particular area:               
        else:
            if session_dict[region] is not None:
                start_ch = session_dict[region][0]
                stop_ch = session_dict[region][1]            
                channels = list(np.arange(start_ch, stop_ch))
            else:
                return None
    else:
        return None
    
    # Sort and find unique:
    channels = list(np.sort(np.unique(channels)))
    
    return channels



def date_2_chs(date):
    
    # Ultra-hacky; map dates to good channel ranges; find a better way of dealing
    # with this in the future:
        
    ch_lookup = {
        
        '20230914' : {
            'IT': [280, 383],
            'WM' : [0, 279],
            'HC' : None,
            'PH' : None
            },
        
        '20231011' : {
            'IT': None,
            'WM' : [0, 299],
            'HC' : None,
            'PH' : [300, 384]
            },

        '20231102' : {
            'IT': [160, 383],
            'WM' : None,
            'HC' : None,
            'PH' : [0, 159]
            },
        
        '20231109' : {
            'IT': None,
            'WM' : [250, 383],
            'HC' : None,
            'PH' : [0, 249]
            },        

        '20231207' : {
            'IT': [75, 384],
            'WM' : None,
            'HC' : None,
            'PH' : [0, 74]
            },                

        '20231211' : {
            'IT': [185, 383],
            'WM' : None,
            'HC' : None,
            'PH' : [0, 184]
            },                 

        '20240110' : {
            'IT': [285, 383],
            'WM' : [220, 285],
            'HC' : [0, 220],
            'PH' : None
            },            

        '20240116' : {
            'IT': [295, 383],
            'WM' : [120, 199],
            'HC' : [120, 240],
            'PH' : [0, 70]
            },                    
        
        '20240123' : {
            'IT': [345, 383],
            'WM' : [185, 314],
            'HC' : [110, 314],
            'PH' : [0, 70]
            },           
        
        '20240124' : {
            'IT': [250, 384],
            'WM' : [200, 250],
            'HC' : [75, 200],
            'PH' : [0, 35]
            },           
        
        '20240130' : {
            'IT': [250, 383],
            'WM' : [40, 164],
            'HC' : [75, 210],
            'PH' : [0, 50]
            },                    

        '20240124' : {
            'IT': [250, 383],
            'WM' : [0, 159],
            'HC' : [75, 200],
            'PH' : [0, 35]
            },            

        '20240202' : {
            'IT': [22, 348],
            'WM' : [190, 220],
            'HC' : [80, 190],
            'PH' : None
            },    

        '20240207' : {
            'IT': [280, 383],
            'WM' : [230, 260],
            'HC' : [129, 235],
            'PH' : [0, 119]
            },           

        '20240208' : {
            'IT': [265, 384],
            'WM' : [75, 199],
            'HC' : [130, 230],
            'PH' : [0, 129]
            }, 

        '20240307' : {
            'IT': [250, 330],
            'WM' : [0, 70],
            'HC' : None,
            'PH' : [70, 250]
            }, 

        '20240408' : {
            'IT': [285, 350],
            'WM' : [180, 285],
            'HC' : [0, 180],
            'PH' : None
            }, 

        '20240409' : {
            'IT': [320, 383],
            'WM' : [290, 320],
            'HC' : [0, 290],
            'PH' : None
            }, 

        '20240410' : {
            'IT': [285, 330],
            'WM' : [230, 285],
            'HC' : None,
            'PH' : [0, 60]
            }, 

        '20240412' : {
            'IT': [240, 270],
            'WM' : [150, 240],
            'HC' : None,
            'PH' : [0, 150]
            }, 

        '20240417' : {
            'IT': [275, 300],
            'WM' : [195, 275],
            'HC' : None,
            'PH' : [0, 190]
            },         
        
        '20240418' : {
            'IT': None,
            'WM' : None,
            'HC' : None,
            'PH' : None
            },     
        
        '20240607' : {
            'IT': [220, 360],
            'WM' : [190, 220],
            'HC' : [65, 190],
            'PH' : [0, 65]
            },     

        '20240718' : {
            'IT': [190, 383],
            'WM' : None,
            'HC' : None,
            'PH' : [0, 190]
            },     

        '20240812' : {
            'IT': [260, 383],
            'WM' : [180, 260],
            'HC' : None,
            'PH' : [135, 180]
            },     
        
        '20240816' : {
            'IT': [175, 320],
            'WM' : [14, 175],
            'HC' : None,
            'PH' : None
            },     
        
        }
    
    if date not in ch_lookup.keys():
        output = None
    else:
        output = ch_lookup[date] 
    
    return output

