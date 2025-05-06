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

    

def h5_2_site_coords_df(h5path):
    zero_coords = pd.read_hdf(h5path, 'zero_coordinates')
    imro_tbl = pd.read_hdf(h5path, 'imro_table')
    site_coords_df = get_site_coords(zero_coords, imro_tbl, spacing=15, tip_length=175)
    return site_coords_df



def get_site_coords(zero_coords, imro_tbl, spacing=15, tip_length=175):
    """
    Compute coordinates of neuropixels probe recording site. 

    Parameters
    ----------

    Returns
    -------

    """
    
    spacing = spacing/1000 # convert to mm
    
    # Get 0-coordinates:
    if np.isnan(zero_coords.HAng):
        zero_coords.HAng = 0
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
    bhat = np.matmul(Rvert, bhat) # Unit vector pointing in direction of probe
    
    # Apply scale and offset:
    offset = np.expand_dims(np.array([zero_coords.AP, zero_coords.ML, zero_coords.DV]), axis=1)
    distal_coords =  offset + bhat * depth_adjusted 
    
    # Get number of channels:
    n_chans = imro_tbl.shape[0]
    Chs = np.array(imro_tbl.index)  
    
    # Get bank assignments for each channel:
    Banks = imro_tbl.bank
    
    # Compute distance of each recording site from (adjusted) tip:
    bank_length = n_chans/2*spacing 
    Distances = bank_length*Banks + spacing*((Chs - Chs % 2)/2) # < Array of recording site distances from (adjusted) tip

    # Compute 3D coordinates of each recoding site:
    B = repmat(bhat.T, n_chans, 1)
    D = np.transpose(repmat(Distances, 3, 1))
    F = repmat(distal_coords.T, n_chans, 1)
    Coords = F - np.multiply(D, B)
    
    # Save as pandas dataframe:
    coords_df = pd.DataFrame(columns=['ch_idx_glx', 'ap', 'dv', 'ml', 'depth'])
    coords_df['ch_idx_glx'] = Chs
    coords_df['ap'] = Coords[:,0]
    coords_df['dv'] = Coords[:,1]
    coords_df['ml'] = Coords[:,2]    
    coords_df['depth'] = depth_adjusted - D
    
    # Add channel index by depth:
    coords_df = coords_df.sort_values(by=['depth', 'ch_idx_glx'], ascending=[False, True])
    coords_df['ch_idx_depth'] = np.arange(coords_df.shape[0])
    coords_df = coords_df[['ch_idx_glx', 'ch_idx_depth', 'ap', 'dv', 'ml', 'depth']]
    
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
        
    # Convert list of tuples to dataframe:
    imro_tbl = pd.DataFrame()
    if tuples[0][0] == 0:
        fields = ['channel', 'bank', 'ref_id', 'ap_gain', 'lf_gain', 'ap_hipass']
        
    for i, field in enumerate(fields):
        imro_tbl[field] = [x[i] for x in tuples[1:]]
        
    imro_tbl.index = imro_tbl.channel
    imro_tbl = imro_tbl.drop(columns=['channel'])
        
    return imro_tbl



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



def add_chs_by_area(df_in):
    
    # Assign some defaults:
    if 'area' not in df_in.columns:
        df_in['area'] = 'all'
    
    # Iterate over sessions x areas:
    df_out = pd.DataFrame()
    areas = df_in[['monkey', 'date', 'area']].drop_duplicates()
    for aidx, area in areas.iterrows():
        monkey = area.monkey
        date = area.date
        area = area.area
        curr_chs = session_2_chs(monkey, date, area)
        if curr_chs is None:
            curr_chs = []
        curr_chs_df = pd.DataFrame(columns=['monkey', 'date', 'area', 'ch_idx_depth'], index=np.arange(len(curr_chs)))
        curr_chs_df['monkey'] = monkey
        curr_chs_df['date'] = date
        curr_chs_df['area'] = area
        curr_chs_df['ch_idx_depth'] = curr_chs
        df_out = pd.concat([df_out, curr_chs_df], axis=0)
            
    return df_out



def session_2_chs(monkey, date=None, area=None):
    
    # Ultra-hacky; map dates to good channel ranges; find a better way of dealing
    # with this in the future:
    
    ch_lookup = {
        
        'West' : {
            
            '20230914' : {
                'IT': np.arange(280, 383),
                'WM' : np.arange(0, 279),
                'HC' : None,
                'PH' : None,
                'all' : np.arange(384)
                },
            
            '20231011' : {
                'IT': None,
                'WM' : np.arange(0, 299),
                'HC' : None,
                'PH' : np.arange(300, 384),
                'all' : np.arange(384)
                },

            '20231102' : {
                'IT': np.arange(160, 383),
                'WM' : None,
                'HC' : None,
                'PH' : np.arange(0, 159),
                'all' : np.arange(384)
                },
            
            '20231109' : {
                'IT': None,
                'WM' : np.arange(250, 383),
                'HC' : None,
                'PH' : np.arange(0, 249),
                'all' : np.arange(384)
                },        

            '20231207' : {
                'IT': np.arange(75, 384),
                'WM' : None,
                'HC' : None,
                'PH' : np.arange(0, 74),
                'all' : np.arange(384)
                },                

            '20231211' : {
                'IT': np.arange(185, 383),
                'WM' : None,
                'HC' : None,
                'PH' : np.arange(0, 184),
                'all' : np.arange(384)
                },                 

            '20240110' : {
                'IT': np.arange(285, 383),
                'WM' : np.arange(220, 285),
                'HC' : np.arange(0, 220),
                'PH' : None,
                'all' : np.arange(384)
                },            

            '20240116' : {
                'IT': np.arange(295, 383),
                'WM' : np.arange(120, 199),
                'HC' : np.arange(120, 240),
                'PH' : np.arange(0, 70),
                'all' : np.arange(384)
                },                    
            
            '20240123' : {
                'IT': np.arange(345, 383),
                'WM' : np.arange(185, 314),
                'HC' : np.arange(110, 314),
                'PH' : np.arange(0, 70),
                'all' : np.arange(384)
                },           
            
            '20240124' : {
                'IT': np.arange(250, 384),
                'WM' : np.arange(200, 250),
                'HC' : np.arange(75, 200),
                'PH' : np.arange(0, 35),
                'all' : np.arange(384)
                },           
            
            '20240130' : {
                'IT': np.arange(250, 383),
                'WM' : np.arange(40, 164),
                'HC' : np.arange(75, 210),
                'PH' : np.arange(0, 50),
                'all' : np.arange(384)
                },                    

            '20240124' : {
                'IT': np.arange(250, 383),
                'WM' : np.arange(0, 159),
                'HC' : np.arange(75, 200),
                'PH' : np.arange(0, 35),
                'all' : np.arange(384)
                },            

            '20240202' : {
                'IT': np.arange(22, 348),
                'WM' : np.arange(190, 220),
                'HC' : np.arange(80, 190),
                'PH' : None,
                'all' : np.arange(384)
                },    

            '20240207' : {
                'IT': np.arange(280, 383),
                'WM' : np.arange(230, 260),
                'HC' : np.arange(129, 235),
                'PH' : np.arange(0, 119),
                'all' : np.arange(384)
                },           

            '20240208' : {
                'IT': np.arange(265, 384),
                'WM' : np.arange(75, 199),
                'HC' : np.arange(130, 230),
                'PH' : np.arange(0, 129),
                'all' : np.arange(384)
                }, 

            '20240307' : {
                'IT': np.arange(250, 330),
                'WM' : np.arange(0, 70),
                'HC' : None,
                'PH' : np.arange(70, 250),
                'all' : np.arange(384)
                }, 

            '20240408' : {
                'IT': np.arange(285, 350),
                'WM' : np.arange(180, 285),
                'HC' : np.arange(0, 180),
                'PH' : None,
                'all' : np.arange(384)
                }, 

            '20240409' : {
                'IT': np.arange(320, 383),
                'WM' : np.arange(290, 320),
                'HC' : np.arange(0, 290),
                'PH' : None,
                'all' : np.arange(384)
                }, 

            '20240410' : {
                'IT': np.arange(285, 330),
                'WM' : np.arange(230, 285),
                'HC' : None,
                'PH' : np.arange(0, 60),
                'all' : np.arange(384)
                }, 

            '20240412' : {
                'IT': np.arange(240, 270),
                'WM' : np.arange(150, 240),
                'HC' : None,
                'PH' : np.arange(0, 150),
                'all' : np.arange(384)
                }, 

            '20240417' : {
                'IT': np.arange(275, 300),
                'WM' : np.arange(195, 275),
                'HC' : None,
                'PH' : np.arange(0, 190),
                'all' : np.arange(384)
                },         
            
            '20240418' : {
                'IT': None,
                'WM' : None,
                'HC' : None,
                'PH' : None,
                'all' : np.arange(384)
                },     
            
            '20240607' : {
                'IT': np.arange(220, 360),
                'WM' : np.arange(190, 220),
                'HC' : np.arange(65, 190),
                'PH' : np.arange(0, 65),
                'all' : np.arange(384)
                },     

            '20240718' : {
                'IT': np.arange(190, 383),
                'WM' : None,
                'HC' : None,
                'PH' : np.arange(0, 190),
                'all' : np.arange(384)
                },     

            '20240812' : {
                'IT': np.arange(260, 383),
                'WM' : np.arange(180, 260),
                'HC' : None,
                'PH' : np.arange(135, 180),
                'all' : np.arange(384)
                },     
            
            '20240816' : {
                'IT': np.arange(175, 320),
                'WM' : np.arange(14, 175),
                'HC' : None,
                'PH' : None,
                'all' : np.arange(384)
                }, 
            
            '20240920' : {
                'all' : np.arange(384),
                'TE0' : np.arange(0,180),
                'TE3' : np.arange(180, 325),
                'IT' : np.arange(0, 325)
                },

            '20240924' : {
                'all' : np.arange(384),
                'TE2' : np.arange(106, 161),
                'TE3' : np.arange(161, 328),
                'PHC' : np.arange(0, 106),
                'IT' : np.arange(106, 328),
                'MT' : np.arange(0,106)
                },

            '20240925' : {
                'all' : np.arange(384),
                'TE2' : np.arange(106, 151),
                'TE3' : np.arange(151, 328),
                'PHC' : np.arange(0, 106),
                'IT' : np.arange(106, 328),
                'MT' : np.arange(0, 106)
                },
            
            '20240927' : {
                'all' : np.arange(384),
                'TE2' : np.arange(106, 151),
                'TE3' : np.arange(151, 304),
                'PHC' : np.arange(0, 106),
                'IT' : np.arange(106, 304),
                'MT' : np.arange(0, 106)
                },
            
            '20241004' : {
                'all' : np.arange(384),
                'TE3' : np.arange(239, 353),
                'WM' : np.arange(166, 239),
                'HC' : np.arange(0, 166),
                'IT' : np.arange(239, 352),
                'MT' : np.arange(0, 165)
                },
            
            '20241010' : {
                'all' : np.arange(384),
                'TE2' : np.arange(131, 215),
                'TE3' : np.arange(215, 304),
                'PHC' : np.arange(84, 131),
                'PRH' : np.arange(0, 84),
                'IT' : np.arange(131, 304),
                'MT' : np.arange(0, 131)
                },
            
            '20241011' : {
                'all' : np.arange(384),
                'TE2' : np.arange(144, 222),
                'TE3' : np.arange(222, 324),
                'PHC' : np.arange(84, 144),
                'PRH' : np.arange(0, 84),
                'IT' : np.arange(144, 324),
                'MT' : np.arange(0, 144)
                }
            },
        
        'Bourgeois' : {

            '20250103' : {
                'IT' : np.arange(0, 263),
                'all' : np.arange(384)
                },
            
            '20250106' : {
                'IT' : np.arange(0, 254),
                'all' : np.arange(384)
                },

            '20250107' : {
                'IT' : np.arange(231, 299),
                'MT' : np.arange(0, 168),                
                'all' : np.arange(384)
                },
            
            '20250110' : {
                'IT' : np.arange(184, 310), 
                'MT' : np.arange(0, 152),
                'all' : np.arange(384) 
                },
            
            '20250310' : {
                'all' : np.arange(384),
                'unclassified' : np.arange(229, 383),
                'TE0' : np.arange(0, 229),
                'IT' : np.arange(0, 229)
                },
            
            '20250311' : {
                'all' : np.arange(384),
                'unclassified' : np.arange(180, 383),
                'TE0' : np.arange(0, 180),
                'IT' : np.arange(0, 180)
                },

            '20250328' : {
                'all' : np.arange(384),
                'unclassified' : np.arange(293, 384),
                'TE3' : np.arange(233, 292),
                'PHC' : np.arange(0, 156),
                'WM' : np.arange(156, 233),
                'IT' : np.arange(233, 292),
                'MT' : np.arange(0, 156)
                },

            '20250331' : {
                'all' : np.arange(384),
                'unclassified' : np.arange(300, 384),
                'TE3' : np.arange(236, 300),
                'PHC' : np.arange(0, 150),
                'WM' : np.arange(151, 235),
                'IT' : np.arange(236, 300),
                'MT' : np.arange(0, 150)
                },
            
            '20250417' : {
                'all' : np.arange(384) 
                },            
            
            '20250418' : {
                'all' : np.arange(384) 
                }
            }
        
        
        }
    
    if monkey not in ch_lookup.keys():
        warnings.warn('Lookup table for requested monkey {} not found.'.format(monkey))
        output = None
    else:
        monkey_dict = ch_lookup[monkey]
        
        # Return channels for specific session if requested:
        if date is None:
            output = monkey_dict
        else:
            if date not in monkey_dict.keys():
                warnings.warn('Lookup table for requested session {}, {} not found.'.format(monkey, date))
                output = None
            else:
                date_dict = monkey_dict[date] 
                
                # Return channels for specific area if requested:
                if area is None:
                    output = date_dict
                else:
                    if area not in date_dict.keys():
                        warnings.warn('Channel range for session {}, {}, area {} not found.'.format(monkey, date, area))
                        output = None
                    else:
                        output = date_dict[area]
    
    return output

