
import socket 
import sys
import pathlib as Path 
import os
from utils import * 

host = socket.gethostname()
if 'rc.zi.columbia.edu' in host: 
    engram_path = Path.Path('/mnt/smb/locker/issa-locker')
elif 'DESKTOP' in host:
    engram_path = Path.Path('Z:/')
elif 'Younah' in host:
    engram_path = Path.Path('/Volumes/issa-locker/')

base_data_path = engram_path  / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

base_data_path = engram_path / 'Data'
base_save_out_path = engram_path / 'users/Younah/ephys'

monkey = sys.argv[1]
date = sys.argv[2]

# make all the bashfiles
shell_script_name = 'analyze_bystim.sbatch'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)


"""
shell_script_name = 'analyze_bystim_0_100.sbatch'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)

shell_script_name = 'analyze_bystim_100_200.sbatch'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)

shell_script_name = 'analyze_bystim_200_300.sbatch'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)

shell_script_name = 'analyze_bystim_300_383.sbatch'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)
"""

shell_script_name = 'template_chan.sh'
script_name = '' 
make_bashfiles(shell_script_name, script_name, monkey, date)

shell_script_name = 'template_all.sh'
script_name = ''
make_bashfiles(shell_script_name, script_name, monkey, date)
