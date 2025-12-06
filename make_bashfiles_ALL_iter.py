
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

#monkey = sys.argv[1]
#date = sys.argv[2]


#"""
monkey = 'West'
dates = [#'20230927',
# '20231001',
# '20231002',
# '20231009',
# '20231010',
# '20231011',
# '20231012',
# '20231013',
# '20231016', # Exception; includes two recording sessions
# '20231109',
# '20231110',
 '20231113',
 '20231114',
 '20231211',
 '20231212',
 '20231213',
 '20231214',
 '20231219',
 '20240116',
 '20240117',
 '20240118',
 '20240119',
 '20240124',
 '20240125',
 '20240126',
 '20240129',
 '20240208',
 '20240209',
 '20240212',
 '20240214',
 '20240215',
 '20240216',
 '20240217',
 '20240219',
 '20240321',
 '20240724',
# '20240725',
# '20240726',
# '20240726',
# '20240731'
]
#"""

"""
monkey = 'Bourgeois'
dates = [
 '20251030'
]
"""


for date in dates:


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
