
import socket 
import sys
import pathlib as Path 
import os

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

monkey = sys.argv[3]
date = sys.argv[4]

shell_script_name = sys.argv[1]
script_name = sys.argv[2]
script_dir = sys.argv[5]

if script_dir is not None:
     shell_script_name = os.path.join(script_dir, shell_script_name)

with open(shell_script_name,'r') as f:
     template = f.read()

new_script = template.replace('date=', f'date={date}')
new_script = new_script.replace('monkey=', f'monkey={monkey}')

if 'py' in script_name:
    f_name = script_name.split('.')[0]
    new_script = new_script.replace('func', f'{script_name}')

else:
    f_name = shell_script_name.split('.')[0]

foldername = 'bashfiles_' + date
os.makedirs(foldername, exist_ok = True)
file_name = f'bashfiles_{date}/{f_name}_{date}.sh'

print(file_name)
with open(file_name, 'w') as rsh:      
    rsh.write(new_script)
