import socket 
from pathlib import Path

host = socket.gethostname()
if 'rc.zi.columbia.edu' in host: 
    ENGRAM_PATH = Path('/mnt/smb/locker/issa-locker')
elif 'DESKTOP' in host:
    ENGRAM_PATH = Path('Z:/')
elif 'Younah' in host:
    ENGRAM_PATH = Path('/Volumes/issa-locker/')
else:
    raise ValueError(f"Unknown host: {host}")