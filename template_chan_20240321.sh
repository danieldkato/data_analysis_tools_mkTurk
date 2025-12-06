#!/bin/bash
#SBATCH --job-name=example   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yj2278@columbia.edu         # Where to send mail (e.g. uni123@columbi$
#SBATCH --time=0-10:05:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.log    # Standard output and error log
#SBATCH --array=0-383        # array range number of channels in neuropixel 
#SBATCH --mem=50gb           # or some high number so that you don't run out of memory 
#SBATCH -c 10    #number of cpu cores 

ls /mnt/smb/locker/issa-locker/
monkey=West
date=20240321
python get_psth_objaverse.py $SLURM_ARRAY_TASK_ID $monkey $date
python get_wf_features.py $SLURM_ARRAY_TASK_ID $monkey $date
