#!/bin/bash
#SBATCH --job-name=example   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dk2643@columbia.edu         # Where to send mail (e.g. uni123@columbi$
#SBATCH --time=0-10:05:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.log    # Standard output and error log
#SBATCH --array=0       # array range
#SBATCH --mem=30gb
#SBATCH -c 10    #number of cpu cores

ls /mnt/smb/locker/issa-locker/
monkey=West
date=20240321
python allchan_scenefile.py $SLURM_ARRAY_TASK_ID $monkey $date
python allchan_bl.py $SLURM_ARRAY_TASK_ID $monkey $date
python allchan_wf.py $SLURM_ARRAY_TASK_ID $monkey $date
python allchan_meanpsth.py $SLURM_ARRAY_TASK_ID $monkey $date
python allchan_objaverse.py $SLURM_ARRAY_TASK_ID $monkey $date
