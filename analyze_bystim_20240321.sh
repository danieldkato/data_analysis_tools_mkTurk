#!/bin/bash
#SBATCH --job-name=example   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yj2278@columbia.edu         # Where to send mail (e.g. uni123@columbia.edu)
#SBATCH --time=0-10:05:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.log    # Standard output and error log
#SBATCH --array=0-383        # array range
#SBATCH -c 10    #number of cpu cores
#SBATCH --mem=50gb

ls /mnt/smb/locker/issa-locker/
monkey=West
date=20240321
python analyze_bystim.py $SLURM_ARRAY_TASK_ID $monkey $date

