#!/bin/bash
#SBATCH --job-name=genPSTH
#SBATCH -c 4
#SBATCH --time=0-18:00:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --output=/home/dk2643/genPSTH_%A.log

source ~/miniconda3/etc/profile.d/conda.sh  # TODO: change to your own conda executable path (same in run_dartsort.sh)
conda activate /share/issa/users/yy3658/shared_env/data_processing  # should keep constant

MONKEY="Bourgeois"
DATE="20260217"

SCRIPT_DIR="/home/yy3658/helpers/data_analysis_tools_mkTurk"  # TODO: change to your path to data_analysis_tools_mkTurk

echo "=== Starting Preprocessing Pipeline ==="
echo "Monkey: $MONKEY, Date: $DATE"

# Submit dartsort job with parameters (GPU-intensive, runs in parallel)
echo "Submitting DARTsort job..."
sbatch --export=ALL,MONKEY="$MONKEY",DATE="$DATE",SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_dartsort.sh"

echo "Running MUA extraction..."
python "$SCRIPT_DIR/get_MUA.py" --monkey "$MONKEY" --date "$DATE"

echo "Generating PSTH files..."
python "$SCRIPT_DIR/get_data_dict_from_mkturk.py" --monkey "$MONKEY" --date "$DATE"

echo "Running data preprocessing pipeline..."
python /mnt/smb/locker/issa-locker/users/Dan/code/scripts/preprocessing/generate_psth_files.py --monkey "$MONKEY" --date "$DATE"  # can change to process_session_pipeline.py (optional)

echo "=== Preprocessing Pipeline Completed ==="