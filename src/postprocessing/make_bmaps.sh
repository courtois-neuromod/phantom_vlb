#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_make_brainmaps
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.12.4
module load opencv/4.11.0

# cactivate project's virtual env
source /home/mstlaure/links/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_rq_venv/bin/activate

IN_PATH="/home/mstlaure/links/projects/rrg-pbellec/mstlaure/phantom_vlb/results/videollama2/brain_finetune/friends/lightning_ckpt/debug/vllama2_vlb_friends_debug_lora_sub-01/version_6"
SCRATCH_PATH="/home/mstlaure/links/scratch/phantom_lazyload"
ATLAS_PATH="${SCRATCH_PATH}/temp_files/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_desc-1000Parcels7Networks_dseg.nii.gz"
OUT_PATH="${SCRATCH_PATH}/res_files/vllama2_vlb_friends_debug_lora_sub-01_version_6"

python make_acc_brainmaps.py \
    --metrics_path ${IN_PATH} \
    --atlas_path ${ATLAS_PATH} \
    --out_path ${OUT_PATH} \
