#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_lazyloading
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.12.4
module load opencv/4.11.0

# cactivate project's virtual env
source /home/mstlaure/links/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_rq_venv/bin/activate

SCRATCH_PATH="/home/mstlaure/links/scratch/phantom_lazyload/temp_files"

SUBNUM="sub-${1}"
SEASON="s${2}"

python videollama2_vlb_lazyloading.py \
    --features_path "${SCRATCH_PATH}/friends_${SEASON}_features.h5" \
    --timeseries_path "${SCRATCH_PATH}/${SUBNUM}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_desc-1000Parcels7Networks_timeseries.h5" \
    --lazyload_path ${SCRATCH_PATH} \
    --subject ${SUBNUM} \
    --season ${SEASON} \
    --n_split 4 \
    --delay 3 \
    --window 3 \
