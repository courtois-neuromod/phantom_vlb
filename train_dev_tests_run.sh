#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_lazyloading
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.10.13
module load httpproxy  # comet can be used online from beluga after loading this module

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SUBNUM="sub-${1}"
#FEATURES_PATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/results/videollama2/lazyloading/friends/friends_*_features.h5"
#TIMESERIES_PATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/friends/${SUBNUM}/func/${SUBNUM}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_desc-1000Parcels7Networks_timeseries.h5"

#rsync -tv --info=progress2 $FEATURES_PATH $SLURM_TMPDIR/
#rsync -tv --info=progress2 $TIMESERIES_PATH $SLURM_TMPDIR/

TEMP_PATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/temp_files/friends_{$SUBNUM}_*_llFile.h5"
rsync -tv --info=progress2 $TEMP_PATH $SLURM_TMPDIR/

python -m train_dev_tests experiment=VLB_vllama2_friends subject=$SUBNUM
