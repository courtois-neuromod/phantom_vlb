#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_lazyloading
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1  # ntasks = 4 with srun launch command, 1 with torchrun launch command! (torchrun handles assignment of tasks to GPUs)
#SBATCH --cpus-per-task=32  # use 32 for torchrun, 32 / num GPUs = 8 with srun
#SBATCH --gpus-per-node=v100:4
#SBATCH --gpus-per-task=1
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

TEMP_PATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/temp_files/friends_${SUBNUM}_*_llFile.h5"
rsync -tv --info=progress2 $TEMP_PATH $SLURM_TMPDIR/

#srun python -m train_dev_tests experiment=VLB_vllama2_friends subject=$SUBNUM

torchrun --nproc_per_node=4 train_dev_tests.py experiment=VLB_vllama2_friends subject=$SUBNUM   # nproc_per_node must match number GPUs
