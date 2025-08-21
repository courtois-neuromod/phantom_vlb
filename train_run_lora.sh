#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_baseline
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.12.4
module load opencv/4.11.0
module load httpproxy  # comet can be used online from the clusters after loading this module

# activate project's virtual env
source /home/mstlaure/links/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_rq_venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export TRANSFORMERS_OFFLINE="1"
export HF_HOME="./models"
export TRANSFORMERS_CACHE="./models"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SUBNUM="sub-${1}"

# access data from my scratch instead of copying it to the compute node (takes hours even with lazy loading...)
export SCRATCH_PATH="/home/mstlaure/links/scratch/phantom_lazyload/temp_files"

python train.py experiment=VLB_vllama2_friends_lora subject=$SUBNUM
