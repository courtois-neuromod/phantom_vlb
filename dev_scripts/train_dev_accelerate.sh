#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_debugging
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # ntasks = 4 with srun launch command, 1 with torchrun launch command! (torchrun handles assignment of tasks to GPUs)
#SBATCH --cpus-per-task=40  # use 32 for torchrun, 32 / num GPUs = 8 with srun
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.10.13
module load httpproxy  # comet can be used online from beluga after loading this module

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_venv/bin/activate

#export GPUS_PER_NODE=4
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export TRANSFORMERS_OFFLINE="1"
export HF_HOME="./models"
export TRANSFORMERS_CACHE="./models"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TEMP_PATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/temp_files/friends_${SUBNUM}_*_llFile_db.h5"
rsync -tv --info=progress2 $TEMP_PATH $SLURM_TMPDIR/


accelerate launch --config_file="fsdp.yaml" --mixed_precision="fp16" --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID --num_processes=4 train_dev_accelerate.py --api_key dATPitXdUDl3JK6YKv1TAkrgS --workspace mariestlaurent
