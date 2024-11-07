#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=videollama2_lazyloading
#SBATCH --output=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/phantom_vlb/slurm_files/slurm-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load python/3.10.13

# cactivate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_venv/bin/activate

DATA_DIR="../../algonauts_dset/competitors"
OUTP_DIR="../results/videollama2/lazyloading/friends"
SEASON = "${1}"

# --num_frames 8 \
python preprocessing.py \
    --input_transcript_path "${DATA_DIR}/stimuli/transcripts/friends/${SEASON}" \
    --input_video_path "../../friends_algonauts/data/friends.stimuli/${SEASON}" \
    --lazy_load_path "${OUTP_DIR}/friends_${SEASON}_features.h5" \
    --model_type videollama2 \
    --model_path mistralai/Mistral-7B-Instruct-v0.2 \
    --model_max_length 2048 \
    --window_max_length 2036 \
    --multimodal_token_index -201 \
    --vision_tower models/vision_tower/openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --bf16 True \
    --device 0 \
    --image_aspect_ratio pad \
    --frames_per_tr 4 \
    --tr 1.49 \
    --bits 16 \
    --window_duration 3 \
