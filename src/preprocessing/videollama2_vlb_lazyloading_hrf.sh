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

# cactivate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/vllama2_venv/bin/activate

DATA_DIR="../../../algonauts_dset/competitors"
SEG_DIR="../../friends_annotations/annotation_results/manual_segmentation"
OUTP_DIR="../../results/videollama2/lazyloading/friends"
SEASON="${1}"

# --num_frames 8 \
python videollama2_vlb_lazyloading_hrf.py \
    --input_transcript_path "${DATA_DIR}/stimuli/transcripts/friends/${SEASON}" \
    --input_seg_path "${SEG_DIR}/${SEASON}" \
    --input_video_path "../../../friends_algonauts/data/friends.stimuli/${SEASON}" \
    --lazy_load_path "${OUTP_DIR}/friends_${SEASON}_features.h5" \
    --cache_dir "../../models" \
    --model_type videollama2 \
    --model_path DAMO-NLP-SG/VideoLLaMA2-7B \
    --model_max_length 2048 \
    --bf16 False \
    --frames_per_tr 4 \
    --tr 1.49 \
    --window_duration 3 \
