"""
TODO:
Step1: implement prepocessing steps to build .h5 of input data from vision & language input...
Current script

Step 2: implement my own Dataset class w __getitem__ and __len__
TODO: adapt from data module dataset (LazySupervisedDataset)'s __getitem__ to preprocess video chunks per TRs
determine start and duration of video frame window for each target TR
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L387
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L315

Step 3: implement datamodule

# TODO: (down the road) add audio branch to video-llama2
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from decord import VideoReader, cpu
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('../')

from VideoLLaMA2.videollama2.mm_utils import (
    expand2square,
    frame_sample,
    tokenizer_multimodal_token,
)
from VideoLLaMA2.videollama2.model.encoder import CLIPVisionTower

"""
Just here as a reference to help pick videoframe sampling frequency
Values from
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/constants.py#L20
"""
NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1


def get_arguments():
    """
    Params adapted from videollama2 dataclass : https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L71
    Default parameters from : https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45
    """
    parser = argparse.ArgumentParser(
        description="Compile parameters for input feature lazy_loading for VideoLLaMa2"
    )
    parser.add_argument(
        '--input_transcript_path', required=True, type=str, help='Path to the input transcripts data directory.'
    )
    parser.add_argument(
        '--input_video_path', required=True, type=str, help='Path to the input video data directory.'
    )
    parser.add_argument(
        '--lazy_load_path', required=True, type=str, help='Path where to save the processed features in an .h5 output file.'
    )
    parser.add_argument(
        '--model_type', type=str, default='videollama2', choices=['videollama2', 'videollama2_llama', 'videollama2_mistral', 'videollama2_mixtral', 'videollama2_qwen2'], help='Model type selected from list.'
    )
    parser.add_argument(
        '--model_path', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
    )
    parser.add_argument(
        '--model_max_length', type=int, default=2048,
    )
    parser.add_argument(
        '--window_max_length', type=int, default=2036,
    )
    parser.add_argument(
        '--multimodal_token_index', type=int, default=-201,
    )
    parser.add_argument(
        '--vision_tower', type=str,
        #default='openai/clip-vit-large-patch14-336',  # requires internet access
        default='../models/vision_tower/openai/clip-vit-large-patch14-336',  # locally saved copy
    )
    parser.add_argument(
        '--mm_vision_select_layer', type=int, default=-2,
    )
    parser.add_argument(
        '--mm_vision_select_feature', type=str, default='patch',
    )
    parser.add_argument(
        '--bf16', type=bool, default=True,
    )
    parser.add_argument(
        '--device', type=int, default=0,
    )
    parser.add_argument(
        '--image_aspect_ratio', type=str, default='pad',
    )
    parser.add_argument(
        '--frames_per_tr', type=int, default=4,
    )
    parser.add_argument(
        '--num_frames', type=int, default=8,
    )
    parser.add_argument(
        '--tr', type=float, default=1.49,
    )
    parser.add_argument(
        '--bits', type=int, default=16,
    )
    parser.add_argument(
        '--window_duration', type=int, default=3, help='number of TRs worth of video frames included in prediction window',
    )
    return parser.parse_args()


def get_input_paths(ll_args):

    transcript_path = str(Path(ll_args.input_transcript_path).resolve())
    video_path = str(Path(ll_args.input_video_path).resolve())

    input_paths = {}
    for tr_file in sorted(glob.glob(f"{transcript_path}/friends_*.tsv")):
        ep_num: str = os.path.basename(tr_file).split('_')[-1].split('.')[0]
        if Path(f"{video_path}/friends_{ep_num}.mkv").exists():
            input_paths[ep_num] = {
                'transcript': tr_file,
                'video': f'{video_path}/friends_{ep_num}.mkv',
            }
    return input_paths


def get_done_ep(lazyload_path):
    """
    Return list of episodes whose input features are
    already processed and saved
    """
    if not Path(lazyload_path).exists():
        ll_file = h5py.File(lazyload_path, "w")
        epi_list = []
    else:
        ll_file = h5py.File(lazyload_path, "r")
        epi_list = sorted(list(ll_file.keys()))

    ll_file.close()
    return epi_list


def prep_video_processor(ll_args):
    """
    Adapted from:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L510
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/videollama2_arch.py#L43
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/encoder.py#L12

    VideoLLaMA2 has a vision tower, which is a pre-trained encoder from OpenAI's Clip (weights are frozen during fine-tuning)
    The VideoLLaMA2 datamodule has a dataset that, with __getitem__, processes a video through the vision tower (nn.Module's forward)
    before handing it over to the datamodule that gives batches to the main model.

    Here, with lazy-loading, I'm processing each video in advance through the vision tower and storing its features as .h5
    for lazy loading to speed up processing

    From FP's video_transformer, decide on the window of video (here n = 3 TRs) to pair w a given target TR

    Configs are split into data classes (model, training and data) in train script, and specified in finteune script:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/scripts/custom/finetune.sh#L45
    """

    vision_tower = CLIPVisionTower(ll_args.vision_tower, args=ll_args)

    vision_tower.to(dtype=torch.bfloat16 if ll_args.bf16 else torch.float16, device=ll_args.device)

    ll_args.image_size = vision_tower.image_size
    ll_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor
    ll_args.is_multimodal = True

    return ll_args


def prep_tokenizer(ll_args):
    """
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L494
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ll_args.model_path,
        model_max_length=ll_args.model_max_length,
        padding_side="right",
        use_fast=True,
        truncation=False,
        #token=access_token,  # just needed first time
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def prep_text(text, tokenizer, window_max_length):
    """
    Adapted from
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L168
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L289

    videollama2 inference script:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/__init__.py#L32

    constants from
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/constants.py#L28

    As a reference (for video frame input)
    multimodal_token_index = -201
    """

    # truncate text input to fix max language window length (in tokens)
    tokens = tokenizer.tokenize(text.strip())
    if len(tokens) > window_max_length:
        tokens = tokens[-window_max_length:]
    text = tokenizer.convert_tokens_to_string(tokens)

    # Text Prep:
    # pass text through .strip() to remove extra white spaces,
    # add modal token to begining of input text for each entry
    modal_token = "<video>"
    message = [{'role': 'user', 'content': modal_token + '\n' + text.strip()}]

    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    # returns list, not pytorch tensor (otherwise set return_tensors='pt')
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors=None)#.unsqueeze(0).long().cuda()

    return input_ids


def load_video(video_path, tr = 1.49):
    """
    TODO: adapt to break video into chunks of frames assigned to a TR
    Source:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L133
    """
    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    fps = vreader.get_avg_fps()
    num_frames = len(vreader)
    duration: float = num_frames / fps

    # END (in seconds) of input movieframe window for a given TR (from video onset)
    tr_list = np.array(range(1, math.ceil(duration/tr)))*tr

    return vreader, fps, num_frames, duration, tr_list.tolist()


def extract_video_chunk(vreader, processor, end_time, win_dur, fps, num_frames_of_video, tr, frames_per_tr, aspect_ratio):
    """
    Adapted from:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L133
    """
    # Input videoframe window onset time (in seconds); up to n = win_dur (in TRs) before end_tr
    start_time = max(0, end_time - tr*(win_dur))

    # Determine frame range & Calculate frame indices
    f_start = max(int(start_time * fps) - 1, 0)
    f_end   = min(int(end_time * fps) - 1, num_frames_of_video - 1)
    all_frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    num_frames = round((end_time - start_time) / tr) * frames_per_tr
    sampled_frame_indices = [all_frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

    video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    # pad to 12 images w blank ones for TRs at onset of video (can't go back to full window lenght)
    while len(video_data) < (win_dur * frames_per_tr):
        #video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))
        video_data.append(Image.fromarray(np.swapaxes(np.zeros((*video_data[-1].size, 3), dtype=np.uint8), 0, 1)))

    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        #video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        video = np.array(processor.preprocess(images)['pixel_values'])
    else:
        images = [f for f in video_data]
        #video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        video = np.array(processor.preprocess(images)['pixel_values'])
    return video


def make_lazy_loading_videollama2(ll_args):
    """
    Support function to preprocess input data (language and video frames) into
    pre-saved features for lazy loading during multimodal brain alignment
    experiments with video-LLaMA2.

    Script to be ran separately ahead of training / testing / validation,
    for each season of Friends.

    Output is a HDF5 file to be used by a dataloader for lazy loading.
    For each modality, have one row corresponding of features correspond to a single TR

    This script is adapted from
    https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/datasets/replay_datamodule.py#L125

    The video and text data pre-processing steps are adapted from video-LLaMA2
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/train.py
    """
    print(ll_args)


    ll_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output file with lazyloading data
    ll_path = str(Path(f"{ll_args.lazy_load_path}").resolve())
    done_epnum = get_done_ep(ll_path)
    input_file_paths = get_input_paths(ll_args)

    ll_args = prep_video_processor(ll_args)
    tokenizer = prep_tokenizer(ll_args)

    print(input_file_paths)

    for ep_num, in_paths in tqdm(input_file_paths.items(), desc='Processing season episodes'):

        if ep_num not in done_epnum:

            transcript = pd.read_csv(in_paths['transcript'], sep = '\t')
            text_chunk: str = ''
            transcript_tokens = []
            for i in range(transcript.shape[0]):
                if not pd.isnull(transcript["text_per_tr"][i]):
                    text_chunk += str(transcript["text_per_tr"][i])
                    # TODO: run some sanity checks to validate truncating side (from the left)
                text_ids = prep_text(text_chunk, tokenizer, ll_args.window_max_length)
                transcript_tokens.append(
                    np.pad(
                        text_ids, (0, ll_args.window_max_length-len(text_ids))
                    )
                )

            with h5py.File(ll_path, "a") as f:
                group = f.create_group(ep_num) if not ep_num in f else f[ep_num]
                group.create_dataset(
                    "transcript_features",
                    data=np.array(transcript_tokens),
                    **{
                        "compression": "gzip",
                        "compression_opts": 4,
                    },
                )


            vreader, fps, num_frames, duration, tr_list = load_video(in_paths['video'], ll_args.tr)
            video_tokens = []
            for end_tr in tr_list:
                video_tokens.append(extract_video_chunk(
                    vreader,
                    ll_args.video_processor,
                    end_tr,
                    ll_args.window_duration,
                    fps,
                    num_frames,
                    ll_args.tr,
                    ll_args.frames_per_tr,
                    ll_args.image_aspect_ratio,
                ))

            with h5py.File(ll_path, "a") as f:
                group = f.create_group(ep_num) if not ep_num in f else f[ep_num]
                group.create_dataset(
                    "video_features",
                    data=np.array(video_tokens),
                    **{
                        "compression": "gzip",
                        "compression_opts": 4,
                    },
                )


if __name__ == "__main__":

    args = get_arguments()

    make_lazy_loading_videollama2(args)
