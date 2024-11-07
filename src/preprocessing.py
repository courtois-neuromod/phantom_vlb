"""
TODO:

Step1: implement prepocessing steps to build .h5 of input data from brain, vision, language (and audio)...

Step 2: implement my own Dataset class w __getitem__ and __len__

Step 3: implement datamodule

From video-llama2 dataset (in pytorch)
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/train.py#L224


https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/mm_utils.py

# TODO: (down the road) add audio branch to video-llama2

https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/videollama2_arch.py#L115

https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/eval/inference_video_mcqa_videomme.py#L65

link to making lazy-loading file from F Paugam's video_transformer project
https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/datasets/replay_datamodule.py#L125
"""

import glob
import math
import os
import sys
from dataclasses import dataclass, field
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

#from transformers import (
#    CLIPImageProcessor,
#    CLIPVisionConfig,
#    CLIPVisionModel,
#)

sys.path.append('../')

from VideoLLaMA2.videollama2.mm_utils import (
    expand2square,
    frame_sample,
    tokenizer_multimodal_token,
)
from VideoLLaMA2.videollama2.model.encoder import CLIPVisionTower

# HuggingFace read-access token
# token name: vllama2_from_beluga
# access_token = "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"

"""
Values from
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/constants.py#L20
"""
NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1

# Added; trying something
FRAMES_PER_TR = 4

"""
Dataclass adapted from : https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L71
Default parameters from : https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45
"""
@dataclass
class LazyLoadArguments:
    input_transcript_path: str = field(default="../data/stimuli/transcripts", metadata={"help": "Path to the input transcripts data."})
    input_video_path: str = field(default="../data/stimuli/videos", metadata={"help": "Path to the input video data."})
    lazy_load_path: str = field(default="../results/videollama/lazy_loading.h5", metadata={"help": "Path to save the .h5 data."})
    model_type: str = field(default="videollama2", metadata={"help": "Model type selected in the list: videollama2, videollama2_llama, videollama2_mistral, videollama2_mixtral, videollama2_qwen2"})
    model_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    window_max_length: int = field(default=2036, metadata={"help": "Maximum sequence length per input window. Sequences will be truncated from the left up to end of window."})
    multimodal_token_index: int = field(default=-201)
    #vision_tower: str = field(default="openai/clip-vit-large-patch14-336") # load from HuggingFace
    vision_tower: str = field(default="models/vision_tower/openai/clip-vit-large-patch14-336") # load from local path
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default="patch")
    bf16 = True
    device: int = 0
    image_aspect_ratio: str = field(default="pad")
    frames_per_tr: int = field(default=4)
    #num_frames: int = field(default=8)
    tr: float = field(default=1.49)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    window_duration: int = 3  # duration, in TRs, of the input time window (e.g., 3 = 3 TRs worth of video frames)

    # How far back in time (in TRs) does the input window ENDS in relation to the TR (can set, and vary, in data_module...)
    # it predicts.
    # E.g., back = 3 means that input features are sampled up to 3 TRs before the target BOLD TR onset"
    #input_back: int = 3


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
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/videollama2_arch.py#L43

    VideoLLaMA2 has a vision tower, which is a pre-trained encoder from OpenAI's Clip (weights are frozen during finetuning)
    The VideoLLaMA2 datamodule has a dataset that, with __getitem__, processes a video through the vision tower (nn.Module's forward)
    before handing it over to the datamodule that gives batches to the main model.

    Here, with lazy-loading, I'm trying to process each video in advance through the data tower and store as .h5 for lazy loading
    to speed up processing

    From FP's video_transformer, decide on the window of video, and corresponding language, to pair w each TR

    Configs are split into data classes (model, training and data) in train script, and specified in finteune script:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/scripts/custom/finetune.sh#L45
    """

    # TODO: implement vllama2 library to import and re-use the CLIPVisionTower class as is (it's the vision tower)
    #https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45

    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/videollama2_arch.py#L43
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/encoder.py#L12
    vision_tower = CLIPVisionTower(ll_args.vision_tower, args=ll_args)

    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L510
    vision_tower.to(dtype=torch.bfloat16 if ll_args.bf16 else torch.float16, device=ll_args.device)

    # alternative (for beluga); TODO: test in interactive session
    #torch.cuda.set_device('cuda:0')
    #model.to(device)

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
        #truncation=True,  # TODO: this was added; I need to truncate, but from the left; check how to do it...
        #truncation_side="left",
        #token=access_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def prep_text(text, tokenizer, window_max_length):
    """
    Source
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L168
    """
    # from
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/constants.py#L28
    multimodal_token_index = -201

    # truncate text input to fix max language window length (in tokens)
    tokens = tokenizer.tokenize(text.strip())
    if len(tokens) > window_max_length:
        tokens = tokens[-window_max_length:]
    text = tokenizer.convert_tokens_to_string(tokens)

    # prep: pass text through .strip() to remove extra white spaces,
    # and add modal token to begining of input text for each entry

    modal_token = "<video>"
    message = [{'role': 'user', 'content': modal_token + '\n' + text.strip()}]

    #then process message to pass through tokenizer...
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L168
    #https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L289

    #check also: inference script
    #https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/__init__.py#L32
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    # return list, not pytorch tensor (otherwise set return_tensors='pt')
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


def make_lazy_loading_videollama2():
    """Support function to preprocess input and output data for multimodal brain alignment
    experiments with video-LLaMA2. To run separately ahead of training / testing / validation.

    Generates a HDF5 file to be used by a dataloader for lazy loading.
    This script is adapted from
    https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/datasets/replay_datamodule.py#L125

    The video and text data pre-processing steps are adapted from video-LLaMA2
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/train.py

    TODO: adapt this below
    The frames are uniformly downsampled, effectively dividing the original
    framerate by `time_downsample`.

    Args:
        brain_path: str, path to the HDF5 file containing the extracted brain timeseries
        input_data_path: str, path to the repository containing the .mkv (visual, audio) and .tsv (language) files
        lazy_load_path: str, path to the HDF5 to create
        epi_list: list of str, list of episodes to use
        n_frames: int, number of video frames in one sample (to match one TR)
        modalities: list of str, name of the input modalities to use
        time_downsample: int, Factor by which the frame frequency is divided, i.e.
            one out of every `time_downsample` frames will be kept
        stride: int, stride between samples, None will make it equal to n_frames.
            Defaults to None.
        fps: int, Numbre of frames per seconds in the original data. Defaults to 60. (videogames; friends: 29.97?)


    OLD Args:
        data_path: str, path to the HDF5 file containing the frames and modalities
        lazy_load_path: str, path to the HDF5 to create
        data_list: list of str, list of the HDF5 dataset paths in the data file to
            use
        n_frames: int, number of frames in one sample
        modalities: list of str, name of the other modalities to use, only frames
            are used if the list is empty
        time_downsample: int, Factor by which the frame frequency is divided, i.e.
            one out of every `time_downsample` frames will be kept
        stride: int, stride between samples, None will make it equal to n_frames.
            Defaults to None.
        fps: int, Numbre of frames per seconds in the original data. Defaults to 60.
    """
    parser = transformers.HfArgumentParser((LazyLoadArguments))
    ll_args = parser.parse_args_into_dataclasses()

    # debug hack
    #ll_args = LazyLoadArguments(input_transcript_path="../algonauts_dset/competitors/stimuli/transcripts/friends/s1", input_video_path="../friends_algonauts/data/friends.stimuli/s1", lazy_load_path="results/videollama2/lazyloading/friends/friends_s1_features.h5")

    ll_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output file with lazyloading data
    ll_path = str(Path(f"{ll_args.lazy_load_path}").resolve())
    done_epnum = get_done_ep(ll_path)
    input_file_paths = get_input_paths(ll_args)

    ll_args = prep_video_processor(ll_args)
    tokenizer = prep_tokenizer(ll_args)

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

            # loose notes to clean up
            # TODO: produce hdf5 file of encoded text and video frames for lazy loading
            # For each modality, have one row corresponding to a single TR
            # Important : don't align based on time windows (e.g., pad text, or cut last brain TRs after
            # movie done, or skip first TRs for which predictor window is not full length), as this will be done inside the dataloader
            # # (no need to re-generate the data to play w window)

            # TODO: adapt from data module dataset (LazySupervisedDataset)'s __getitem__ to preprocess video chunks per TRs
            # determine start and duration of video frame window for each target TR
            # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L387
            # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L315

