"""
This script preprocesses raw vision (movie frames) and language (movie transcripts) input from the CNeuroMod Friends dataset
to implement lazy loading in order to fine-tune VideoLLaMA2 on brain imaging data (fMRI).

The target VideoLLaMA2 model is the pre-trained DAMO-NLP-SG/VideoLLaMA2-7B with a mistral 7B language model and clip vision tower.

Preprocessed text and video frames (input features) are produced per fMRI TR (1.49s).
They are saved in a .h5 file to support lazy loading to speed up fine-tuning.

The processing steps are adapted from the VideoLLaMA2 datamodule dataset class LazySupervisedDataset's __getitem__ function.
- movie frames are processed with the pre-trained CLIP vision tower's processor
- text is formatted and tokenized with the DAMO-NLP-SG/VideoLLaMA2-7B tokenizer

For each TR, movie frames have a short temporal window (default of 3 TRs = 4.47s). Text has a long window, reaching back from the target TR until the episode onset or the model's maximal lenght of input tokens.

Adapted from
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L387
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L315

# TODO: (future step) add audio stream using video-llama2's audio branch
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/audio_visual/videollama2/train.py

Just here as a reference to help select videoframe sampling frequency
Values from
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/constants.py#L20

NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1
"""

import argparse
import ast
import glob
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def get_arguments():
    """."""
    parser = argparse.ArgumentParser(
        description="Compile parameters for input feature lazy_loading for VideoLLaMa2"
    )
    parser.add_argument(
        '--input_transcript_path', required=True, type=str, help='Path to the input transcripts data directory.'
    )
    parser.add_argument(
        '--input_seg_path', required=True, type=str, help='Path to the manual scene segmentation data directory.'
    )
    parser.add_argument(
        '--input_video_path', required=True, type=str, help='Path to the input video data directory.'
    )
    parser.add_argument(
        '--lazy_load_path', required=True, type=str, help='Path where to save the processed features in an .h5 output file.'
    )
    parser.add_argument(
        '--cache_dir', type=str, default="../../models", help='Directory where pre-trained model weights are downloaded.'
    )
    parser.add_argument(
        '--model_type', type=str, default='videollama2', choices=['videollama2', 'videollama2_llama', 'videollama2_mistral', 'videollama2_mixtral', 'videollama2_qwen2'], help='Model type selected from list.'
    )
    parser.add_argument(
        '--model_path', type=str, default='DAMO-NLP-SG/VideoLLaMA2-7B', choices=['mistralai/Mistral-7B-Instruct-v0.2', 'DAMO-NLP-SG/VideoLLaMA2-7B'],
    )
    parser.add_argument(
        '--model_max_length', type=int, default=2048,
    )
    parser.add_argument(
        '--bf16', type=bool, default=True,
    )
    parser.add_argument(
        '--frames_per_tr', type=int, default=4,
    )
    parser.add_argument(
        '--tr', type=float, default=1.49,
    )
    parser.add_argument(
        '--window_duration', type=int, default=3, help='number of TRs worth of video frames included in prediction window',
    )
    return parser.parse_args()


def get_input_paths(ll_args):
    """
    Return dict of paths for available transcript and video files, per episode
    """
    transcript_path = str(Path(ll_args.input_transcript_path).resolve())
    segmentation_path = str(Path(ll_args.input_seg_path).resolve())
    video_path = str(Path(ll_args.input_video_path).resolve())

    input_paths = {}
    for tr_file in sorted(glob.glob(f"{transcript_path}/friends_*.tsv")):
        ep_num = os.path.basename(tr_file).split('_')[-1].split('.')[0]
        v_path = f"{video_path}/friends_{ep_num}.mkv"
        s_path = (f"{segmentation_path}/friends_{ep_num}_manualseg.tsv").replace("s0", "s")

        if Path(v_path).exists() and Path(s_path).exists():
            input_paths[ep_num] = {
                'transcript': tr_file,
                'seg': s_path,
                'video': v_path,
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


def get_sceneonsets(seg_file):
    """."""

    scene_onsets = []
    seen_scenes = []

    for i in range(seg_file.shape[0]):
        scene_num = seg_file["scene"].iloc[i]
        if scene_num not in seen_scenes:
            scene_onsets.append(seg_file["onset"].iloc[i])
            seen_scenes.append(scene_num)

    return scene_onsets


def prep_video_processor(ll_args):
    """
    Instantiate the pre-trained vision tower (CLIP)'s processor to preprocess raw video frames.

    Adapted from:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L510
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh#L45
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/videollama2_arch.py#L43
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/encoder.py#L12
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/__init__.py

    VideoLLaMA2 has a vision tower, which is a pre-trained encoder from OpenAI's Clip model (weights are frozen during pre-training and fine-tuning)
    The VideoLLaMA2 datamodule has a dataset whose __getitem__ function processes raw video input through its vision tower (nn.Module's forward)
    before passing it to the datamodule that feeds batches to the main model.

    Here, we are processing each video in advance through the vision tower and storing its features as .h5
    to speed up fine-tuning using lazy loading.

    Note: VLLaMA2 config params are split into three data classes (model, training and data) in the train script.
    E.g., they are specified in the finteune script here:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/scripts/custom/finetune.sh#L45
    """
    mm_config = transformers.AutoConfig.from_pretrained(ll_args.model_path)

    vision_tower = build_vision_tower(mm_config)

    vision_tower.to(dtype=torch.bfloat16 if ll_args.bf16 else torch.float16, device=ll_args.device)

    ll_args.image_size = vision_tower.image_size
    ll_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor
    ll_args.is_multimodal = True

    return ll_args


def prep_tokenizer(ll_args):
    """
    Instantiate the pre-trained model's tokenizer to save input text as int arrays

    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L494
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/model/__init__.py#L165C9-L165C91
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ll_args.model_path,
        use_fast=True,
        truncate=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def get_max_token(args):
    """
    Cap the number of input text tokens based on the model capacity and visual sequence lenght
    """
    num_frames = args.window_duration * args.frames_per_tr  # number video frames per examplar

    # Input frames are downsampled by the vllama2 connector's sampler
    # the sampler is na n.Conv3d over time (frames), height, width
    # with pad=1, stride=2
    # e.g., 12, 24, 24 -> 7, 13, 13
    num_downsampled_frames = math.floor(num_frames/2) + 1
    max_text_tokens = args.model_max_length - (num_downsampled_frames*169)  # visual seq len; 13 * 13 = 169

    # -1, because the modality token gets removed by vllama2 during input processing
    return max_text_tokens + 1


def prep_text(
    scene_text,
    seg_text,
    word_lists,
    onset_lists,
    tokenizer,
    max_tokens,
):
    """
    Tokenizes raw input text
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
    all_words = [w for w_list in word_lists for w in w_list]
    all_onsets = [o for o_list in onset_lists for o in o_list]
    assert len(all_words) == len(all_onsets)

    if seg_text == "":
        seg_dialog = "No dialogue."
        token_onsets = [0.5, 1.0]  # dummy token times
    else:
        token_onsets = []
        seg_dialog = ""
        for w, o in zip(all_words, all_onsets):
            w_t = tokenizer.tokenize(w)
            token_onsets += [o] * len(w_t)
            seg_dialog += f"{w} "
        assert len(token_onsets) == len(tokenizer.tokenize(seg_dialog.strip()))
        #assert len(token_onsets) == len(tokenizer.tokenize(seg_text.strip()))

    """
    Truncate scene text to fit within max model window length (in tokens)
    after accounting for visual features

    73 is the number of tokens for the instructions + syst_message without the added dialogue
    80 to give a bit of a buffer
    """
    tokens = tokenizer.tokenize(scene_text.strip())
    seg_len = len(tokenizer.tokenize(seg_dialog.strip()))
    max_scene_length = max_tokens - (80 + seg_len)
    if len(tokens) > max_scene_length:
        tokens = tokens[-max_scene_length:]
    background_text = tokenizer.convert_tokens_to_string(tokens).strip()


    # Text Prep
    # pass text through .strip() to remove extra white spaces,
    # add modal token to begining of input text for each entry
    modal_token = "<video>"
    inst_text = "Here are the words spoken in the video:"
    inst_len = len(tokenizer.tokenize(inst_text.strip()))
    instructions = f"{inst_text.strip()} {seg_dialog.strip()}"

    """
    +2 tokens before (/n -> '▁', '<0x0A>')
    +4 tokens after ([/INST] -> '▁[', '/', 'INST', ']')
    """
    message = [{'role': 'user', 'content': modal_token + '\n' + instructions.strip()}]

    system_message = [
        {'role': 'system', 'content': (
        """<<SYS>>\nThis video is from a scene from the TV show Friends. Try to understand what is happening in the video."""
        """\n"""
        f"""For context, here is the dialogue that was spoken just before the video onset: {background_text}.\n<</SYS>>""")
        }
    ]
    message = system_message + message

    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    # returns list, not pytorch tensor (otherwise set return_tensors='pt')
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors=None)#.unsqueeze(0).long().cuda()

    return (
        input_ids, token_onsets, inst_len,
    )


def load_video(video_path, tr = 1.49):
    """
    Adapted from
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L133
    """
    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    fps = vreader.get_avg_fps()
    num_frames = len(vreader)
    duration: float = num_frames / fps

    # END (in seconds) of input movieframe window for a given TR (from video onset)
    tr_list = np.array(range(1, math.ceil(duration/tr)))*tr

    return vreader, fps, num_frames, duration, tr_list.tolist()


def extract_video_chunk(vreader, processor, end_time, win_dur, fps, num_frames_of_video, tr, frames_per_tr):
    """
    Tokenizes raw video frames
    Adapted from:
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/mm_utils.py#L133
    """
    # Input videoframe window onset time (in seconds); up to n = win_dur (in TRs) before end_tr
    start_time = max(0, end_time - tr*(win_dur))

    # Determine frame range & Calculate frame indices
    f_start = max(int(start_time * fps) - 1, 0)
    f_end   = min(int(end_time * fps) - 1, num_frames_of_video - 1)
    all_frame_indices = list(range(f_start, f_end + 1))

    duration = len(all_frame_indices)
    num_frames = round((end_time - start_time) / tr) * frames_per_tr
    sampled_frame_indices = [all_frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

    video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    # pad to 12 images w blank ones for TRs at onset of video (can't go back to full window lenght)
    while len(video_data) < (win_dur * frames_per_tr):
        #video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))
        video_data.append(Image.fromarray(np.swapaxes(np.zeros((*video_data[-1].size, 3), dtype=np.uint8), 0, 1)))

    images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
    #video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    video = np.array(processor.preprocess(images)['pixel_values'])

    return video


def make_lazy_loading_videollama2(ll_args):
    """
    Preprocesses input data (language and video frames) into
    tokenized input features for lazy loading during multimodal brain alignment
    experiments with VideoLLaMA2.

    Script to be ran separately ahead of training / testing / validation,
    for each season of Friends.

    Output is a HDF5 file to be used by a dataloader for lazy loading.
    For each modality, each row of features corresponds to a single fMRI TR (1.49s)

    This script is adapted from F. Paugam's video transformer scripts
    https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/datasets/replay_datamodule.py#L125

    The video and text data pr-processing steps are adapted from video-LLaMA2
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
    maxnum_tokens = get_max_token(ll_args)

    for ep_num, in_paths in tqdm(input_file_paths.items(), desc='Processing season episodes'):

        if ep_num not in done_epnum:

            transcript = pd.read_csv(in_paths['transcript'], sep = '\t')
            seg_times = get_sceneonsets(pd.read_csv(in_paths['seg'], sep = '\t'))

            # run's list of tokens per TR
            run_tokens = []
            run_tk_times = []
            # script text from scene unset until the shown video frames
            scene_chunk = ''
            j = 1
            # script for period matching the shown video frames, split per TR
            tr_chunk = [""] * ll_args.window_duration
            tr_words = [[]] * ll_args.window_duration
            tr_onsets = [[]] * ll_args.window_duration

            # number of padded zeros at the right-side of language tokens
            mask_params = []

            for i in range(transcript.shape[0]):
                if (i * ll_args.tr) > seg_times[j] and j < (len(seg_times) - 1):
                    # reset previously-spoken scene text with new scene onset
                    scene_chunk = ''
                    tr_chunk = [""] * ll_args.window_duration
                    tr_words = [[]] * ll_args.window_duration
                    tr_onsets = [[]] * ll_args.window_duration
                    j += 1

                if not pd.isnull(transcript["text_per_tr"][i]):
                    i_text = str(transcript["text_per_tr"][i])
                    i_words = ast.literal_eval(transcript["words_per_tr"][i])
                    i_times = ast.literal_eval(transcript["onsets_per_tr"][i])
                    assert len(i_words) == len(i_times)
                else:
                    i_text = ""
                    i_words = []
                    i_times = []
                scene_chunk += tr_chunk[0]
                tr_chunk = tr_chunk[1:] + [i_text]
                tr_words = tr_words[1:] + [i_words]
                tr_onsets = tr_onsets[1:] + [i_times]

                """
                Outputs token ids for instructions, previous scene dialogue
                and current video dialogue.
                Also outputs timing (in sec, from episode onset) for each token
                from the current video dialogue.

                Index for -201 id is where systems and users messages are split

                Where 169 (13w x 13h) is the number of embedded tokens per video frame
                for vllama2 (output from connector sampler, a 3D conv layer over time, h and w)
                """
                run_ids, id_onsets, instru_len = prep_text(
                    scene_chunk, ''.join(tr_chunk), tr_words,
                    tr_onsets, tokenizer, maxnum_tokens,
                )

                tr_pad = (maxnum_tokens) - len(run_ids)
                run_tokens.append(
                    np.pad(run_ids, (0, tr_pad))
                )
                # max len is 58 tokens; checked for 6 seaons of friends
                time_pad = 64 - len(id_onsets)
                run_tk_times.append(
                    np.pad(id_onsets, (0, time_pad))
                )
                mask_params.append(
                    np.array(
                        [tr_pad, instru_len, len(id_onsets)]
                ))

            with h5py.File(ll_path, "a") as f:
                group = f.create_group(ep_num) if not ep_num in f else f[ep_num]
                group.create_dataset(
                    "transcript_features",
                    data=np.array(run_tokens),
                    **{
                        "compression": "gzip",
                        "compression_opts": 4,
                    },
                )
                group.create_dataset(
                    "transcript_onsets",
                    data=np.array(run_tk_times),
                    **{
                        "compression": "gzip",
                        "compression_opts": 4,
                    },
                )
                group.create_dataset(
                    "masking_params",
                    data=np.array(mask_params),
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

    os.environ['HF_HOME'] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    import torch
    import transformers
    from decord import VideoReader, cpu
    from PIL import Image
    from tqdm import tqdm

    sys.path.append('../../')
    from VideoLLaMA2.videollama2.mm_utils import (
        expand2square,
        frame_sample,
        tokenizer_multimodal_token,
    )
    from VideoLLaMA2.videollama2.model.encoder import build_vision_tower


    make_lazy_loading_videollama2(args)
