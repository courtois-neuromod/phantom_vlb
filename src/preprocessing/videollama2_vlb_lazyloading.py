import argparse

import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append('../../')

from src import (
    get_hrf_weight,
)


def get_arguments():
    """."""
    parser = argparse.ArgumentParser(
        description="Compile parameters for input feature lazy_loading for VideoLLaMa2"
    )
    parser.add_argument(
        '--features_path', required=True, type=str, help='Path to the extracted features file.'
    )
    parser.add_argument(
        '--timeseries_path', required=True, type=str, help="Path to the subject's extracted timeseries file."
    )
    parser.add_argument(
        '--lazy_load_path', required=True, type=str, help='Path where to save the processed features in .h5 output files.'
    )
    parser.add_argument(
        '--subject', required=True, type=str, help='subject id, e.g., "sub-01".'
    )
    parser.add_argument(
        '--season', required=True, type=str, help='Friends season from which to extract lazy loading features. E.g., "s1"'
    )
    parser.add_argument(
        '--n_split', default=4, type=int, help="Number of .h5 files into which to split the season's data."
    )
    parser.add_argument(
        '--delay', default=3, type=int, help='Offset, in TRs, between the last TR in the prediction window and the target brain TR.'
    )
    parser.add_argument(
        '--window', default=3, type=int, help='Number of TRs included in the prediction window.'
    )

    return parser.parse_args()


def make_lazy_loading_dsets(config):
    """
    Load subject's BOLD data timeseries 
    Compile list of extracted episodes (all seasons)
    """
    b_file = h5py.File(config.timeseries_path, "r")
    ep_keys = {
        run.split("_")[1].split("-")[-1]: (ses, run) for ses, val in b_file.items() for run in val.keys()
    }

    """
    Load season's pre-extracted vision and language features
    Compile list of the season's episodes whose data have corresponding
    BOLD timeseries for that subject.
    
    Assign episodes to n_split chunks and align the input (video) features and
    target brain (BOLD) timeseries into examplars for easy batching.

    Since the input window (len = w) includes multiple TRs worth of movie frames,
    drop the first w - 1 TRs from the INPUT and OUTPUT arrays.
    E.g., drop first 2 TRs of the INPUT and OUTPUT arrays when window = 3)

    If delay is n TRs back (the input window offset, as in, how far back from the target TR)
    - also remove the first n TRs from the brain timeseries (so n + w - 1)
    - remove the excedent TRs of video frames at the tail END of the input features
        as they lack corresponding BOLD data
    - truncate or pad tail end of language features to match length of timeseries matrix
    """
    f_file = h5py.File(config.features_path, "r")
    epi_list = [
        x for x in f_file.keys() if x in ep_keys
    ]

    chunk_idx = np.floor(
        np.arange(len(epi_list))/(len(epi_list)/config.n_split)
    ).astype(int)

    for i in range(config.n_split):
        ll_path = f"{config.lazyload_path}/friends_llFile_{config.subject}_{config.season}.h5"

        idx = 0
        chunk_epi_list = np.array(epi_list)[chunk_idx==i].tolist()
        for ep_num in chunk_epi_list:
            ses, run = ep_keys[ep_num]
            run_tseries = np.array(b_file[ses][run])[(config.window-1)+config.delay:]
            # TR onset assigned to the middle of a BOLD TR; onset + (1.49s/2)
            run_tr_onsets = [((config.window-1)+config.delay+0.5+i)*1.49 for i in range(run_tseries.shape[0])]
            
            run_vision = np.array(f_file[ep_num]['video_features'])[(config.window-1):]
            num_frames = run_vision.shape[1]
            """
            Time diff from middle of TR for each downsampled frame's hidden features
            Downsampled by sampler of vllama2 connector, a nn.3DConv layer (pad=1, stride=2)
            12 frames of 24x24 -> 7 downsampled frames of 13x13 (169 features/frame)
            """
            num_ds_frames = math.floor(num_frames/2) + 1
            step = config.window/(num_ds_frames-1)
            # delay between onset of input window and target TR's time stamp (assigned to middle of a TR, hence +0.5)
            abs_tr_delay = (self.config.window-1)+self.config.delay + 0.5
            run_vis_onsets = 1.49*(abs_tr_delay - np.arange(0, (config.window+step), step))
            run_vis_weights = np.array([
                get_hrf_weight(t) for t in run_vis_onsets
            ])

            run_language = np.array(f_file[ep_num]['transcript_features'])[(config.window-1):]
            run_lang_onsets = np.array(f_file[ep_num]['transcript_onsets'])[(config.window-1):]
            """
            Three int saved per examplar
            index 0: (pad_len) number of 0s padded at end of language input ids (right-side padding)
            index 1: (inst_len) number of tokens in the instruction portion of the input lang sequence
            index 2: (diag_len) number of tokens in the dialogue portion of the input lang sequence
            """
            run_maskval = np.array(f_file[ep_num]['masking_params'])[(config.window-1):]

            assert run_maskval.shape[0] == run_language.shape[0]
            n_rows = min(
                (run_tseries.shape[0], run_vision.shape[0], run_language.shape[0]),
            )

            # Save run's examplars (input and output) to .h5 file 
            for n in range(n_rows):
                pad_len, inst_len, diag_len = run_maskval[n]
                trial_lang_weights = np.array([
                    get_hrf_weight(t) for t in run_tr_onsets[n] - run_lang_onsets[n][:diag_len]
                ])
                run_lang_onsets[n][:diag_len] = trial_lang_weights
                #run_lang_onsets[n][:diag_len] = run_tr_onsets[n] - run_lang_onsets[n][:diag_len]

                with h5py.File(ll_path, "a") as f:
                    group = f.create_group(f"{idx}")
                    group.create_dataset(
                        f"{idx}_timeseries", data=run_tseries[n],
                    )
                    group.create_dataset(
                        f"{idx}_vision", data=run_vision[n],
                    )
                    group.create_dataset(
                        f"{idx}_vis_weights", data=run_vis_weights,
                    )
                    group.create_dataset(
                        f"{idx}_language", data=run_language[n],
                    )
                    group.create_dataset(
                        f"{idx}_lang_weights", data=run_lang_onsets[n],  # converted to weights
                    )
                    group.create_dataset(
                        f"{idx}_padvals", data=run_maskval[n],
                    )
                idx += 1

    f_file.close()
    b_file.close()

    with h5py.File(ll_path, "a") as f:
        f.create_dataset("dset_len", data=[idx+1])
        
    print(f"Built lazy loading dset for {subject}, season {s}")


if __name__ == "__main__":

    args = get_arguments()

    make_lazy_loading_dsets(args)
