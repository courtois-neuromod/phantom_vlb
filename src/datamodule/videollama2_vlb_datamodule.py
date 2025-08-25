"""
Dataloader

Links:
- CNeuromax base datamodule:
https://github.com/courtois-neuromod/cneuromax/blob/main/cneuromax/fitting/deeplearning/datamodule/base.py
- Video transformer replay datamodule:
https://github.com/courtois-neuromod/video_transformer/blob/main/src/datasets/replay_datamodule.py

- phantom_LLM
Isil data loading for brain alignment:
https://github.com/courtois-neuromod/phantom_LLM/blob/dev_align/phantom_LLM/src/models/run_baseline.py

- VideoLLaMa2
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L248
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

#from datasets import Dataset as HFDataset  # huggingface datasets
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')

from src import (
    get_hrf_weight,
)


@dataclass
class VLBDataModuleConfig:
    """Holds :class:`VLBDataModuleConfig` config values.

    Args:
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        shuffle_val_data: Whether to shuffle the validation data\
            during training.
        max_per_device_batch_size: See\
            :attr:`~BaseDataModule.per_device_batch_size`. Sets an\
            upper bound on the aforementioned attribute.
        fixed_per_device_batch_size: See\
            :attr:`~BaseDataModule.per_device_batch_size`. Setting this\
            value skips the batch size search in\
            :func:`.find_good_per_device_batch_size` which is\
            not recommended for resource efficiency.
        fixed_per_device_num_workers: See\
            :attr:`~BaseDataModule.per_device_num_workers`. Setting\
            this value skips the num workers search in\
            :func:`.find_good_per_device_num_workers` which is\
            not recommended for resource efficiency.
    """

    features_path: str
    timeseries_path: str
    lazyload_path: str
    subject: str
    seasons: list[str]
    delay: int
    window: int
    random_state: int
    shuffle_val_data: bool
    batch_size: int = 1
    num_workers: int = 0

    # from cneuromax; re-use?
    #device: An[str, one_of("cpu", "gpu")] = "${config.device}"
    #shuffle_val_data: bool = True
    #max_per_device_batch_size: An[int, ge(1)] | None = None
    #fixed_per_device_batch_size: An[int, ge(1)] | None = None
    #fixed_per_device_num_workers: An[int, ge(0)] | None = None


class VLB_Dataset(Dataset):
    def __init__(
        self: "VLB_Dataset",
        config: VLBDataModuleConfig,
        seasons: list[str],
        set_name: str,
    ) -> None:
        """
        Vision-Language-Brain dataloader for VideoLLaMa2 fine-tuning
        Args:
            config: datamodule configuration parameters
            seasons: list of str, friends seasons dedicated to a given dataset
        """
        self.config = config
        self.seasons = seasons
        self.ds_file = None
        self.set_name = set_name

        # todo: fix this!
        # implement function that builds datamatrices for predictor features and output target features
        # Align them!
        # For input: since includes 3 TRs window of frames, can only start at
        # input TR = 2 (drop first 2 TRs of INPUT and OUTPUT when window = 3)

        # delay is 3 TRs back (window offset)
        # - remove first 3 TRs of brain timeseries; (if 3 = self.delay)
        # - remove excedent TRs of video frames input features at the tail END
        # - truncate or pad tail end of language features to match length of timeseries
        # - concatenate across all runs... sample from it w getitem function
        #self.ll_path = self.config.lazyload_path.replace('$SLURM_TMPDIR', os.environ["SLURM_TMPDIR"]).replace('*', f"{self.set_name}")
        self.ll_path = self.config.lazyload_path.replace('$SCRATCH_PATH', os.environ["SCRATCH_PATH"]).replace('*', f"{self.set_name}")
        #f_path = self.config.features_path

        if not Path(self.ll_path).exists():
            """
            Build dataset's lazy loading file
            """
            idx = 0
            # load brain timeseries .h5 file
            #b_path = self.config.timeseries_path.replace('$SLURM_TMPDIR', os.environ["SLURM_TMPDIR"])
            b_path = self.config.timeseries_path.replace('$SCRATCH_PATH', os.environ["SCRATCH_PATH"])
            b_file = h5py.File(b_path, "r")
            ep_keys = {
                run.split("_")[1].split("-")[-1]: (ses, run) for ses, val in b_file.items() for run in val.keys()
            }

            for s in self.seasons:

                #f_path = self.config.features_path.replace('$SLURM_TMPDIR', os.environ["SLURM_TMPDIR"]).replace('*', f"s{s[-1]}")
                f_path = self.config.features_path.replace('$SCRATCH_PATH', os.environ["SCRATCH_PATH"]).replace('*', f"s{s[-1]}")
                f_file = h5py.File(f_path, "r")
                epi_list = [
                    x for x in f_file.keys()
                ]

                for ep_num in epi_list:
                    if ep_num in ep_keys:
                        ses, run = ep_keys[ep_num]
                        run_tseries = np.array(b_file[ses][run])[(self.config.window-1)+self.config.delay:]
                        # TR onset assigned to the middle of a TR; onset + (1.49s/2)
                        run_tr_onsets = [((self.config.window-1)+self.config.delay+0.5+i)*1.49 for i in range(run_tseries.shape[0])]

                        run_vision = np.array(f_file[ep_num]['video_features'])[(self.config.window-1):]
                        num_frames = run_vision.shape[1]
                        """
                        Time diff from middle of TR for each downsampled frame's hidden features
                        Downsampled by sampler of vllama2 connector, a nn.3DConv layer (pad=1, stride=2)
                        12 frames of 24x24 -> 7 downsampled frames of 13x13 (169 features/frame)
                        """
                        num_ds_frames = math.floor(num_frames/2) + 1
                        step = self.config.window/(num_ds_frames-1)
                        # delay between onset of input window and target TR's time stamp (assigned to middle of a TR, hence +0.5)
                        abs_tr_delay = (self.config.window-1)+self.config.delay + 0.5
                        run_vis_onsets = 1.49*(abs_tr_delay - np.arange(0, (self.config.window+step), step))
                        run_vis_weights = np.array([
                                get_hrf_weight(t) for t in run_vis_onsets
                            ])

                        run_language = np.array(f_file[ep_num]['transcript_features'])[(self.config.window-1):]
                        run_lang_onsets = np.array(f_file[ep_num]['transcript_onsets'])[(self.config.window-1):]
                        """
                        Three int per examplar
                        0: number of 0s padded at end of language input ids (right-side padding)
                        1: number of tokens in the instruction portion of the input lang sequence
                        2: number of tokens in the dialogue portion of the input lang sequence
                        """
                        run_maskval = np.array(f_file[ep_num]['masking_params'])[(self.config.window-1):]

                        assert run_maskval.shape[0] == run_language.shape[0]
                        n_rows = min(
                            (run_tseries.shape[0], run_vision.shape[0], run_language.shape[0]),
                        )

                        for n in range(n_rows):
                            pad_len, inst_len, diag_len = run_maskval[n]
                            trial_lang_weights = np.array([
                                get_hrf_weight(t) for t in run_tr_onsets[n] - run_lang_onsets[n][:diag_len]
                            ])
                            run_lang_onsets[n][:diag_len] = trial_lang_weights
                            #run_lang_onsets[n][:diag_len] = run_tr_onsets[n] - run_lang_onsets[n][:diag_len]

                            with h5py.File(self.ll_path, "a") as f:
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

            with h5py.File(self.ll_path, "a") as f:
                f.create_dataset("dset_len", data=[idx+1])

        with h5py.File(self.ll_path, "r") as f:
            #self.length = max([int(s) for s in f.keys()]) + 1
            self.length = np.array(f["dset_len"])[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.ds_file is None:
            self.ds_file = h5py.File(self.ll_path, "r")
        item = {}
        # TODO: validate this part w lit module train step
        for mod in ['timeseries', 'vision', 'language']:
            item[mod] = torch.from_numpy(np.array(self.ds_file[f"{idx}"][f"{idx}_{mod}"])).float()
        for mod in ['padvals', 'vis_weights', 'lang_weights']:
            item[mod] = np.array(self.ds_file[f"{idx}"][f"{idx}_{mod}"])
        return item


@dataclass
class VLBDatasets:
    """Holds phase-specific :class:`~torch.utils.data.Dataset` objects.

    Using the word ``phase`` to not overload :mod:`lightning` ``stage``
    terminology used for ``fit``, ``validate`` and ``test``.

    Args:
        train: Training dataset.
        val: Validation dataset.
        test: Testing dataset.
        predict: Prediction dataset.
    """
    config: VLBDataModuleConfig
    train: VLB_Dataset | None = None
    val: VLB_Dataset | None = None
    test: VLB_Dataset | None = None

    def __post_init__(self):
        # split seasons between train and val datasets
        r = np.random.RandomState(self.config.random_state)
        val_season = r.choice(self.config.seasons, 1).tolist()
        train_seasons = [
            s for s in self.config.seasons if s not in val_season
        ]
        # instantiate lazyloading datasets
        self.val = VLB_Dataset(self.config, val_season, "val")
        self.train = VLB_Dataset(self.config, train_seasons, "train")


class VLBDataModule(LightningDataModule):
    """Base :mod:`lightning` ``DataModule``.

    With ``<phase>`` being any of ``train``, ``val``, ``test`` or
    ``predict``, subclasses need to properly define the
    ``datasets.<phase>`` attribute(s) for each desired phase.

    Args:
        config: See :class:`VLBDataModuleConfig`.

    Attributes:
        config (:class:`VLBDataModuleConfig`)
        datasets (:class:`Datasets`)
        collate_fn (``callable``): See \
            :paramref:`torch.utils.data.DataLoader.collate_fn`.
        pin_memory (``bool``): Whether to copy tensors into device\
            pinned memory before returning them (is set to ``True`` by\
            default if :paramref:`~BaseDataModuleConfig.device` is\
            ``"gpu"``).
        per_device_batch_size (``int``): Per-device number of samples\
            to load per iteration. Temporary value (``1``) is\
            overwritten in :func:`.set_batch_size_and_num_workers`.
        per_device_num_workers (``int``): Per-device number of CPU\
            processes to use for data loading (``0`` means that the\
            data will be loaded by each device's assigned CPU\
            process). Temporary value (``0``) is later overwritten\
            in :func:`.set_batch_size_and_num_workers`.
    """

    def __init__(self: "VLBDataModule", config: VLBDataModuleConfig) -> None:
        super().__init__()
        self.config: VLBDataModuleConfig = config
        self.datasets = VLBDatasets(self.config)

    def x_dataloader(
        self: "VLBDataModule",
        dataset: VLB_Dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        """Generic :class:`~torch.utils.data.DataLoader` factory method.

        Args:
            dataset: A :mod:`torch` ``Dataset`` to wrap with a\
                :class:`~torch.utils.data.DataLoader`
            shuffle: Whether to shuffle the dataset when iterating\
                over it.

        Raises:
            AttributeError: If :paramref:`dataset` is ``None``.

        Returns:
            A new :class:`~torch.utils.data.DataLoader` instance\
                wrapping the :paramref:`dataset` argument.
        """
        if dataset is None:
            raise AttributeError
        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def train_dataloader(self: "VLBDataModule") -> DataLoader:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.train``.

        Returns:
            A new training :class:`torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(dataset=self.datasets.train)

    def val_dataloader(self: "VLBDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.val``.

        Returns:
            A new validation :class:`~torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(
            dataset=self.datasets.val,
            shuffle=self.config.shuffle_val_data,
        )


def build_datamodule(
    config: DictConfig,
) -> VLBDataModule:
    return VLBDataModule(config)

