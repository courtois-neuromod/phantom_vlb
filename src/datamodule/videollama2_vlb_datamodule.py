import math
import os, glob
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


def get_idx(ranges, val):
    for i, (min_val, max_val) in enumerate(ranges):
        if min_val <= val < max_val:
            return i
    return -1


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
    lazyload_path: str
    subject: str
    seasons: list[str]
    delay: int
    window: int
    random_state: int
    shuffle_val_data: bool
    batch_size: int = 1
    num_workers: int = 0


class VLB_Dataset(Dataset):
    def __init__(
        self: "VLB_Dataset",
        ds_paths: list[str],
    ) -> None:
        """
        Vision-Language-Brain dataloader for VideoLLaMa2 fine-tuning
        
        Features were preprocessed with videollama2_vlb_extractfeatures.py
        and videollama2_vlb_lazyloading.py for fast and easy batching
        Args:
            config: datamodule configuration parameters
            seasons: list of str, friends seasons dedicated to a given dataset
        """
        self.ds_files = {}
        self.length = 0
        self.ranges = []

        for i, ds_path in enumerate(ds_paths):
            self.ds_files[i] = {
                'ds_file': h5py.File(ds_path, "r"),
                'idx_from': self.length,
            }

            ds_length = int(np.array(
                self.ds_files[i]['ds_file']['dset_len']
            )[0])
            self.ranges.append((self.length, self.length + ds_length))
            self.length += ds_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """."""
        i = get_idx(self.ranges, idx)
        ds_file = self.ds_files[i]['ds_file']
        set_idx = idx - self.ds_files[i]['idx_from']

        item = {}
        for mod in ['timeseries', 'vision', 'language']:
            item[mod] = torch.from_numpy(np.array(ds_file[f"{set_idx}"][f"{set_idx}_{mod}"])).float()
        for mod in ['padvals', 'vis_weights', 'lang_weights']:
            item[mod] = np.array(ds_file[f"{set_idx}"][f"{set_idx}_{mod}"])
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
        """
        Split feature files into train and val dsets 
        """
        # list lazy loading feature files
        f_list = []
        for s in self.config.seasons:
            f_list += sorted(glob.glob(
                self.config.lazyload_path.replace('$SCRATCH_PATH', os.environ["SCRATCH_PATH"]).replace('s*', f"{s}")
            )) 
        # split feature files between train and val datasets
        r = np.random.RandomState(self.config.random_state)
        val_file = r.choice(f_list, 1).tolist()
        train_files = [
            x for x in f_list if x not in val_file
        ]
        # log train and val dsets as h-params
        self.dset_names = {
            'val_set': [os.path.basename(x) for x in val_file],
            'train_set': [os.path.basename(x) for x in train_files],
        }
        # instantiate lazyloading datasets
        self.val = VLB_Dataset(val_file)
        self.train = VLB_Dataset(train_files)


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
