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

from dataclasses import dataclass

from datasets import Dataset as HFDataset  # huggingface datasets
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class VLB_Dataset(Dataset):
    def __init__(self, config, seasons):
        """
        Vision-Language-Brain dataloader for VideoLLaMa2 fine-tuning
        Args:
            config: VLBDataModuleConfig, datamodule configuration parameters
            seasons: list of str, friends seasons dedicated to a given dataset
        """
        self.config = config
        self.seasons: list[str] = seasons

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
        idx = 0
        self.idx_dict = {}
        # load brain timeseries .h5 file
        bpath = Path(self.config.timeseries_path.replace('*', self.config.subject)).resolve()
        b_file = h5py.File(bpath, "r")
        ep_keys = {
            run.split("_")[2]: (ses, run) for run in bfile[ses].keys() for ses in b_file.keys()
        }

        for s in self.seasons:

            fpath = Path(self.config.features_path.replace('*', s)).resolve()
            epi_list = [
                x for x in h5py.File(fpath, "r").keys()
            ]

            for ep_num in epi_list:
                if ep_num in ep_keys:
                    s, r = ep_keys[ep_num]
                    run_tseries = np.array(bpath[s][r])[(self.config.window-1)+self.config.delay:]

                    run_vision = np.array(h5py.File(fpath, "r")[ep_num]['video_features'])[(self.config.window-1):]
                    run_language = np.array(h5py.File(fpath, "r")[ep_num]['transcript_features'])[(self.config.window-1):]

                    n_rows = np.min(
                        run_tseries.shape[0], run_vision.shape[0], run_language.shape[0],
                    )

                    for n in range(n_rows):
                        with h5py.File(self.config.lazyload_path, "a") as f:
                            group = f.create_group(idx)
                            group.create_dataset(
                                f"{idx}_timeseries", data=run_tseries[idx],
                            )
                            group.create_dataset(
                                f"{idx}_vision", data=run_vision[idx],
                            )
                            group.create_dataset(
                                f"{idx}_language", data=run_language[idx],
                            )

                        idx += 1

        with h5py.File(self.config.lazyload_path, "r") as f:
            self.length = max([int(s) for s in f.keys()]) + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.ds_file is None:
            self.ds_file = h5py.File(self.config.lazyload_path, "r")
        item = {}
        for mod in ['timeseries', 'vision', 'language']:
            item[mod] = torch.from_numpy(self.ds_file[f"{idx}"][f"{idx}_{mod}"]).float()
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
    config: VLBDataModuleConfig = config
    train: VLB_Dataset | None = None
    val: VLB_Dataset | None = None
    test: VLB_Dataset | None = None

    def __post_init__(self):
        # TODO: implement instantiation of datasets
        # split seasons between train and val datasets
        r = np.random.RandomState(self.config.random_state)
        val_season = r.choice(self.config.seasons, 1).tolist()
        train_seasons = [
            s for s in self.config.seasons if s not in val_season
        ]
        self.val = VLB_Dataset(self.config, val_season)
        self.train = VLB_Dataset(self.config, train_seasons)


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
    delay: str
    window: str
    random_state: int
    batch_size: 1
    num_workers: 0
    shuffle_val_data: bool

    # from cneuromax; re-use?
    #device: An[str, one_of("cpu", "gpu")] = "${config.device}"
    #shuffle_val_data: bool = True
    #max_per_device_batch_size: An[int, ge(1)] | None = None
    #fixed_per_device_batch_size: An[int, ge(1)] | None = None
    #fixed_per_device_num_workers: An[int, ge(0)] | None = None


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
        self.config = config
        self.datasets = VLBDatasets(self.config)

    @final
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

    @final
    def train_dataloader(self: "VLBDataModule") -> DataLoader:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.train``.

        Returns:
            A new training :class:`torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(dataset=self.datasets.train)

    @final
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
