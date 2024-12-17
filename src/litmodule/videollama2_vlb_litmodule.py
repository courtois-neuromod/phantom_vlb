import argparse
import math

import numpy as np
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


sys.path.append('../../')

from VideoLLaMA2.videollama2.model.videollama2_mistral import (
    Videollama2MistralForCausalLM,
    Videollama2MistralConfig,
)

from VideoLLaMA2.videollama2.mm_utils import get_model_name_from_path
from VideoLLaMA2.videollama2.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    MODAL_INDEX_MAP,
)

NUM_FRAMES = 12  # (must match lazyloading... 3TRs and 4 frames per TR); higher than default of 8 in vllama2's CONSTANTS


def load_pretrained_vllama2(
    config,
) -> Videollama2MistralForCausalLM:
    """
    sources (old branch):
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/3fa0ea5d33ee66a9915c43443ea5e9b19bb0c66e/videollama2/model/__init__.py#L48
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py
    """
    model_name = get_model_name_from_path(config.model_path)

    model_config = Videollama2MistralConfig.from_pretrained(config.model_path, trust_remote_code=True)
    #model_config = AutoConfig.from_pretrained(config.model_path)

    model_config._attn_implementation = "flash_attention_2"

    model_type = config.model_type
    is_pretraining = False  # config.tune_mm_mlp_adapter

    # todo to implement Lora...
    #https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/3fa0ea5d33ee66a9915c43443ea5e9b19bb0c66e/videollama2/model/__init__.py#L87

    model =  Videollama2MistralForCausalLM.from_pretrained(
        config.model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        do_sample=True,
    )
    model.config.use_cache = False

    if config.freeze_backbone:
        model.model.requires_grad_(False)

    # Needed w lightning?
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.config.mm_projector_lr = None
    model.config.num_frames = NUM_FRAMES

    return model


@dataclass
class VLBLitModuleConfig:
    """Holds :class:`VLBLitModuleConfig` config values.
    Args:
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        shuffle_val_data: Whether to shuffle the validation data\
            during training.
        max_per_device_batch_size: See\
            :attr:`~BaseDataModule.per_device_batch_size`. Sets an\
            upper bound on the aforementioned attribute.
    """
    model_type: str
    model_path: str
    pretrain_mm_mlp_adapter: str
    freeze_backbone: bool

    def __post_init__(self):
        self.dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VLBLitModule(LightningModule):
    """."""

    def __init__(self: "VLBLitModule", config: VLBLitModuleConfig) -> None:
        super().__init__()

        self.config = config

        self.nnmodule = load_pretrained_vllama2(self.config)

        self.optimizer = instantiate(
            config.optimizer,
        )
        optimizer(params=self.parameters())
        self.lrscheduler = lrscheduler(optimizer=self.optimizer)


    def training_step(

    ):
