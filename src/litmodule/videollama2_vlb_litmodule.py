#import argparse
#import math
import sys
from dataclasses import dataclass

#import numpy as np
import torch

#from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from torch.optim import Adam, AdamW, lr_scheduler
from transformers import (
    AutoConfig,
    #    AutoModelForCausalLM,
    #    AutoTokenizer,
    #    BitsAndBytesConfig,
    #    PretrainedConfig,
)

sys.path.append('../../')

#from VideoLLaMA2.videollama2.constants import (
#    DEFAULT_IMAGE_TOKEN,
#    DEFAULT_VIDEO_TOKEN,
#    MODAL_INDEX_MAP,
#)
#from VideoLLaMA2.videollama2.mm_utils import get_model_name_from_path
from VideoLLaMA2.videollama2.model.videollama2_mistral import (
    #Videollama2MistralConfig,
    Videollama2MistralForCausalLM,
)

NUM_FRAMES = 12  # (must match lazyloading... 3TRs and 4 frames per TR); higher than default of 8 in vllama2's CONSTANTS


def load_pretrained_vllama2(
    config,
) -> Videollama2MistralForCausalLM:
    """
    sources (old branch):
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/3fa0ea5d33ee66a9915c43443ea5e9b19bb0c66e/videollama2/model/__init__.py#L48
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/scripts/custom/finetune.sh
    https://arxiv.org/pdf/2406.07476

    TODO implement Lora...
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/3fa0ea5d33ee66a9915c43443ea5e9b19bb0c66e/videollama2/model/__init__.py#L87
    """
    model_config = AutoConfig.from_pretrained(config.model_path)
    #model_config = Videollama2MistralConfig.from_pretrained(config.model_path)
    model_config._attn_implementation = "sdpa"  # "flash_attention_2", "sdpa", None

    model =  Videollama2MistralForCausalLM.from_pretrained(
        config.model_path,
        config=model_config,
        torch_dtype=config.dtype,  # torch.bfloat16, torch.float16
        device_map=config.device_map,  # "auto",
    )
    model.config.use_cache = False

    # for pre-training
    if config.freeze_backbone:
        model.model.requires_grad_(False)

    # Needed w lightning?
    #if hasattr(model, "enable_input_require_grads"):
    #    model.enable_input_require_grads()
    #else:
    #    def make_inputs_require_grad(module, input, output):
    #        output.requires_grad_(True)
    #    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.config.mm_projector_lr = None
    model.config.num_frames = NUM_FRAMES

    # freeze vision tower
    # # vision_tower is not trainable in VideoLLaMA2
    model.get_model().vision_tower.requires_grad_(False)

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
    model_path: str
    freeze_backbone: bool
    lr: float
    betas: list[float]
    eps: float
    weight_decay: float
    lr_scheduler_name: str
    last_epoch: int
    t_max: int

    def __post_init__(self):
        self.dtype = torch.float16  # torch.bfloat16 for newer GPUs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map="auto"


class VLBLitModule(LightningModule):
    """
    Required methods are forward, training_step and configure_optimizers
    Source: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html#starter-example

    example from FP
    https://github.com/courtois-neuromod/video_transformer/blob/main/src/videogpt/vqvae.py
    """

    def __init__(
        self: "VLBLitModule",
        config: VLBLitModuleConfig,
    ) -> None:
        super().__init__()

        self.config: VLBLitModuleConfig = config
        self.nnmodule = load_pretrained_vllama2(self.config)
        #self.hrf_layer
        #self.ridge_layer

        # https://github.com/courtois-neuromod/phantom_LLM/blob/7258a5e95fe256d9ae4669dc5a1ca1be34a0d867/phantom_LLM/src/models/ridge_align.py#L76


    def forward(self, x_video, x_lang, attention_mask=None, hrf_weights=None):
        """."""
        # https://github.com/courtois-neuromod/phantom_LLM/blob/7258a5e95fe256d9ae4669dc5a1ca1be34a0d867/phantom_LLM/src/models/ridge_align.py#L76

        # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414/videollama2/model/videollama2_llama.py#L61
        outputs = self.nnmodule(
            input_ids = x_lang,
            attention_mask = attention_mask,
            output_hidden_states=True,
            images=x_video,
        )
        """
        outputs has two keys: 'logits' and 'hidden_states'
        outputs.logits is a single tensor of dim=(1, 3231, 32000)
        outputs.hidden_states is a tuple of len=33, each item is a tensor of dim=(1, 3230*, 4096), *depends on len of x_lang
        """
        # outputs
        hidden_states = outputs.hidden_states[-1]  # which to pick when there is padding? Just don't use padding and a batch size of 1??

        #hidden_states = outputs.last_hidden_state

        # Apply HRF Convolution
        hrf_embeddings = self.hrf_layer(hidden_states, hrf_weights).squeeze(1)

        # Remove the singleton dimension
        hrf_embeddings = hrf_embeddings.squeeze(-1)
        # print(f"HRF embeddings shape after squeeze: {hrf_embeddings.shape}")

        # Apply Ridge Regression
        regression_output = self.ridge_layer(hrf_embeddings)
        # print(f"Ridge Regression output shape: {regression_output.shape}")

        return regression_output


    def training_step(self, batch):
        # TODO: how to deal w batch structure from data module??
        """
        Note: the DataLoader creates batches based on Dataset's __getitems__
        If it returns tensors, then those are stacked
        If it returns a dictionary, items get stacked under each key, and can
        get called by that key from a batch dictionary. E.g. here "timeseries", "vision", "language"
        """
        #y = batch["timeseries"].cuda()  # dim = (batch_size, 1000,) dtype = torch.float32
        y = batch["timeseries"].to(self.config.dtype).to(self.config.device)  # dim = (batch_size, 1000,) dtype = torch.float32

        # batch["vision"]: # list[tuple] of len == batch size
        # each tuple in the list is (tensor , 'video'),
        # where tensor is (dim = (12, 3, 336, 336), dtype = torch.float32)

        # videollama2 inference
        # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/__init__.py#L32
        # image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        #x_video = torch.tensor(batch["vision"], dtype=torch.long).half().cuda()
        #x_video = [(batch["vision"][i].half().cuda(), "video") for i in range(batch["vision"].shape[0])]
        x_video = [(batch["vision"][i].to(self.config.dtype).to(self.config.device), "video") for i in range(batch["vision"].shape[0])]

        #x_video = batch["vision"].half().cuda()  # dim = (12, 3, 336, 336), dtype = torch.float32
        # TODO: determine if .half() to torch.float16 is just for inference or also for training...
        #x_video = [(x_video, 'video')]

        #x_lang = batch["language"].unsqueeze(0).long().cuda()
        #x_lang = batch["language"].long().cuda()  # tensor dim = (batch_size, num_feat,) dtype = torch.float32
        x_lang = batch["language"].long().to(self.config.device)  # tensor dim = (batch_size, num_feat,) dtype = torch.float32

        pad_idx = int(batch["mask"])  # int
        #x_lang = torch.tensor(
        #    batch["language"],
        #    dtype=torch.long
        #).unsqueeze(0).long().cuda(),

        # from: attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
        #attention_masks = x_lang.ne(tokenizer.pad_token_id).long().cuda()
        # tokenizer.pad_token == tokenizer.unk_token = '<unk>'
        # tokenizer.pad_token_id == 0
        #attention_mask = x_lang.ne(0).long().cuda()
        attention_mask = x_lang.ne(0).long().to(self.config.device)

        output = self.forward(
             x_video,
             x_lang,
             attention_mask = attention_mask,
        )
        # prepare input ids for multimodal
        # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414/videollama2/model/videollama2_arch.py#L161

        # TODO: output loss function

    def configure_optimizers(
        self: "VLBLitModule",
    ) -> tuple[list[Optimizer], list[dict[str, LRScheduler | str | int]]]:
        """.
        see also:
        https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/videogpt/vqvae_ba.py#L165

        Returns:
            A tuple containing the PyTorch ``Optimizer`` and
            ``LRScheduler`` instance attributes (each nested in a
            list).
        """
        # https://hydra.cc/docs/advanced/instantiate_objects/overview/
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW
        self.optimizer = AdamW(
            params=self.parameters(),
            lr = self.config.lr,
            betas = self.config.betas,
            eps = self.config.eps,
            weight_decay = self.config.weight_decay,
        )

        self.lr_scheduler_args = {
            "last_epoch": self.config.last_epoch,
            "T_max": self.config.t_max,
        }
        self.scheduler = getattr(lr_scheduler, self.config.lr_scheduler_name)(
            self.optimizer, **self.lr_scheduler_args
        )

        return [self.optimizer], [
            {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        ]
