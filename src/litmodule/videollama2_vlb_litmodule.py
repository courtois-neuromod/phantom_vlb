#import argparse
#import math
import os
import sys
from dataclasses import dataclass

#import numpy as np
import torch
import torch.nn.functional as F

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

from src import (
    HRFConvolveLayer,
    RidgeRegressionLayer,
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
        #device=config.device,
        low_cpu_mem_usage=True,
        local_files_only=True,
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
    dropout_rate: float
    num_target: int
    l2_lambda: float
    lr: float
    betas: list[float]
    eps: float
    weight_decay: float
    lr_scheduler_name: str
    last_epoch: int
    t_max: int

    def __post_init__(self):
        self.dtype = torch.bfloat16  # torch.bfloat16 for newer GPUs
        #self.device = "cpu"
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


    def make_weight_mask(self, pad_vals, vis_weights, lang_weights, lang_len, max_len):

        feature_len = (vis_weights.shape[1]*13*13) + lang_len - 1
        assert feature_len == max_len # padded so text + vis == 2048
        
        weight_batch = []
        for i in range(pad_vals.shape[0]):

            pad_len, inst_len, dialog_len = pad_vals[i]
            #pad_len = pad_vals[i][0].item()
            #inst_len = pad_vals[i][1].item()
            #dialog_len = pad_vals[i][2].item()

            trial_weights = torch.cat([
                vis_weights[i].repeat_interleave(13*13).to(self.config.dtype),
                torch.zeros(2 + inst_len, device=self.device).to(self.config.dtype),
                lang_weights[i][:dialog_len].to(self.config.dtype),
                torch.zeros(4 + pad_len, device=self.device).to(self.config.dtype),
            ], dim=0)

            pad_left = feature_len - trial_weights.shape[0]
            weight_batch.append(
                torch.cat([
                    torch.zeros(pad_left, device=self.device).to(self.config.dtype),
                    trial_weights,
                ], dim=0).unsqueeze(dim=0)
            )

        return torch.cat(weight_batch, dim=0)


    def configure_model(
        self: "VLBLitModule",
    ) -> None:
        """
        When training model on multiple GPUs with FSDP with lightening,
        do not instantiate large layers in __init__() .
        Instead, delay the creation of large layers to the configure_model() hook.
        Source
        https://lightning.ai/docs/pytorch/stable/advanced/model_init.html#model-parallel-training-fsdp-and-deepspeed
        """
        # Debugging distributed environment
        #print(f"[{os.getpid()}] Rank: {os.getenv('RANK', 'N/A')}")
        #print(f"[{os.getpid()}] Local Rank: {os.getenv('LOCAL_RANK', 'N/A')}")
        #print(f"[{os.getpid()}] World Size: {os.getenv('WORLD_SIZE', 'N/A')}")
        #print(f"[{os.getpid()}] Master Addr: {os.getenv('MASTER_ADDR', 'N/A')}")
        #print(f"[{os.getpid()}] Master Port: {os.getenv('MASTER_PORT', 'N/A')}")
        #print(f"[{os.getpid()}] CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'N/A')}")

        # Check if distributed is initialized (will be False if not)
        #print(f"[{os.getpid()}] torch.distributed.is_initialized(): {dist.is_initialized()}")
        #if dist.is_initialized():
        #    print(f"[{os.getpid()}] dist.get_rank(): {dist.get_rank()}")
        #    print(f"[{os.getpid()}] dist.get_world_size(): {dist.get_world_size()}")
        #    print(f"[{os.getpid()}] Current device: {torch.cuda.current_device()}")

        self.nnmodule = load_pretrained_vllama2(self.config)
        kwargs = {
            #"device": self.config.device,
            "dtype": self.config.dtype,
        }

        # https://github.com/courtois-neuromod/phantom_LLM/blob/7258a5e95fe256d9ae4669dc5a1ca1be34a0d867/phantom_LLM/src/models/ridge_align.py#L76
        self.hrf_layer = HRFConvolveLayer()
        self.ridge_layer = RidgeRegressionLayer(
            self.nnmodule.config.hidden_size,
            self.config.num_target,  # brain target voxel count or parcel count
            self.config.l2_lambda,
            **kwargs,
        )
        self.layer_norm1 = torch.nn.LayerNorm(self.nnmodule.config.hidden_size, **kwargs)  # embedding dim == 4096 for vllama2
        self.layer_norm2 = torch.nn.LayerNorm(self.nnmodule.config.hidden_size, **kwargs)  # embedding dim == 4096 for vllama2
        self.dropout = torch.nn.Dropout(self.config.dropout_rate)


    def forward(self, x_video, x_lang, weight_mask, attention_mask=None):
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
        outputs.logits is a single tensor of dim=(batch_size, 3230, 32000) where 32000 is vocab size, 3230 is input sequence lenght
        outputs.hidden_states is a tuple of len=33 (number of layers), each item is a tensor of dim=(batch_size, 3230*, 4096), *depends on len of x_lang; 4096 is hidden dim

        connector output (temporally agregated video features) are of dim torch.Size([1, 1183, 4096])
        where 4096 is hidden dim size, and 1183 is the sequence lenght for 12 frames of 336x336 video input  (169*7 = 1183...)
        """
        # outputs (from last hidden layer)
        hidden_states = self.layer_norm1(outputs.hidden_states[-1])

        # video_gpt: pca, ridge on brain encooding (but not w gradient descent)
        # https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/scripts/apply_gpt_ridge_encoding.py#L96

        # Apply HRF Convolution
        # TODO: figure out squeezing, output dim... (depends batch size)
        hrf_embeddings = self.dropout(
            self.layer_norm2(
                self.hrf_layer(hidden_states, weight_mask,
                )#.squeeze(1)
        ))

        # Remove the singleton dimension (NEEDED?)
        #hrf_embeddings = hrf_embeddings.squeeze(-1)
        # print(f"HRF embeddings shape after squeeze: {hrf_embeddings.shape}")

        # Apply Ridge Regression
        regression_output, l2_reg = self.ridge_layer(hrf_embeddings)

        return regression_output, l2_reg


    def training_step(self, batch):
        """
        Note: the DataLoader creates batches based on the Dataset's __getitems__
        If it returns tensors, then those are stacked by the Datamodule

        If it returns a dictionary, items get stacked under each given key, and can
        get called by that key from a batch dictionary. E.g. here "timeseries", "vision", "language"
        """
        # Adapted from videollama2 inference
        # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/__init__.py#L32
        # image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        # From VLLaMA2 code: prepare input ids for multimodal
        # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414/videollama2/model/videollama2_arch.py#L161
        #x_video = [(batch["vision"][i].to(self.config.dtype).to(self.config.device), "video") for i in range(batch["vision"].shape[0])]
        #x_video = [(batch["vision"][i].to(self.config.dtype).to(self.device), "video") for i in range(batch["vision"].shape[0])]
        x_video = [(batch["vision"][i].to(self.config.dtype), "video") for i in range(batch["vision"].shape[0])]

        #x_lang = batch["language"].long().to(self.config.device)  # tensor dim = (batch_size, num_feat,) dtype = torch.float32
        #attention_mask = x_lang.ne(0).long().to(self.config.device)
        #x_lang = batch["language"].long().to(self.device)  # tensor dim = (batch_size, num_feat,) dtype = torch.float32
        x_lang = batch["language"].long()

        #attention_mask = x_lang.ne(0).long().to(self.device)
        attention_mask = x_lang.ne(0).long()

        weight_mask = self.make_weight_mask(
            batch["padvals"],
            batch["vis_weights"],
            batch["lang_weights"],
            x_lang.shape[1],
            self.nnmodule.config.tokenizer_model_max_length,
        )#.to(self.device)
        #).to(self.config.device)

        brain_encoding, l2_reg = self.forward(
             x_video,
             x_lang,
             weight_mask,
             attention_mask = attention_mask,
        )

        #y = batch["timeseries"].to(self.config.dtype).to(self.config.device)  # dim = (batch_size, 1000,) dtype = torch.float32
        y = batch["timeseries"].to(self.config.dtype)#.to(self.device)  # dim = (batch_size, 1000,) dtype = torch.float32

        """
        Implement loss function...
        From phantom_LLM: two alternatives, cosine similarity loss and mean square error
        https://github.com/courtois-neuromod/phantom_LLM/blob/5505873e190b4b4b1c8103daf02d68fa37e0156e/phantom_LLM/src/run_training_brain_corpus.py#L160

        brain_loss = (1 - F.cosine_similarity(brain_encoding, y, dim=-1).mean()) + l2_reg
        brain_loss = torch.nn.MSELoss(brain_encoding, y) + l2_reg

        From video_gpt:
        https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/videogpt/vqvae_ba.py#L121
        brain_loss = F.mse_loss(brain_encoding, y) + l2_reg
        """
        brain_loss = F.mse_loss(brain_encoding, y) + l2_reg

        self.log("train/brain_loss", brain_loss)

        return brain_loss


    def validation_step(self, batch):
        """."""
        x_video = [
            #(batch["vision"][i].to(self.config.dtype).to(self.config.device), "video") for i in range(batch["vision"].shape[0])
            #(batch["vision"][i].to(self.config.dtype).to(self.device), "video") for i in range(batch["vision"].shape[0])
            (batch["vision"][i].to(self.config.dtype), "video") for i in range(batch["vision"].shape[0])
        ]

        #x_lang = batch["language"].long().to(self.config.device)
        #attention_mask = x_lang.ne(0).long().to(self.config.device)
        x_lang = batch["language"].long()#.to(self.device)
        attention_mask = x_lang.ne(0).long()#.to(self.device)

        weight_mask = self.make_weight_mask(
            batch["padvals"],
            batch["vis_weights"],
            batch["lang_weights"],
            x_lang.shape[1],
            self.nnmodule.config.tokenizer_model_max_length,
        )#.to(self.device)
        #).to(self.config.device)

        brain_encoding, l2_reg = self.forward(
             x_video,
             x_lang,
             weight_mask,
             attention_mask = attention_mask,
        )

        #y = batch["timeseries"].to(self.config.dtype).to(self.config.device)
        y = batch["timeseries"].to(self.config.dtype)#.to(self.device)
        brain_loss = F.mse_loss(brain_encoding, y) + l2_reg
        self.log("val/brain_loss", brain_loss)

        return brain_loss


    def configure_optimizers(
        self: "VLBLitModule",
    ):
        """.
        see also:
        https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/src/videogpt/vqvae_ba.py#L165

        Returns:
            A tuple containing the PyTorch ``Optimizer`` and
            ``LRScheduler`` instance attributes (each nested in a
            list).
        """
        #print(f"Rank: {os.getenv('RANK', 'N/A')}, Local Rank: {os.getenv('LOCAL_RANK', 'N/A')}")
        #print(f"World Size: {os.getenv('WORLD_SIZE', 'N/A')}")
        #print(f"Master Address: {os.getenv('MASTER_ADDR', 'N/A')}, Master Port: {os.getenv('MASTER_PORT', 'N/A')}")

        # https://hydra.cc/docs/advanced/instantiate_objects/overview/
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW
        self.optimizer = AdamW(
            #params=self.parameters(),
            params = filter(lambda p: p.requires_grad, self.parameters()),
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
