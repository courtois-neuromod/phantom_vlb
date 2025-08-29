import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightning.pytorch import LightningModule
from peft import LoraConfig, get_peft_model
#from torchmetrics import PearsonCorrCoef
from torch.optim import Adam, AdamW, lr_scheduler
from transformers import (
    AutoConfig,
)

sys.path.append('../../')

from VideoLLaMA2.videollama2.model.videollama2_mistral import (
    Videollama2MistralForCausalLM,
)

from src import (
    HRFConvolveLayer,
    RidgeRegressionLayer,
)

"""
NUM_FRAMES must be compatible with the preprocessing & lazyloading
here, 3TRs of input window, and 4 frames per TR = 12 frames

Note how this is higher than the default of 8 in vllama2's CONSTANTS
"""
NUM_FRAMES = 12 


def find_all_linear_names(model):
    """
    Source: taken from
    https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414/videollama2/videollama2_trainer.py#L75C1-L88C35
    
    Can't do straight import from videollama2 code base ATM due to library incompatibilities 
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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
    model_config._attn_implementation = "flash_attention_2"  # "flash_attention_2", "sdpa", None

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
        model.requires_grad_(False)
        model.model.requires_grad_(False)  # only freezes the llm backbone but not the mm_projector
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_projector_lr = None
    model.config.num_frames = NUM_FRAMES

    """
    Freeze the vision tower
    The pre-trained vision_tower is not trainable in VideoLLaMA2 framework
    """
    model.get_model().vision_tower.requires_grad_(False)

    if config.use_lora:
        """
        default vllama2 params
            lora_r: int = 64
            lora_alpha: int = 16
            lora_dropout: float = 0.05

        from their finetune_lora.sh script
        --lora_r 128 --lora_alpha 256

        https://huggingface.co/docs/peft/en/package_reference/peft_types
        """
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION", 
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=find_all_linear_names(model),
        )
        model = get_peft_model(model, lora_config)

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
    use_lora: bool
    lora_r: int | None
    lora_alpha: int | None
    lora_dropout: float | None
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
        assert feature_len == max_len
        
        weight_batch = []
        for i in range(pad_vals.shape[0]):

            pad_len, inst_len, dialog_len = pad_vals[i]

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
        """."""
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
        hidden_states = self.layer_norm1(outputs.hidden_states[-1])

        hrf_embeddings = self.dropout(
            self.layer_norm2(
                self.hrf_layer(hidden_states, weight_mask,
                )
        ))

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
        x_video = [(batch["vision"][i].to(self.config.dtype), "video") for i in range(batch["vision"].shape[0])]

        x_lang = batch["language"].long()

        attention_mask = x_lang.ne(0).long()

        weight_mask = self.make_weight_mask(
            batch["padvals"],
            batch["vis_weights"],
            batch["lang_weights"],
            x_lang.shape[1],
            self.nnmodule.config.tokenizer_model_max_length,
        )

        brain_encoding, l2_reg = self.forward(
             x_video,
             x_lang,
             weight_mask,
             attention_mask = attention_mask,
        )

        y = batch["timeseries"].to(self.config.dtype)

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
            (batch["vision"][i].to(self.config.dtype), "video") for i in range(batch["vision"].shape[0])
        ]

        x_lang = batch["language"].long()
        attention_mask = x_lang.ne(0).long()

        weight_mask = self.make_weight_mask(
            batch["padvals"],
            batch["vis_weights"],
            batch["lang_weights"],
            x_lang.shape[1],
            self.nnmodule.config.tokenizer_model_max_length,
        )

        brain_encoding, l2_reg = self.forward(
             x_video,
             x_lang,
             weight_mask,
             attention_mask = attention_mask,
        )

        y = batch["timeseries"].to(self.config.dtype)
        brain_loss = F.mse_loss(brain_encoding, y) + l2_reg

        self.log("val/brain_loss", brain_loss)

        return {
            'loss': brain_loss,
            'brain_preds': brain_encoding.detach(),
            'brain_vals': y.detach(),
        }


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
        self.optimizer = AdamW(
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
