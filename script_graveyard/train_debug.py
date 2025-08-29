import functools
import os

import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    size_based_auto_wrap_policy,
    wrap,
)
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet152


def setup():
    # initialise the process group
    dist.init_process_group("nccl")  # of gloo


def main(args):
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])

    setup()

    imagenet_train_data = ImageNet(args.data_dir, split='train', transform=transform)
    train_sampler = DistributedSampler(imagenet_train_data, rank=rank, num_replicas=world_size, shuffle=True)
    train_workers = 0
    train_loader = torch.utils.DataLoader(imagenet_train_data,
                                          batch_size=1,  # batch per rank
                                          sampler=train_sampler,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=train_workers)

    torch.cuda.set_device(local_rank)
    resnet_auto_wrap_policy = functools.partial(
        size_baesed_auto_wrap_policy,
        min_num_params=1000,
    )

    model = resnet152()
    torch.compile(model)

    model = FSDP(model,
                 auto_wrap_policy = resnet_auto_wrap_policy,
                 device_id=torch.cuda.cuddent_device())

    print(model)

    #trainer = Trainer(gpus=4, num_nodes=2, accelerator='fsdp')
