import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.distributed as dist

import moco.builder
import moco.loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

DEFAULT_CONFIG = {
    'arch': 'resnet34',
    'moco_dim': 128,
    'moco_k': 65536,
    'moco_m': 0.999,
    'moco_t': 0.07,
    'mlp': False,
    'dist_backend' : 'nccl',
    'dist_url' : 'tcp://localhost:10001',
    'world_size' : 1,
    'rank' : 0

}


def create_model(checkpoint_file: str,
                 arch: str = DEFAULT_CONFIG['arch'],
                 moco_dim: int = DEFAULT_CONFIG['moco_dim'],
                 moco_k: int = DEFAULT_CONFIG['moco_k'],
                 moco_m: float = DEFAULT_CONFIG['moco_m'],
                 moco_t: float = DEFAULT_CONFIG['moco_t'],
                 mlp: bool = DEFAULT_CONFIG['mlp'],
                 gpu: bool = True):
    """
    Create a model from a given checkpoint and config.
    """
    # Initialize Distributed Data Parallel
    if not dist.is_initialized():
        dist.init_process_group(backend= DEFAULT_CONFIG['dist_backend'], init_method=DEFAULT_CONFIG['dist_url'],
                                world_size=DEFAULT_CONFIG['world_size'], rank=DEFAULT_CONFIG['rank'])

    print("=> creating model '{}'".format(arch))
    model = moco.builder.MoCo(
        models.__dict__[arch],
        moco_dim, moco_k, moco_m, moco_t, mlp)
    print(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    print("=> loading checkpoint '{}'".format(checkpoint_file))
    if not gpu:
        checkpoint = torch.load(checkpoint_file)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:0' # TODO : add multiple gpu support if ever needed
        checkpoint = torch.load(checkpoint_file, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'"
          .format(checkpoint_file))

    return model

