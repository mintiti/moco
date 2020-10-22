import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
    'rank' : 0,
    'aug_plus': False,
    'batch_size' : 32,
    'workers' : 8 #adjust to cpu capabilities
}


def create_model(checkpoint_file: str,
                 arch: str = DEFAULT_CONFIG['arch'],
                 moco_dim: int = DEFAULT_CONFIG['moco_dim'],
                 moco_k: int = DEFAULT_CONFIG['moco_k'],
                 moco_m: float = DEFAULT_CONFIG['moco_m'],
                 moco_t: float = DEFAULT_CONFIG['moco_t'],
                 mlp: bool = DEFAULT_CONFIG['mlp'],
                 gpu: bool = True) -> torch.nn.Module:
    """
    Create a model from a given checkpoint and config.
    checkpoint.keys() should contain the following keys :
        - 'epoch' : (int) The epoch at which the training was stopped. e.g. 256
        - 'arch' : (str) The encoder architecture used. e.g. 'resnet34'
        - 'state_dict' : The PyTorch state_dict for the nn.Module
        - 'optimizer' : The PyTorch optimizer state
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

def create_train_val_loader(folder,
                            train_fraction = 0.8, is_structured = True,aug_plus = DEFAULT_CONFIG['aug_plus'],
                            batch_size = DEFAULT_CONFIG['batch_size'],
                            num_workers = DEFAULT_CONFIG['workers']) -> torch.utils.data.DataLoader:
    """Loads a folder for training and validation
    The folder can either be structured or unstructered :
    - structured :
        root
        |
        |--> cls1
        |   sample1
        |   sample2
        |--> cls2
        |   sample1
        |   sample2
        .
        .
        |--> clsN
        |   sample1
        |   sample2
    - unstructured ;
    root
    |   sample1
    |   sample2
    |   .
    |   .
    |   .
    |   sampleN

    ----------------------------------------------------
    args:
        - folder : (str) Folder containing the dataset.
        - is_structured : (bool) whether the folder is structured or not

"""
    # TODO : implement unstructured folder when needed
    datadir = os.path.join(folder,'train')

    # Data transforms
    # TODO : change the values to data-relevant mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if aug_plus :
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # Create the torch data loading pipeline
    dataset = datasets.ImageFolder(
        datadir,
        moco.loader.TwoCropsTransform(augmentation)
    )
    # Split the dataset into
    train_data_len = int(len(dataset) * train_fraction)
    val_data_len = len(dataset) - train_data_len
    split = [train_data_len, val_data_len]
    train_set, val_set = torch.utils.data.random_split(dataset, split)

    # Change the transforms for the val set
    val_set.transform = transforms.Compose([
        transforms.resize(256),
        transforms.CenterCrop(224),
        normalize
    ])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size= batch_size, shuffle= False, num_workers = num_workers, pin_memory= True
    )

    return train_loader, val_loader








