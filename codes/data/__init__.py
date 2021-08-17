"""create dataset and dataloader"""
import importlib
import logging
import os
import os.path as osp

import torch
import torch.utils.data

from utils.registry import DATASET_REGISTRY

data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(data_folder)
    if v.endswith("_dataset.py")
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"data.{file_name}") for file_name in dataset_filenames
]


def create_dataloader(dataset, dataset_opt, dist=False):
    phase = dataset_opt["phase"]
    if phase == "train":
        num_workers = dataset_opt["workers_per_gpu"]
        batch_size = dataset_opt["imgs_per_gpu"]
        if dist:
            sampler = torch.utils.data.DistributedSampler(
                dataset, shuffle=True, drop_last=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )
    else:
        if dist:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            indices = list(range(rank, len(dataset), world_size))
            dataset = torch.utils.data.Subset(dataset, indices)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )


def create_dataset(dataset_opt, **kwarg):
    mode = dataset_opt["mode"]
    dataset = DATASET_REGISTRY.get(mode)(dataset_opt, **kwarg)
    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
