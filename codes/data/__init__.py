"""create dataset and dataloader"""
import importlib
import logging
import os
import os.path as osp
import numpy as np
import random
from functools import partial

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

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


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def create_dataloader(dataset, dataset_opt, dist=False):
    phase = dataset_opt["phase"]
    if phase == "train":
        num_workers = dataset_opt["workers_per_gpu"]
        batch_size = dataset_opt["imgs_per_gpu"]
        if dist:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            sampler = torch.utils.data.DistributedSampler(
                dataset, shuffle=True, drop_last=True, rank=rank, world_size=world_size
            )
        else:
            rank = 0
            world_size = 1
            sampler = None
        return DataLoaderX(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=num_workers,
            worker_init_fn=partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=rank),
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
#            prefetch_factor=4
        )
    else:
        if dist:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            indices = list(range(rank, len(dataset), world_size))
            dataset = torch.utils.data.Subset(dataset, indices)

        return DataLoaderX(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
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

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
