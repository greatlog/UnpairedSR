import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from metrics import IQA
from models import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--opt", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "--root_path",
        help="experiment configure file name",
        default="../../../",
        type=str,
    )
    # distributed training
    parser.add_argument("--gpu", help="gpu id for multiprocessing training", type=str)
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    return args


def setup_dataloaer(opt):

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, opt["dist"])
            total_iters = opt["train"]["niter"]
            total_epochs = total_iters // (len(train_loader) - 1) + 1
            if rank == 0:
                print(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), len(train_loader)
                    )
                )
                print(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, opt["train"]["niter"]
                    )
                )

        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt["dist"])
            if rank == 0:
                print(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

    assert train_loader is not None
    assert val_loader is not None

    return train_set, train_loader, val_set, val_loader, total_iters, total_epochs


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    opt["dist"] = args.world_size > 1

    if opt["dist"]:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt, args))
    else:
        main_worker(0, 1, opt, args)


def main_worker(gpu, ngpus_per_node, opt, args):

    if opt["dist"]:
        if args.dist_url == "env://" and args.rank == -1:
            rank = int(os.environ["RANK"])

        rank = args.rank * ngpus_per_node + gpu
        print(
            f"Init process group: dist_url: \
            {args.dist_url}, world_size: {args.world_size}, rank: {rank}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )

        torch.cuda.set_device(gpu)

    else:
        rank = 0

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create dataset
    (
        train_set,
        train_loader,
        val_set,
        val_loader,
        total_iters,
        total_epochs,
    ) = setup_dataloaer(opt)

   
    current_step = 0
    start_epoch = 0

    print(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in tqdm(enumerate(train_loader)):
            current_step += 1
            # validation
            if current_step % opt["train"]["val_freq"] == 0:
                print("Validation")
                validate(
                    val_set, val_loader, opt, epoch, current_step
                )
        print(f"epoch: {epoch}, iter: {current_step}")

def validate(dataset, dist_loader, opt, epoch, current_step):

    test_results = {}
    for metric in opt["metrics"]:
        test_results[metric] = torch.zeros((len(dataset))).cuda()

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    indices = list(range(rank, len(dataset), world_size))
    for (
        idx,
        val_data,
    ) in enumerate(dist_loader):
        idx = 0


if __name__ == "__main__":
    main()
