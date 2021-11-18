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
from glob import glob

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
        default="tcp://127.0.0.1:2345",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    return args

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

    if opt["train"].get("resume_state", None) is None:
        util.mkdir_and_rename(
            opt["path"]["experiments_root"]
        )  # rename experiment folder if exists
        util.mkdirs(
            (path for key, path in opt["path"].items() if not key == "experiments_root")
        )
        os.system("rm ./log")
        os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

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

    seed = opt["train"]["manual_seed"]
    if seed is None:
        util.set_random_seed(rank)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # setup tensorboard and val logger
    if rank == 0:
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))

        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )

    measure = IQA(metrics=opt["metrics"], cuda=True)

    # config loggers. Before it, the log will not work
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO if rank == 0 else logging.ERROR,
        screen=True,
        tofile=True,
    )

    logger = logging.getLogger("base")
    if rank == 0:
        logger.info(option.dict2str(opt))
    
    # create dataset
    for name, dataset_opt in opt["datasets"].items():

        img_list = sorted(glob(os.path.join(dataset_opt["dataroot_src"], "*.png")))

        for img_path in img_list:
            # create model
            model = create_model(opt)

            img_name = img_path.split("/")[-1].split(".")[0]
            if rank == 0:
                logger.info(f"Testing {img_name} in {dataset_opt['name']}")

            dataset = create_dataset(dataset_opt, img_path=img_path)
            dataloader = create_dataloader(dataset, dataset_opt, opt["dist"])

            total_iters = opt["train"]["niter"]
            train_size = len(dataset) // dataset_opt["imgs_per_gpu"] - 1
            total_epochs = total_iters // train_size + 1

            decay_lr_prop = opt["train"]["decay_lr_prop"]

            data_time, iter_time = time.time(), time.time()
            avg_data_time = avg_iter_time = 0
            count = 0
            current_step = 0
            for epoch in range(total_epochs):
                for train_data in dataloader:

                    current_step += 1
                    count += 1
                    if current_step > total_iters:
                        break
                    
                    data_time = time.time() - data_time
                    avg_data_time = (avg_data_time * (count - 1) + data_time) / count
                    
                    model.feed_data(train_data)
                    model.optimize_parameters(current_step)
                    
                    factor = (
                        (total_iters - decay_lr_prop * current_step) / 
                        (total_iters - decay_lr_prop * current_step + decay_lr_prop)
                    )
                    for optimizer in model.optimizers.values():
                        for param_grop in optimizer.param_groups:
                            param_grop["lr"] *= factor
                    
                    iter_time = time.time() - iter_time
                    avg_iter_time = (avg_iter_time * (count - 1) + iter_time) / count
                    
                    # log
                    if current_step % opt["logger"]["print_freq"] == 0:
                        logs = model.get_current_log()
                        message = (
                            f"<iter:{current_step:8,d}, "
                            f"lr:{model.get_current_learning_rate():.3e}> "
                        )

                        message += f'[time (data): {avg_iter_time:.3f} ({avg_data_time:.3f})] '
                        for k, v in logs.items():
                            message += "{:s}: {:.4e}; ".format(k, v)
                            # tensorboard logger
                            if opt["use_tb_logger"] and "debug" not in opt["name"]:
                                if rank == 0:
                                    tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)
                    
                    if current_step % opt["train"]["val_freq"] == 0 and rank == 0:
                        scores = validate(model, dataset.img_src, img_name, opt, measure, current_step)
                        message = "iter:{:8,d} ".format(current_step)
                        for k, v in scores.items():
                            message += f"{k:s}: {v:.6f}; "
                        logger.info(message)

                    data_time = iter_time = time.time()

            del model
            torch.cuda.empty_cache()

            
def validate(model, img, img_name, opt, measure, current_step):
    
    test_img = np.ascontiguousarray(img.transpose(2, 0, 1)[[2, 1, 0]])
    test_img = torch.FloatTensor(test_img[None])

    val_data = {"src": test_img}

    model.test(val_data)
    visuals = model.get_current_visuals()

    # Save images for reference
    img_dir = os.path.join(opt["path"]["val_images"], img_name)

    util.mkdir(img_dir)
    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))
    util.save_img(util.tensor2img(test_img), save_lr_path)

    sr_img = util.tensor2img(visuals["sr"])  # uint8
    save_img_path = os.path.join(
        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
    )
    util.save_img(sr_img, save_img_path)

    # calculate scores
    crop_size = opt["scale"]
    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
    if "tgt" in val_data.keys():
        gt_img = util.tensor2img(val_data["tgt"])
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
    else:
        cropped_gt_img = gt_img = None

    scores = measure(res=cropped_sr_img, ref=cropped_gt_img, metrics=opt["metrics"])

    torch.cuda.empty_cache()
    return scores

if __name__ == "__main__":
    main()
