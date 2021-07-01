import argparse
import logging
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
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

        rank = rank * ngpus_per_node + gpu
        print(
            "Init process group: dist_url: {}, world_size: {}, rank: {}".format(
                args.dist_url, args.world_size, rank
            )
        )

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )
    else:
        rank = 0

    seed = opt["train"]["manual_seed"]
    if seed is None:
        util.set_random_seed(rank)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### loading resume state if exists
    if opt["train"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = gpu
        resume_state = torch.load(
            opt["train"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
    else:
        resume_state = None

    #### mkdir and loggers
    if rank == 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        measure = IQA(metrics=opt["metrics"], cuda=True)

        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))

    # config loggers. Before it, the log will not work
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO,
        screen=(rank == 0),
        tofile=True,
    )

    logger = logging.getLogger("base")
    if rank == 0:
        logger.info(option.dict2str(opt))

    #### create train and val dataloader
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)

            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))

            if opt["dist"]:
                train_sampler = DistIterSampler(train_set, world_size, rank)
                total_epochs = int(math.ceil(total_iters / train_size))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )

        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank == 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            model.forward(train_data, current_step)
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e}; ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank == 0:
                            tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if rank == 0 and current_step % opt["train"]["val_freq"] == 0:
                test_results = defaultdict(list)
                for _, val_data in enumerate(val_loader):

                    LR_img = val_data["src"]
                    lr_img = util.tensor2img(LR_img)  # save LR image for reference

                    model.test(val_data["src"])
                    visuals = model.get_current_visuals()

                    # Save images for reference
                    img_name = val_data["src_path"][0].split("/")[-1].split(".")[0]
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)

                    util.mkdir(img_dir)
                    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))
                    util.save_img(lr_img, save_lr_path)

                    sr_img = util.tensor2img(visuals["sr"])  # uint8
                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    util.save_img(sr_img, save_img_path)

                    # calculate scores
                    crop_size = opt["scale"]
                    cropped_sr_img = sr_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    if "tgt" in val_data.keys():
                        gt_img = util.tensor2img(val_data["tgt"])
                        cropped_gt_img = gt_img[
                            crop_size:-crop_size, crop_size:-crop_size, :
                        ]
                    else:
                        cropped_gt_img = gt_img = None

                    scores = measure(
                        res=cropped_sr_img, ref=cropped_gt_img, metrics=opt["metrics"]
                    )
                    for k, v in scores.items():
                        test_results[k].append(v)

                # log
                avg_results = {}
                message = " <epoch:{:3d}, iter:{:8,d}, Average sccores:\t".format(
                    epoch, current_step
                )
                for k, v in test_results.items():
                    avg_results[k] = sum(v) / len(v)
                    message += "{}: {:.6f}; ".format(k, avg_results[k])

                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(message)
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    for k, v in avg_results.items():
                        tb_logger.add_scalar(k, v, current_step)

            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank == 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank == 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of training.")
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger.close()


if __name__ == "__main__":
    main()
