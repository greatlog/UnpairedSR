import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from metrics import IQA
from models import create_model
from utils import bgr2ycbcr, imresize


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
    opt = option.parse(args.opt, args.root_path, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    opt["dist"] = args.world_size > 1

    util.mkdirs(
        (path for key, path in opt["path"].items() if not key == "experiments_root")
    )

    os.system("rm ./result")
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

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
            f"Init process group: dist_url: {args.dist_url}, world_size: {args.world_size}, rank: {rank}"
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

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO,
        screen=True,
        tofile=True,
    )

    measure = IQA(metrics=opt["metrics"], cuda=True)

    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_datasets = []
    test_loaders = []

    for phase, dataset_opt in sorted(opt["datasets"].items()):

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt["dist"])

        if rank == 0:
            logger.info(
                "Number of test images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(test_set)
                )
            )
        test_datasets.append(test_set)
        test_loaders.append(test_loader)

    # load pretrained model by default
    model = create_model(opt)

    for test_dataset, test_loader in zip(test_datasets, test_loaders):

        test_set_name = test_dataset.opt["name"]
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)

        if rank == 0:
            logger.info("\nTesting [{:s}]...".format(test_set_name))
            util.mkdir(dataset_dir)

        validate(
            model,
            test_dataset,
            test_loader,
            opt,
            measure,
            dataset_dir,
            test_set_name,
            logger,
        )


def validate(
    model, dataset, dist_loader, opt, measure, dataset_dir, test_set_name, logger
):

    test_results = {}
    test_results_y = {}
    for metric in opt["metrics"]:
        test_results[metric] = torch.zeros((len(dataset))).cuda()
        test_results_y[metric] = torch.zeros((len(dataset))).cuda()

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    indices = list(range(rank, len(dataset), world_size))
    for (
        idx,
        test_data,
    ) in enumerate(dist_loader):
        idx = indices[idx]

        img_path = test_data["src_path"][0]
        img_name = img_path.split("/")[-1].split(".")[0]

        model.test(test_data)
        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals["sr"])  # uint8
       
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(sr_img, save_img_path)

        message = "img:{:15s}; ".format(img_name)

        crop_border = opt["crop_border"] if opt["crop_border"] else opt["scale"]

        if crop_border == 0:
            cropped_sr_img = sr_img
        else:
            cropped_sr_img = sr_img[
                crop_border:-crop_border, crop_border:-crop_border, :
            ]

        if "tgt" in test_data.keys():
            gt_img = util.tensor2img(test_data["tgt"][0].double().cpu())

            if crop_border == 0:
                cropped_gt_img = gt_img
            else:
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border, :
                ]
        else:
            cropped_gt_img = None
        
        message += "Scores - "
        scores = measure(res=cropped_sr_img, ref=cropped_gt_img, metrics=opt["metrics"])
        for k, v in scores.items():
            test_results[k][idx] = v
            message += "{}: {:.6f}; ".format(k, v)

        if sr_img.shape[2] == 3:  # RGB image
            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
            if crop_border == 0:
                cropped_sr_img_y = sr_img_y * 255
            else:
                cropped_sr_img_y = (
                    sr_img_y[crop_border:-crop_border, crop_border:-crop_border] * 255
                )
            if gt_img is not None:
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_gt_img_y = gt_img_y * 255
                else:
                    cropped_gt_img_y = (
                        gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                        * 255
                    )
            else:
                gt_img_y = None

            message += "Y Scores - "
            scores = measure(
                res=cropped_sr_img_y, ref=cropped_gt_img_y, metrics=opt["metrics"]
            )
            for k, v in scores.items():
                test_results_y[k][idx] = v
                message += "{}: {:.6f}; ".format(k, v)

        logger.info(message)

    if opt["dist"]:
        for k, v in test_results.items():
            dist.reduce(v, dst=0)
        dist.barrier()

        for k, v in test_results_y.items():
            dist.reduce(v, dst=0)
        dist.barrier()

    # log
    avg_results = {}
    message = "Average Results for {}\n".format(test_set_name)

    if rank == 0:
        for k, v in test_results.items():
            avg_results[k] = sum(v) / len(v)
            message += "{}: {:.6f}; ".format(k, avg_results[k])

        logger.info(message)

    avg_results_y = {}
    message = "Average Results on Y channel for {}\n".format(test_set_name)

    if rank == 0:
        for k, v in test_results_y.items():
            avg_results[k] = sum(v) / len(v)
            message += "{}: {:.6f}; ".format(k, avg_results[k])

        logger.info(message)


if __name__ == "__main__":
    main()
