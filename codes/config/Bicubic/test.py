import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from IPython import embed

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from metrics import IQA
from models import create_model
from utils import bgr2ycbcr, imresize


#### options
parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, required=True, help="Path to options YMAL file.")
parser.add_argument(
    "--root_path",
    help="experiment configure file name",
    default="../../../",
    type=str,
)

args = parser.parse_args()
opt = option.parse(args.opt, args.root_path, is_train=False)
opt = option.dict_to_nonedict(opt)

measure = IQA(metrics=opt["metrics"], cuda=True)

#### mkdir and logger
util.mkdirs(
    (path for key, path in opt["path"].items() if not key == "experiments_root")
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)

logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = defaultdict(list)
    test_results_y = defaultdict(list)

    for test_data in test_loader:

        img_path = test_data["src_path"][0]
        img_name = img_path.split("/")[-1].split(".")[0]
        
        sr_img = imresize(test_data["src"], opt["scale"])
        sr_img = util.tensor2img(sr_img)

#        model.test(test_data["src"])
#        visuals = model.get_current_visuals()
#        sr_img = util.tensor2img(visuals["sr"])  # uint8
#
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
            test_results[k].append(v)
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
                test_results_y[k].append(v)
                message += "{}: {:.6f}; ".format(k, v)

        logger.info(message)

    ave_results = {}
    message = "Average Results for {}\n".format(test_set_name)
    for k, v in test_results.items():
        ave_result = sum(v) / len(v)
        ave_results[k] = ave_result
        message += "{}: {:.6f}\t".format(k, ave_result)
    logger.info(message)

    if len(test_results_y) > 0:
        ave_results = {}
        message = "Average Results on Y channel for {}\n".format(test_set_name)
        for k, v in test_results_y.items():
            ave_result = sum(v) / len(v)
            ave_results[k] = ave_result
            message += "{}: {:.6f}\t".format(k, ave_result)
        logger.info(message)
