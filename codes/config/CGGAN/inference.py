import argparse
import logging
import math
import os
import os.path as osp
import random
import sys
import cv2
from collections import defaultdict
from glob import glob
from tqdm import tqdm

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


#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "--opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument("--input_dir", type=str, default="/mnt/hdd/lzx/SRDatasets/NTIRE2020/track1/track1_test_input/")
parser.add_argument("--output_dir", type=str, default="results/2020Track1Test/")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

test_files = glob(osp.join(args.input_dir, "*png"))
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split(".")[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()

    model.test(img_t)
    visuals = model.get_current_visuals()

    sr_im = util.tensor2img(visuals["sr"])


    save_path = osp.join(args.output_dir, "{}_x{}.png".format(name, opt["scale"]))
    cv2.imwrite(save_path, sr_im)
