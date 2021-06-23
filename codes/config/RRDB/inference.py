import argparse
import logging
import math
import os
import os.path as osp
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


#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument("-input_dir", type=str, default="../../../data_samples/LR")
parser.add_argument("-output_dir", type=str, default="../../../data_samples/DANv1_SR")
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

    sr = model.fake_SR.detach().float().cpu()[0]
    sr_im = util.tensor2img(sr)

    save_path = osp.join(args.output_dir, "{}_x{}.png".format(name, opt["scale"]))
    cv2.imwrite(save_path, sr_im)
