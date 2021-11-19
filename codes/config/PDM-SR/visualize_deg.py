import argparse
import logging
import os.path
import sys
import time
import cv2
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
from glob import glob

import torch

sys.path.append("../../")
import utils as util
import utils.option as option
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
  
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    model = create_model(opt)
    model.set_network_state(["netDeg", "netSR"], "eval")

    img_list = glob("/home/lzx/SRDatasets/DIV2K_valid/HR/x4/*.png")

    for i in range(100):

        img = cv2.imread(img_list[i])
        H, W, C = img.shape
        img = img.transpose(2, 0, 1)[None] / 255
        img = torch.FloatTensor(img).cuda()

        x, kernel, noise = model.netDeg(img)
        
        if x is not None:
            x_np = x.view(C, H//4, W//4).permute(1, 2, 0).detach().cpu().numpy() * 255.0
            cv2.imwrite(f"log/vis/img/{i}.png", x_np)

        if kernel is not None:
            ksize = kernel.shape[1]
            kernel_np = kernel.view(ksize, ksize).detach().cpu().numpy()
            kernel_np = kernel_np / np.max(kernel_np) * 255.0
            cv2.imwrite(f"log/vis/kernel/{i}.png", kernel_np)

        if noise is not None:
            noise_np = noise.view(3, H//4, W//4).permute(1,2,0).detach().cpu().numpy()
            noise_np = (noise_np - np.min(noise_np)) / (np.max(noise_np) -  np.min(noise_np)) * 255.0
            cv2.imwrite(f"log/vis/noise/{i}.png", noise_np)

if __name__ == "__main__":
    main()
