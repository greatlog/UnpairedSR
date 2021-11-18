import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import utils as util
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
    """
    Read Single Image.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.img_paths, self.img_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot"]
        )

        if opt["data_type"] == "lmdb":
            self.lmdb_envs = False

    def _init_lmdb(self, dataroots):
        envs = []
        for dataroot in dataroots:
            envs.append(
                lmdb.open(
                    dataroot, readonly=True, lock=False, readahead=False, meminit=False
                )
            )
        self.lmdb_envs = True
        return envs[0] if len(envs) == 1 else envs

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb" and (not self.lmdb_envs):
            self.env = self._init_lmdb([self.opt["dataroot"]])

        scale = self.opt["scale"]

        # get image
        img_path = self.img_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.img_sizes[index].split("_")]
        else:
            resolution = None
        img = util.read_img(self.env, img_path, resolution)

        if self.opt["phase"] != "train" and self.opt.get("scale"):
            img = util.modcrop(img, self.opt["scale"])

        if self.opt["phase"] == "train":
            H, W, C = img.shape
            cropped_size = self.opt["img_size"]

            # randomly crop
            rnd_h = random.randint(0, max(0, H - cropped_size))
            rnd_w = random.randint(0, max(0, W - cropped_size))
            img = img[rnd_h : rnd_h + cropped_size, rnd_w : rnd_w + cropped_size]
            # augmentation - flip, rotate
            img = util.augment(
                [img],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # change color space if necessary
        if self.opt["color"]:
            # TODO during val no definition
            img = util.channel_convert(self.opt["color"], [img])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]

        img = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        ).float()

        data_dict = {
            "img": img,
            "img_path": img_path,
        }

        return data_dict

    def __len__(self):
        return len(self.img_paths)
