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
class PairedDataset(data.Dataset):
    """
    Read paired reference images, i.e., source (src) and target (tgt),
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.src_paths, self.src_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_src"]
        )
        self.tgt_paths, self.tgt_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_tgt"]
        )

        if not len(self.src_paths) == len(self.tgt_paths):
            raise ValueError(
                "src and tgt datasets have different number of images - {}. {}.".format(
                    len(self.src_paths), len(self.tgt_paths)
                )
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
        return envs

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb" and (not self.lmdb_envs):
            self.src_env, self.tgt_env = self._init_lmdb(
                [
                    self.opt["dataroot_src"],
                    self.opt["dataroot_tgt"],
                ]
            )

        scale = self.opt["scale"]
        cropped_src_size, cropped_tgt_size = self.opt["src_size"], self.opt["tgt_size"]

        # get tgt image
        tgt_path = self.tgt_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.tgt_sizes[index].split("_")]
        else:
            resolution = None
        img_tgt = util.read_img(
            self.tgt_env, tgt_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_tgt = util.modcrop(img_tgt, scale)

        # get src image
        src_path = self.src_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.src_sizes[index].split("_")]
        else:
            resolution = None
        img_src = util.read_img(self.src_env, src_path, resolution)

        if self.opt["phase"] == "train":
            H, W, C = img_src.shape
            assert (
                cropped_src_size == cropped_tgt_size // scale
            ), "tgt size does not match src size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - cropped_src_size))
            rnd_w = random.randint(0, max(0, W - cropped_src_size))
            img_src = img_src[
                rnd_h : rnd_h + cropped_src_size, rnd_w : rnd_w + cropped_src_size
            ]
            rnd_h_tgt, rnd_w_tgt = int(rnd_h * scale), int(rnd_w * scale)
            img_tgt = img_tgt[
                rnd_h_tgt : rnd_h_tgt + cropped_tgt_size,
                rnd_w_tgt : rnd_w_tgt + cropped_tgt_size,
            ]
            # augmentation - flip, rotate
            img_tgt, img_src = util.augment(
                [img_tgt, img_src],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # change color space if necessary
        if self.opt["color"]:
            # TODO during val no definition
            img_src, img_tgt = util.channel_convert(
                self.opt["color"], [img_src, img_tgt]
            )

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_src.shape[2] == 3:
            img_src = img_src[:, :, [2, 1, 0]]
            img_tgt = img_tgt[:, :, [2, 1, 0]]

        img_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_src, (2, 0, 1)))
        ).float()
        img_tgt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_tgt, (2, 0, 1)))
        ).float()

        data_dict = {
            "src": img_src,
            "tgt": img_tgt,
            "src_path": src_path,
            "tgt_path": tgt_path,
        }

        return data_dict

    def __len__(self):
        return len(self.src_paths)
