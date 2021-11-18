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
class PairedRefDataset(data.Dataset):
    """
    Read paired reference images, i.e., source (src) and target (tgt), and unparied source images.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.ref_src_paths, self.ref_src_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_ref_src"]
        )
        self.ref_tgt_paths, self.ref_tgt_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_ref_tgt"]
        )
        self.src_paths, self.src_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_src"]
        )

        if not len(self.ref_src_paths) == len(self.ref_tgt_paths):
            raise ValueError(
                "Reference source and Reference target datasets have different number of images - {}. {}.".format(
                    len(self.ref_src_paths), len(self.ref_tgt_paths)
                )
            )

        if opt.get("ratios"):
            ratio_ref, ratio_src = opt["ratios"]
            self.ref_src_paths *= ratio_ref
            self.ref_src_sizes *= ratio_ref
            self.ref_tgt_paths *= ratio_ref
            self.ref_tgt_sizes *= ratio_ref
            self.src_paths *= ratio_src
            self.src_sizes *= ratio_src

        merged_src = list(zip(self.src_paths, self.src_sizes))
        random.shuffle(merged_src)
        self.src_paths[:], self.src_sizes[:] = zip(*merged_src)

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
            self.ref_src_env, self.ref_tgt_env, self.src_env = self._init_lmdb(
                [
                    self.opt["dataroot_ref_src"],
                    self.opt["dataroot_ref_tgt"],
                    self.opt["dataroot_src"],
                ]
            )

        scale = self.opt["scale"]
        cropped_src_size, cropped_tgt_size = self.opt["src_size"], self.opt["tgt_size"]

        # get ref target image
        ref_tgt_path = self.ref_tgt_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.ref_tgt_sizes[index].split("_")]
        else:
            resolution = None
        img_ref_tgt = util.read_img(
            self.ref_tgt_env, ref_tgt_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_ref_tgt = util.modcrop(img_ref_tgt, scale)

        # get ref source image
        ref_src_path = self.ref_src_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.ref_src_sizes[index].split("_")]
        else:
            resolution = None
        img_ref_src = util.read_img(self.ref_src_env, ref_src_path, resolution)

        # get source image
        src_path = self.src_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.src_sizes[index].split("_")]
        else:
            resolution = None
        img_src = util.read_img(self.src_env, src_path, resolution)

        if self.opt["phase"] == "train":
            H, W, C = img_ref_src.shape
            assert (
                cropped_src_size == cropped_tgt_size // scale
            ), "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - cropped_src_size))
            rnd_w = random.randint(0, max(0, W - cropped_src_size))
            img_ref_src = img_ref_src[
                rnd_h : rnd_h + cropped_src_size, rnd_w : rnd_w + cropped_src_size
            ]
            rnd_h_tgt, rnd_w_tgt = int(rnd_h * scale), int(rnd_w * scale)
            img_ref_tgt = img_ref_tgt[
                rnd_h_tgt : rnd_h_tgt + cropped_tgt_size,
                rnd_w_tgt : rnd_w_tgt + cropped_tgt_size,
                :,
            ]

            src_h, src_w, _ = img_src.shape
            rnd_h = random.randint(0, max(0, src_h - cropped_src_size))
            rnd_w = random.randint(0, max(0, src_w - cropped_src_size))
            img_src = img_src[
                rnd_h : rnd_h + cropped_src_size, rnd_w : rnd_w + cropped_src_size
            ]

            # augmentation - flip, rotate
            img_ref_tgt, img_ref_src = util.augment(
                [img_ref_tgt, img_ref_src],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )
            img_src = util.augment(
                [img_src],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # change color space if necessary
        if self.opt["color"]:
            # TODO during val no definition
            img_ref_src, img_ref_tgt, img_src = util.channel_convert(
                self.opt["color"], [img_ref_src, img_ref_tgt, img_src]
            )

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_ref_src.shape[2] == 3:
            img_ref_src = img_ref_src[:, :, [2, 1, 0]]
            img_ref_tgt = img_ref_tgt[:, :, [2, 1, 0]]
            img_src = img_src[:, :, [2, 1, 0]]

        img_ref_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_ref_src, (2, 0, 1)))
        ).float()
        img_ref_tgt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_ref_tgt, (2, 0, 1)))
        ).float()
        img_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_src, (2, 0, 1)))
        ).float()

        data_dict = {
            "ref_src": img_ref_src,
            "ref_tgt": img_ref_tgt,
            "src": img_src,
            "ref_src_path": ref_src_path,
            "ref_tgt_path": ref_tgt_path,
            "src_path": src_path,
        }

        return data_dict

    def __len__(self):
        return len(self.src_paths)
