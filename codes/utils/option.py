import logging
import os
import os.path as osp
import sys
from collections import OrderedDict

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.
    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, root_path=".", is_train=True):

    opt_path = osp.abspath(opt_path)
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = is_train
    # datasets
    for phase, dataset in opt["datasets"].items():
        for p in ["train", "val", "test"]:
            if p in phase:
                dataset["phase"] = phase
        dataset["scale"] = opt.get("scale", 1)

    # path
    if not opt.get("path"):
        opt["path"] = {}
    opt["path"]["root"] = osp.abspath(root_path)
    config_paths = osp.abspath(opt_path).split("/")
    config_dir = config_paths[config_paths.index("config") + 1]
    if is_train:
        experiments_root = osp.join(
            opt["path"]["root"], "experiments", config_dir, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root

        for dirname in ["models", "training_state", "log", "val_images"]:
            opt["path"][dirname] = osp.join(experiments_root, dirname)

        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(opt["path"]["root"], "results", config_dir, opt["name"])
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = osp.join(results_root, "log")

    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
