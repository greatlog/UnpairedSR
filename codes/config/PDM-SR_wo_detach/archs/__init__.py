import importlib
import os
import os.path as osp

from utils.registry import ARCH_REGISTRY, LOSS_REGISTRY, LR_SCHEDULER_REGISTRY

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(arch_folder)
    if v.endswith(".py")
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f"archs.{file_name}") for file_name in arch_filenames
]


def build_network(net_opt):
    which_network = net_opt["which_network"]
    net = ARCH_REGISTRY.get(which_network)(**net_opt["setting"])
    return net


def build_loss(loss_opt):
    loss_type = loss_opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**loss_opt)
    return loss

def build_scheduler(optimizer, scheduler_opt):
    scheduler_type = scheduler_opt.pop("type")
    scheduler = LR_SCHEDULER_REGISTRY.get(scheduler_type)(optimizer, **scheduler_opt)
    return scheduler
