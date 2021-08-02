import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from archs import build_network, build_loss

from utils.registry import MODEL_REGISTRY
from .lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.is_train = opt["is_train"]
        self.log_dict = OrderedDict()

        self.data_names = []
        self.network_names = []
        self.networks = {}

        self.optimizers = {}
        self.schedulers = {}

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def build_network(self, net_opt):

        net = build_network(net_opt)
        net = self.model_to_device(net)

        if net_opt.get("pretrain"):
            pretrain = net_opt.pop("pretrain")
            self.load_network(net, pretrain["path"], pretrain["strict_load"])

        self.print_network(net)
        return net

    def build_loss(self, loss_config):
        loss = build_loss(loss_config)
        loss = loss.to(self.device)
        return loss

    @staticmethod
    def build_optimizer(net, optim_config):
        optim_params = []
        for v in net.parameters():
            if v.requires_grad:
                optim_params.append(v)
        optim_type = optim_config.pop("type")
        optimizer = getattr(torch.optim, optim_type)(
            params=optim_params, **optim_config
        )
        return optimizer

    def setup_schedulers(self, scheduler_opt):
        """Set up schedulers."""
        scheduler_type = scheduler_opt.pop("type")

        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            for name, optimizer in self.optimizers.items():
                self.schedulers[name] = MultiStepRestartLR(optimizer, **scheduler_opt)

        elif scheduler_type == "CosineAnnealingRestartLR":
            for name, optimizer in self.ptimizers.items():
                self.schedulers[name] = CosineAnnealingRestartLR(
                    optimizer, **scheduler_opt
                )
        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt["dist"]:
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        else:
            net = DataParallel(net)
        return net

    def print_network(self, net):
        # Generator
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            net_struc_str = "{} - {}".format(
                net.__class__.__name__, net.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(net.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def set_optimizer(self, names, operation):
        for name in names:
            getattr(self.optimizers[name], operation)()

    def set_requires_grad(self, names, requires_grad):
        for name in names:
            for v in self.networks[name].parameters():
                v.requires_grad = requires_grad

    def set_network_state(self, names, state):
        for name in names:
            getattr(self.networks[name], state)()
    
    def clip_grad_norm(self, names, norm):
        for name in names:
            nn.utils.clip_grad_norm_(
                self.networks[name].parameters(), max_norm=norm
            )

    def _set_lr(self, lr_groups_l):
        """set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for _, scheduler in self.schedulers.items():
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return list(self.optimizers.values())[0].param_groups[0]["lr"]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save(self, iter_label):
        for name in self.optimizers.keys():
            self.save_network(self.networks[name], name, iter_label)

    def load_network(self, network, load_path, strict=True):
        if load_path is not None:
            if isinstance(network, nn.DataParallel) or isinstance(
                network, DistributedDataParallel
            ):
                network = network.module
            load_net = torch.load(load_path)
            load_net_clean = OrderedDict()  # remove unnecessary 'module.'
            for k, v in load_net.items():
                if k.startswith("module."):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        """Saves training state during training, which will be used for resuming"""
        state = {"epoch": epoch, "iter": iter_step, "schedulers": {}, "optimizers": {}}
        for k, s in self.schedulers.items():
            state["schedulers"][k] = s.state_dict()
        for k, o in self.optimizers.items():
            state["optimizers"][k] = o.state_dict()
        save_filename = "{}.state".format(iter_step)
        save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for name, o in resume_optimizers.items():
            self.optimizers[name].load_state_dict(o)
        for name, s in resume_schedulers.items():
            self.schedulers[name].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def get_current_log(self):
        return self.log_dict
