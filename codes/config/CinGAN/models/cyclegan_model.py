import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class CycleGANModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["src", "tgt"]

        self.network_names = ["netG1", "netG2", "netD1"]
        self.networks = {}

        self.loss_names = ["g1d1_adv", "g1g2_cycle", "lr_tv"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        self.netG1 = self.build_network(opt["netG1"])
        self.networks["netG1"] = self.netG1

        if self.is_train:
            train_opt = opt["train"]

            # build networks
            for name in self.network_names[1:]:
                setattr(self, name, self.build_network(opt[name]))
                self.networks[name] = getattr(self, name)

            # define losses
            loss_opt = train_opt["losses"]
            for name in self.loss_names:
                loss_conf = loss_opt.get(name)
                if loss_conf:
                    if loss_conf["weight"] > 0:
                        self.loss_weights[name] = loss_conf.pop("weight")
                        self.losses[name] = self.build_loss(loss_conf)

            # build optmizers
            self.set_train_state(self.networks, "train")
            optimizer_opt = train_opt["optimizers"]
            for name in self.network_names:
                if optimizer_opt.get(name):
                    optim_config = optimizer_opt[name]
                    self.optimizers[name] = self.build_optimizer(
                        getattr(self, name), optim_config
                    )
                else:
                    logger.info(
                        "Network {} has no Corresponding Optimizer!!".format(name)
                    )

            # set schedulers
            scheduler_opt = train_opt["scheduler"]
            self.setup_schedulers(scheduler_opt)

            # set to training state
            self.set_train_state(self.networks.keys(), "train")

    def forward(self, data, step):

        self.src = data["src"].to(self.device)
        self.tgt = data["tgt"].to(self.device)

        self.fake_tgt = self.netG1(self.src)
        self.rec_src = self.netG2(self.fake_tgt)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        loss_G = 0
        # set D fixed
        self.set_requires_grad(["netD1"], False)

        g1_adv_loss = self.calculate_rgan_loss_G(
            self.netD1, self.losses["g1d1_adv"], self.tgt, self.fake_tgt
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_G += self.loss_weights["g1d1_adv"] * g1_adv_loss

        lr_tv = self.losses["lr_tv"](self.fake_tgt)
        loss_dict["lr_tv"] = lr_tv.item()
        loss_G += self.loss_weights["lr_tv"] * lr_tv

        g1g2_cycle = self.losses["g1g2_cycle"](self.rec_src, self.src)
        loss_dict["g1g2_cycle"] = g1g2_cycle.item()
        loss_G += self.loss_weights["g1g2_cycle"] * g1g2_cycle

        self.optimizer_operator(names=["netG1", "netG2"], operation="zero_grad")
        loss_G.backward()
        self.optimizer_operator(names=["netG1", "netG2"], operation="step")

        ## update D1, D2
        self.set_requires_grad(["netD1"], True)

        loss_D = 0
        loss_d1 = self.calculate_rgan_loss_D(
            self.netD1, self.losses["g1d1_adv"], self.tgt, self.fake_tgt
        )
        loss_dict["d1_adv"] = loss_d1.item()
        loss_D += loss_d1

        self.optimizer_operator(names=["netD1"], operation="zero_grad")
        loss_D.backward()
        self.optimizer_operator(names=["netD1"], operation="step")

        self.log_dict = loss_dict

    def calculate_rgan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake.detach())
        d_pred_real = netD(real)
        loss_real = criterion(
            d_pred_real - d_pred_fake.detach().mean(), True, is_disc=False
        )
        loss_fake = criterion(
            d_pred_fake - d_pred_real.detach().mean(), False, is_disc=False
        )

        loss = (loss_real + loss_fake) / 2

        return loss

    def calculate_rgan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)
        d_pred_real = netD(real).detach()
        loss_real = criterion(d_pred_real - d_pred_fake.mean(), False, is_disc=False)
        loss_fake = criterion(d_pred_fake - d_pred_real.mean(), True, is_disc=False)

        loss = (loss_real + loss_fake) / 2

        return loss

    def test(self, src):
        self.src = src.to(self.device)
        self.netG1.eval()
        with torch.no_grad():
            self.fake_tgt = self.netG1(self.src)
        self.netG1.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
        return out_dict
