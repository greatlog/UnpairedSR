import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["lr", "hr"]

        self.network_names = ["netSR"]
        self.networks = {}

        self.loss_names = ["sr_adv", "sr_pix", "sr_percep"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        self.netSR = self.build_network(opt["netSR"])
        self.networks["netSR"] = self.netSR

        if self.is_train:
            train_opt = opt["train"]

            # define losses
            loss_opt = train_opt["losses"]
            for name in self.loss_names:
                loss_conf = loss_opt.get(name)
                if loss_conf:
                    if loss_conf["weight"] > 0:
                        if name == "sr_adv":
                            self.network_names.append("netD")
                            self.netD = self.build_network(opt["netD"])
                            self.networks["netD"] = self.netD
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

        self.lr = data["src"].to(self.device)
        self.hr = data["tgt"].to(self.device)

        self.sr = self.netSR(self.lr)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        l_sr = 0

        sr_pix = self.losses["sr_pix"](self.hr, self.sr)
        loss_dict["sr_pix"] = sr_pix.item()
        l_sr += self.loss_weights["sr_pix"] * sr_pix

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], False)
            sr_adv_g = self.calculate_rgan_loss_G(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_g"] = sr_adv_g.item()
            l_sr += self.loss_weights["sr_adv"] * sr_adv_g

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](self.hr, self.sr)
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                l_sr += self.loss_weights["sr_percep"] * sr_style
            l_sr += self.loss_weights["sr_percep"] * sr_percep

        self.optimizer_operator(names=["netSR"], operation="zero_grad")
        l_sr.backward()
        self.optimizer_operator(names=["netSR"], operation="step")

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], True)
            sr_adv_d = self.calculate_rgan_loss_D(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_d"] = sr_adv_d.item()

            self.optimizers["netD"].zero_grad()
            sr_adv_d.backward()
            self.optimizers["netD"].step()

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

    def test(self, real_lr):
        self.real_lr = real_lr.to(self.device)
        self.netSR.eval()
        with torch.no_grad():
            self.fake_real_hr = self.netSR(self.real_lr)
        self.netSR.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict
