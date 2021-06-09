import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class CinGANModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["syn_lr", "syn_hr", "real_lr"]

        self.network_names = ["netSR", "netG1", "netG2", "netG3", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            "srd2_adv",
            "sr_tv",
            "srg3_cycle",
            "g1d1_adv",
            "g1g2_cycle",
            "lr_tv"
        ]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        self.netSR = self.build_network(opt["netSR"])
        self.networks["netSR"] = self.netSR

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

        self.syn_lr = data["ref_src"].to(self.device)
        self.syn_hr = data["ref_tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)

        self.fake_syn_lr = self.netG1(self.real_lr)
        self.rec_real_lr_g2 = self.netG2(self.fake_syn_lr)
        self.fake_real_hr = self.netSR(self.fake_syn_lr.detach())
        self.rec_real_lr_g3 = self.netG3(self.fake_real_hr)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        loss_trans = 0
        self.set_requires_grad(["netD1", "netD2"], False)

        g1_adv_loss = self.calculate_rgan_loss_G(
            self.netD1, self.losses["g1d1_adv"], self.syn_lr, self.fake_syn_lr
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_trans += self.loss_weights["g1d1_adv"] * g1_adv_loss

        lr_tv_loss = self.losses["lr_tv"](self.fake_syn_lr)
        loss_dict["lr_tv"] = lr_tv_loss.item()
        loss_trans += self.loss_weights["lr_tv"] * lr_tv_loss

        g1g2_cycle = self.losses["g1g2_cycle"](self.rec_real_lr_g2, self.real_lr)
        loss_dict["g1g2_cycle"] = g1g2_cycle.item()
        loss_trans += self.loss_weights["g1g2_cycle"] * g1g2_cycle

        self.optimizer_operator(
            names=["netG1", "netG2"], operation="zero_grad"
        )
        loss_trans.backward()
        self.optimizer_operator(names=["netG1", "netG2"], operation="step")

        ## update D1, D2
        self.set_requires_grad(["netD1"], True)

        loss_d1d2 = 0
        loss_d1 = self.calculate_rgan_loss_D(
            self.netD1, self.losses["g1d1_adv"], self.syn_lr, self.fake_syn_lr
        )
        loss_dict["d1_adv"] = loss_d1.item()
        loss_d1d2 += loss_d1

        self.optimizer_operator(names=["netD1"], operation="zero_grad")
        loss_d1d2.backward()
        self.optimizer_operator(names=["netD1"], operation="step")

        l_sr = 0
        if self.losses.get("srd2_adv"):
            self.set_requires_grad(["netD2"], False)
            srd2_adv_g = self.calculate_rgan_loss_G(
                self.netD2, self.losses["srd2_adv"], self.syn_hr, self.fake_real_hr
            )
            loss_dict["srd2_adv_g"] = srd2_adv_g.item()
            l_sr += srd2_adv_g
        
        sr_tv_loss = self.losses["sr_tv"](self.fake_real_hr)
        loss_dict["sr_tv"] = sr_tv_loss.item()
        l_sr += sr_tv_loss

        srg3_cycle = self.losses["srg3_cycle"](self.rec_real_lr_g3, self.real_lr)
        loss_dict["srg3_cycle"] = srg3_cycle.item()
        l_sr += srg3_cycle

        self.optimizer_operator(names=["netSR", "netG3"], operation="zero_grad")
        l_sr.backward()
        self.optimizer_operator(names=["netSR", "netG3"], operation="step")

        if self.losses.get("srd2_adv"):
            self.set_requires_grad(["netD2"], True)
            srd2_adv_d = self.calculate_rgan_loss_D(
                self.netD2, self.losses["srd2_adv"], self.syn_hr, self.fake_real_hr
            )
            loss_dict["srd2_adv_d"] = srd2_adv_d.item()

            self.optimizers["netD2"].zero_grad()
            srd2_adv_d.backward()
            self.optimizers["netD2"].step()

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
        self.set_requires_grad(["netSR", "netG1"], "eval")
        with torch.no_grad():
            self.fake_syn_lr = self.netG1(self.real_lr)
            self.fake_real_hr = self.netSR(self.fake_syn_lr)
        self.set_requires_grad(["netSR", "netG1"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict
