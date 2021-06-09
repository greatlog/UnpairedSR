import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class CycleSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["syn_lr", "syn_hr", "real_lr"]

        self.network_names = ["netSR", "netG1", "netG2", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            "sr_adv",
            "sr_pix",
            "sr_percep",
            "g1_d1_adv",
            "g2_d2_adv",
            "g1g2_cycle",
            "g2g1_cycle",
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
                        if name == "sr_adv":
                            self.network_names.append("netD3")
                            self.netD3 = self.build_network(opt["netD3"])
                            self.networks["netD3"] = self.netD3
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

        self.fake_real_lr = self.netG1(self.syn_lr)
        self.fake_syn_hr = self.netSR(self.fake_real_lr)

        if step % 2 == 0:  # if update G1 G2
            self.rec_syn_lr = self.netG2(self.fake_real_lr)

            self.fake_syn_lr = self.netG2(self.real_lr)
            self.rec_real_lr = self.netG1(self.fake_syn_lr)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        if step % 2 == 0:  # Alternating optimize Ltrans
            loss_trans = 0
            # set D fixed
            self.set_requires_grad(["netD1", "netD2"], False)

            g1_adv_loss = self.calculate_rgan_loss_G(
                self.netD1, self.losses["g1_d1_adv"], self.real_lr, self.fake_real_lr
            )
            loss_dict["g1_adv"] = g1_adv_loss.item()
            loss_trans += self.loss_weights["g1_d1_adv"] * g1_adv_loss

            g2_adv_loss = self.calculate_rgan_loss_G(
                self.netD2, self.losses["g2_d2_adv"], self.syn_lr, self.fake_syn_lr
            )
            loss_dict["g2_adv"] = g2_adv_loss.item()
            loss_trans += self.loss_weights["g2_d2_adv"] * g2_adv_loss

            g1g2_cycle = self.losses["g1g2_cycle"](self.rec_real_lr, self.real_lr)
            loss_dict["g1g2_cycle"] = g1g2_cycle.item()
            loss_trans += self.loss_weights["g1g2_cycle"] * g1g2_cycle

            g2g1_cycle = self.losses["g2g1_cycle"](self.rec_syn_lr, self.syn_lr)
            loss_dict["g2g1_cycle"] = g2g1_cycle.item()
            loss_trans += self.loss_weights["g2g1_cycle"] * g2g1_cycle

            loss_sr_pix = self.losses["sr_pix"](self.fake_syn_hr, self.syn_hr)
            loss_dict["sr_pix"] = loss_sr_pix.item()
            loss_trans += self.loss_weights["sr_pix"] * loss_sr_pix

            self.optimizer_operator(names=["netG1", "netG2"], operation="zero_grad")
            loss_trans.backward()
            self.optimizer_operator(names=["netG1", "netG2"], operation="step")

            ## update D1, D2
            self.set_requires_grad(["netD1", "netD2"], True)

            loss_d1d2 = 0
            loss_d1 = self.calculate_rgan_loss_D(
                self.netD1, self.losses["g1_d1_adv"], self.real_lr, self.fake_real_lr
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_d1d2 += loss_d1

            loss_d2 = self.calculate_rgan_loss_D(
                self.netD2, self.losses["g2_d2_adv"], self.syn_lr, self.fake_syn_lr
            )
            loss_dict["d2_adv"] = loss_d2.item()
            loss_d1d2 += loss_d2

            self.optimizer_operator(names=["netD1", "netD2"], operation="zero_grad")
            loss_d1d2.backward()
            self.optimizer_operator(names=["netD1", "netD2"], operation="step")

        else:  # updating L SR only
            l_sr = 0

            sr_pix = self.losses["sr_pix"](self.syn_hr, self.fake_syn_hr)
            loss_dict["sr_pix"] = sr_pix.item()
            l_sr += sr_pix

            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD3"], False)
                sr_adv_g = self.calculate_rgan_loss_G(
                    self.netD3, self.losses["sr_adv"], self.syn_hr, self.fake_syn_hr
                )
                loss_dict["sr_adv_g"] = sr_adv_g.item()
                l_sr += sr_adv_g

            if self.losses.get("sr_percep"):
                sr_percep, sr_style = self.losses["sr_percep"](
                    self.syn_hr, self.fake_syn_hr
                )
                loss_dict["sr_percep"] = sr_percep.item()
                if sr_style is not None:
                    loss_dict["sr_style"] = sr_style.item()
                    l_sr += self.loss_weights["sr_percep"] * sr_style
                l_sr += self.loss_weights["sr_percep"] * sr_percep

            self.optimizer_operator(names=["netG1", "netSR"], operation="zero_grad")
            l_sr.backward()
            self.optimizer_operator(names=["netG1", "netSR"], operation="step")

            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD3"], True)
                sr_adv_d = self.calculate_rgan_loss_D(
                    self.netD3, self.losses["sr_adv"], self.syn_hr, self.fake_syn_hr
                )
                loss_dict["sr_adv_d"] = sr_adv_d.item()

                self.optimizers["netD3"].zero_grad()
                sr_adv_d.backward()
                self.optimizers["netD3"].step()

        for k, v in loss_dict.items():
            self.log_dict[k] = v

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
