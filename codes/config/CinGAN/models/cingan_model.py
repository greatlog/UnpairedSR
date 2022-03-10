import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel
from .trans_model import ShuffleBuffer

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
            "lr_tv",
        ]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))
        
        for name in defined_network_names:
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)
            
        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(train_opt["train"])
            self.max_grad_norm = train_opt["max_grad_norm"]

            # buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])

    def feed_data(self, data):

        self.syn_lr = data["ref_src"].to(self.device)
        self.syn_hr = data["ref_tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)
    
    def foward_trans(self):
        self.fake_syn_lr = self.netG1(self.real_lr)
        self.rec_real_lr = self.netG2(self.fake_syn_lr)
    
    def forward_sr(self):
        
        self.fake_syn_lr = self.netG1(self.real_lr)
        self.fake_real_hr = self.netSR(self.fake_syn_lr)
        self.rec_real_lr = self.netG3(self.fake_real_hr)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()
        
        # update trans
        ## update generators
        self.set_requires_grad(["netD1"], False)
        self.foward_trans()

        loss_G = 0

        g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["g1d1_adv"], self.syn_lr, self.fake_syn_lr
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_G += self.loss_weights["g1d1_adv"] * g1_adv_loss

        if self.losses.get("lr_tv"):
            lr_tv_loss = self.losses["lr_tv"](self.fake_syn_lr)
            loss_dict["lr_tv"] = lr_tv_loss.item()
            loss_G += self.loss_weights["lr_tv"] * lr_tv_loss

        g1g2_cycle = self.losses["g1g2_cycle"](self.rec_real_lr, self.real_lr)
        loss_dict["g1g2_cycle"] = g1g2_cycle.item()
        loss_G += self.loss_weights["g1g2_cycle"] * g1g2_cycle

        self.set_optimizer(names=["netG1","netG2"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["netG1","netG2"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netG1", "netG2"], operation="step")

        ## update D
        self.set_requires_grad(["netD1"], True)
        loss_d1 = self.calculate_gan_loss_D(
            self.netD1, self.losses["g1d1_adv"], self.syn_lr,
            self.fake_lr_buffer.choose(self.fake_syn_lr)
        )
        loss_dict["d1_adv"] = loss_d1.item()
        loss_D = self.loss_weights["g1d1_adv"] * loss_d1

        self.set_optimizer(["netD1"], "zero_grad")
        loss_D.backward()
        self.clip_grad_norm(["netD1"], self.max_grad_norm)
        self.set_optimizer(["netD1"], "step")

        # update sr
        self.set_requires_grad(["netD2"], False)
        self.forward_sr()

        loss_G = 0

        srd2_adv_g = self.calculate_gan_loss_G(
            self.netD2, self.losses["srd2_adv"], self.syn_hr, self.fake_real_hr
        )
        loss_dict["sr_adv"] = srd2_adv_g.item()
        loss_G += self.loss_weights["srd2_adv"] * srd2_adv_g

        if self.losses.get("sr_tv"):
            sr_tv_loss = self.losses["sr_tv"](self.fake_real_hr)
            loss_dict["sr_tv"] = sr_tv_loss.item()
            loss_G += self.loss_weights["sr_tv"] * sr_tv_loss

        srg3_cycle = self.losses["srg3_cycle"](self.rec_real_lr, self.real_lr)
        loss_dict["srg3_cycle"] = srg3_cycle.item()
        loss_G += self.loss_weights["srg3_cycle"] * srg3_cycle


        self.set_optimizer(names=["netG1", "netSR", "netG3"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(names=["netG1", "netSR", "netG3"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netG1", "netSR", "netG3"], operation="step")

        ## update D1, D2
        self.set_requires_grad(["netD2"], True)

        loss_d2 = self.calculate_gan_loss_D(
            self.netD2, self.losses["srd2_adv"], self.syn_hr,
            self.fake_hr_buffer.choose(self.fake_real_hr.detach())
        )
        loss_dict["d1_adv"] = loss_d2.item()
        loss_D = self.loss_weights["srd2_adv"] * loss_d2

        self.set_optimizer(names=["netD2"], operation="zero_grad")
        loss_D.backward()
        self.clip_grad_norm(["netD2"], self.max_grad_norm)
        self.set_optimizer(names=["netD2"], operation="step")

        self.log_dict = loss_dict
    
    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake.detach())
        d_pred_real = netD(real)

        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)
        loss_real = criterion(d_pred_fake, True, is_disc=False)

        return loss_real

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

    def test(self, data):
        self.real_lr = data["src"].to(self.device)
        self.set_network_state(["netSR", "netG1"], "eval")
        with torch.no_grad():
            self.fake_syn_lr = self.netG1(self.real_lr)
            self.fake_real_hr = self.netSR(self.fake_syn_lr)
        self.set_network_state(["netSR", "netG1"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict
