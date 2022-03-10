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

        self.data_names = ["lr", "hr"]

        self.network_names = ["netSR"]
        self.networks = {}

        self.loss_names = ["sr_adv", "sr_pix", "sr_percep"]
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
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

    def feed_data(self, data):

        self.lr = data["src"].to(self.device)
        self.hr = data["tgt"].to(self.device)

    def forward(self):

        self.sr = self.netSR(self.lr)

    def optimize_parameters(self, step):

        self.forward()

        loss_dict = OrderedDict()

        l_sr = 0

        sr_pix = self.losses["sr_pix"](self.hr, self.sr)
        loss_dict["sr_pix"] = sr_pix
        l_sr += self.loss_weights["sr_pix"] * sr_pix

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], False)
            sr_adv_g = self.calculate_rgan_loss_G(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_g"] = sr_adv_g
            l_sr += self.loss_weights["sr_adv"] * sr_adv_g

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](self.hr, self.sr)
            loss_dict["sr_percep"] = sr_percep
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style
                l_sr += self.loss_weights["sr_percep"] * sr_style
            l_sr += self.loss_weights["sr_percep"] * sr_percep

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        l_sr.backward()
        self.set_optimizer(names=["netSR"], operation="step")

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], True)
            sr_adv_d = self.calculate_rgan_loss_D(
                self.netD, self.losses["sr_adv"], self.hr, self.sr
            )
            loss_dict["sr_adv_d"] = sr_adv_d

            self.optimizers["netD"].zero_grad()
            sr_adv_d.backward()
            self.optimizers["netD"].step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

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

    def test(self, data, crop_size=None):
        self.real_lr = data["src"].to(self.device)
        self.netSR.eval()
        with torch.no_grad():
            if crop_size is None:
                self.fake_real_hr = self.netSR(self.real_lr)
            else:
                self.fake_real_hr = self.crop_test(self.real_lr, crop_size)
        self.netSR.train()
    
    def crop_test(self, lr, crop_size):
        b, c, h, w = lr.shape
        scale = self.opt["scale"]

        h_start = list(range(0, h-crop_size, crop_size))
        w_start = list(range(0, w-crop_size, crop_size))

        sr1 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hs in h_start:
            for ws in w_start:
                lr_patch = lr[:, :, hs: hs+crop_size, ws: ws+crop_size]
                sr_patch = self.netSR(lr_patch)

                sr1[:, :, 
                    int(hs*scale):int((hs+crop_size)*scale),
                    int(ws*scale):int((ws+crop_size)*scale)
                ] = sr_patch
        
        h_end = list(range(h, crop_size, -crop_size))
        w_end = list(range(w, crop_size, -crop_size))

        sr2 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hd in h_end:
            for wd in w_end:
                lr_patch = lr[:, :, hd-crop_size:hd, wd-crop_size:wd]
                sr_patch = self.netSR(lr_patch)

                sr2[:, :, 
                    int((hd-crop_size)*scale):int(hd*scale),
                    int((wd-crop_size)*scale):int(wd*scale)
                ] = sr_patch

        mask1 = (
            (sr1 == -1).float() * 0 + 
            (sr2 == -1).float() * 1 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        mask2 = (
            (sr1 == -1).float() * 1 + 
            (sr2 == -1).float() * 0 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        sr = mask1 * sr1 + mask2 * sr2

        return sr
            
    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict
