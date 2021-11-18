import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel
from .cyclegan_model import ShuffleBuffer

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

        self.network_names = ["netSR", "netG1", "netG2", "netD1", "netD2", "netD3"]
        self.networks = {}

        self.loss_names = [
            "sr_adv",
            "sr_pix",
            "sr_pix_trans",
            "sr_percep",
            "g1_d1_adv",
            "g2_d2_adv",
            "g1_idt",
            "g2_idt",
            "g1g2_cycle",
            "g2g1_cycle",
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
            self.max_grad_norm = train_opt["max_grad_norm"]

            # buffer
            self.fake_src_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_tgt_buffer = ShuffleBuffer(train_opt["buffer_size"])

            # define losses
            loss_opt = train_opt["losses"]
            defined_loss_names = list(loss_opt.keys())
            assert set(defined_loss_names).issubset(set(self.loss_names))

            for name in defined_loss_names:
                loss_conf = loss_opt.get(name)
                if loss_conf["weight"] > 0:
                    self.loss_weights[name] = loss_conf.pop("weight")
                    self.losses[name] = self.build_loss(loss_conf)

            # build optmizers
            optimizer_opt = train_opt["optimizers"]
            defined_optimizer_names = list(optimizer_opt.keys())
            assert set(defined_optimizer_names).issubset(self.networks.keys())

            for name in defined_optimizer_names:
                optim_config = optimizer_opt[name]
                self.optimizers[name] = self.build_optimizer(
                    getattr(self, name), optim_config
                )
                
            # set schedulers
            scheduler_opt = train_opt["scheduler"]
            self.setup_schedulers(scheduler_opt)

            # set to training state
            self.set_network_state(self.networks.keys(), "train")
    
    def feed_data(self, data):
        
        self.syn_lr = data["ref_src"].to(self.device)
        self.syn_hr = data["ref_tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)

    def forward_trans(self):
        self.fake_real_lr = self.netG1(self.syn_lr)
        self.fake_syn_hr = self.netSR(self.fake_real_lr)

        self.rec_syn_lr = self.netG2(self.fake_real_lr)

        self.fake_syn_lr = self.netG2(self.real_lr)
        self.rec_real_lr = self.netG1(self.fake_syn_lr)
    
    def forward_sr(self):
        self.fake_syn_hr = self.netSR(self.fake_real_lr.detach())
        if self.losses.get("sr_adv"):
            self.fake_real_hr = self.netSR(self.real_lr)
    
    def optimize_trans_models(self, step, loss_dict):
        # set D fixed
        self.set_requires_grad(["netD1", "netD2", "netSR"], False)
        self.forward_trans()

        loss_trans = 0

        g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["g1_d1_adv"], self.real_lr, self.fake_real_lr
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_trans += self.loss_weights["g1_d1_adv"] * g1_adv_loss

        g2_adv_loss = self.calculate_gan_loss_G(
            self.netD2, self.losses["g2_d2_adv"], self.syn_lr, self.fake_syn_lr
        )
        loss_dict["g2_adv"] = g2_adv_loss.item()
        loss_trans += self.loss_weights["g2_d2_adv"] * g2_adv_loss

        g1g2_cycle = self.losses["g1g2_cycle"](self.rec_syn_lr, self.syn_lr)
        loss_dict["g1g2_cycle"] = g1g2_cycle.item()
        loss_trans += self.loss_weights["g1g2_cycle"] * g1g2_cycle

        if self.losses.get("g1_idt"):
            self.real_lr_idt = self.netG1(self.real_lr)
            g1_idt = self.losses["g1_idt"](self.real_lr, self.real_lr_idt)
            loss_dict["g1_idt"] = g1_idt.item()
            loss_trans += self.loss_weights["g1_idt"] * g1_idt
        
        if self.losses.get("g2_idt"):
            self.syn_lr_idt = self.netG2(self.syn_lr)
            g2_idt = self.losses["g2_idt"](self.syn_lr, self.syn_lr_idt)
            loss_dict["g2_idt"] = g2_idt.item()
            loss_trans += self.loss_weights["g2_idt"] * g2_idt

        g2g1_cycle = self.losses["g2g1_cycle"](self.rec_real_lr, self.real_lr)
        loss_dict["g2g1_cycle"] = g2g1_cycle.item()
        loss_trans += self.loss_weights["g2g1_cycle"] * g2g1_cycle

        loss_sr_pix = self.losses["sr_pix_trans"](self.fake_syn_hr, self.syn_hr)
        loss_dict["sr_pix_trans"] = loss_sr_pix.item()
        loss_trans += self.loss_weights["sr_pix_trans"] * loss_sr_pix

        self.set_optimizer(names=["netG1", "netG2"], operation="zero_grad")
        loss_trans.backward()
        self.clip_grad_norm(["netG1", "netG2"], self.max_grad_norm)
        self.set_optimizer(names=["netG1", "netG2"], operation="step")

        ## update D1, D2
        self.set_requires_grad(["netD1", "netD2"], True)

        loss_d1d2 = 0
        loss_d1 = self.calculate_gan_loss_D(
            self.netD1, self.losses["g1_d1_adv"], self.real_lr, self.fake_real_lr
        )
        loss_dict["d1_adv"] = loss_d1.item()
        loss_d1d2 += loss_d1

        loss_d2 = self.calculate_gan_loss_D(
            self.netD2, self.losses["g2_d2_adv"], self.syn_lr, self.fake_syn_lr
        )
        loss_dict["d2_adv"] = loss_d2.item()
        loss_d1d2 += loss_d2

        self.set_optimizer(names=["netD1", "netD2"], operation="zero_grad")
        loss_d1d2.backward()
        self.clip_grad_norm(["netD1", "netD2"], self.max_grad_norm)
        self.set_optimizer(names=["netD1", "netD2"], operation="step")

        return loss_dict
    
    def optimize_sr_models(self, step, loss_dict):

        self.set_requires_grad(["netSR"], True)
        self.forward_sr()

        l_sr = 0

        sr_pix = self.losses["sr_pix"](self.syn_hr, self.fake_syn_hr)
        loss_dict["sr_pix"] = sr_pix.item()
        l_sr += self.loss_weights["sr_pix"] * sr_pix

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD3"], False)
            sr_adv_g = self.calculate_gan_loss_G(
                self.netD3, self.losses["sr_adv"], self.syn_hr, self.fake_syn_hr
            )
            loss_dict["sr_adv_g"] = sr_adv_g.item()
            l_sr += self.loss_weights["sr_adv"] * sr_adv_g

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](
                self.syn_hr, self.fake_syn_hr
            )
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                l_sr += self.loss_weights["sr_percep"] * sr_style
            l_sr += self.loss_weights["sr_percep"] * sr_percep

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        l_sr.backward()
        self.clip_grad_norm(["netSR"], self.max_grad_norm)
        self.set_optimizer(names=["netSR"], operation="step")

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD3"], True)
            sr_adv_d = self.calculate_gan_loss_D(
                self.netD3, self.losses["sr_adv"], self.syn_hr, self.fake_syn_hr
            )
            loss_dict["sr_adv_d"] = sr_adv_d.item()
            loss_D = self.loss_weights["sr_adv"] * sr_adv_d

            self.optimizers["netD3"].zero_grad()
            loss_D.backward()
            self.clip_grad_norm(["netD3"], self.max_grad_norm)
            self.optimizers["netD3"].step()
        
        return loss_dict

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        loss_dict = self.optimize_trans_models(step, loss_dict)
        loss_dict = self.optimize_sr_models(step, loss_dict)

        for k, v in loss_dict.items():
            self.log_dict[k] = v
    
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

    def test(self, data):
        self.real_lr = data["src"].to(self.device)
        self.netSR.eval()
        with torch.no_grad():
            self.fake_real_hr = self.netSR(self.real_lr)
        self.netSR.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict
