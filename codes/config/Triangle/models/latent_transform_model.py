import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, 0, 1)
        output = (output * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


@MODEL_REGISTRY.register()
class LatenTransModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["src", "tgt"]

        self.network_names = ["netG1", "netG2", "netSR", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            "lr_adv",
            "g1_idt",
            "g2_idt",
            "sr_adv",
            "sr_pix",
        ]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        for name in self.network_names[:2]:
            setattr(self, name, self.build_network(opt[name]))
            self.networks[name] = getattr(self, name)

        if self.is_train:
            train_opt = opt["train"]
            self.quant = Quantization()
            self.D_ratio = train_opt["D_ratio"]

            # build networks
            for name in self.network_names[2:]:
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

    def feed_data(self, data):

        self.syn_hr = data["ref_tgt"].to(self.device)
        self.syn_lr = data["ref_src"].to(self.device)
        self.real_lr = data["src"].to(self.device)

    def forward_trans(self):
        
        self.latent_img = self.netG1(self.syn_hr)
        self.fake_latent_img = self.netG2(self.real_lr)
        self.syn_sr = self.netSR(self.latent_img)
    
    def forward_sr(self):
        self.syn_sr = self.netSR(self.latent_img.detach())
        if self.losses.get("sr_adv"):
            self.real_sr = self.netSR(self.fake_latent_img.detach())

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        # set D fixed
        self.set_requires_grad(["netD1", "netSR"], False)
        self.forward_trans()

        loss_G = 0
        g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["lr_adv"],
            self.latent_img, self.fake_latent_img
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_G += self.loss_weights["lr_adv"] * g1_adv_loss

        sr_pix = self.losses["sr_pix"](self.syn_hr, self.syn_sr)
        loss_dict["sr_pix"] = sr_pix.item()
        loss_G += self.loss_weights["sr_pix"] * sr_pix * 1000

        if self.losses.get("g1_idt"):
            syn_lr_idt = self.netG2(self.latent_img.detach())
            g1_idt = self.losses["g1_idt"](syn_lr_idt, self.latent_img)
            loss_dict["g1_idt"] = g1_idt.item()
            loss_G += self.loss_weights["g1_idt"] * g1_idt
        
        if self.losses.get("g2_idt"):
            real_lr_idt = self.netG1(self.real_sr)
            g2_idt = self.losses["g2_idt"](self.real_lr, self.fake_latent_img)
            loss_dict["g2_idt"] = g2_idt.item()
            loss_G += self.loss_weights["g2_idt"] * g2_idt

        self.optimizer_operator(names=["netG1", "netG2"], operation="zero_grad")
        loss_G.backward()
        self.optimizer_operator(names=["netG1", "netG2"], operation="step")

        loss_G = 0
        self.set_requires_grad(["netSR"], True)
        self.forward_sr()
        
        sr_pix = self.losses["sr_pix"](self.syn_hr, self.syn_sr)
        loss_dict["sr_pix"] = sr_pix.item()
        loss_G += self.loss_weights["sr_pix"] * sr_pix

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD2"], False)
            sr_adv_loss = self.calculate_gan_loss_G(
                self.netD2, self.losses["sr_adv"], self.syn_hr, self.quant(self.real_sr)
            )
            loss_dict["sr_adv"] = sr_adv_loss.item()
            loss_G += self.loss_weights["sr_adv"] * sr_adv_loss
        
        self.optimizers["netSR"].zero_grad()
        loss_G.backward()
        self.optimizers["netSR"].step()
        
        ## update D1, D2
        if step % self.D_ratio == 0:
            self.set_requires_grad(["netD1"], True)

            loss_d1 = self.calculate_gan_loss_D(
                self.netD1, self.losses["lr_adv"],
                self.latent_img.detach(), self.fake_latent_img.detach()
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_D = self.loss_weights["lr_adv"] * loss_d1
            
            self.optimizers["netD1"].zero_grad()
            loss_D.backward()
            self.optimizers["netD1"].step()

            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.netD2, self.losses["sr_adv"], self.syn_hr, self.quant(self.real_sr).detach()
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_D = self.loss_weights["sr_adv"] * loss_d2

                self.optimizers["netD2"].zero_grad()
                loss_D.backward()
                self.optimizers["netD2"].step()

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

    def test(self, src):
        self.src = src.to(self.device)
        self.set_train_state(["netG2", "netSR"], "eval")
        with torch.no_grad():
            self.fake_latent_img = self.netG2(self.src)
            self.fake_tgt = self.netSR(self.fake_latent_img)
        self.set_train_state(["netG2", "netSR"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
        return out_dict