import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY

from models.base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class DegSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["syn_lr", "syn_hr", "real_lr"]

        self.network_names = ["netSR", "netDeg", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
           "lr_adv",
            "lr_percep",
            "lr_color",
            "lr_tv",
            "sr_tv",
            "sr_pix",
            "sr_adv",
            "sr_percep"
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
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.max_grad_norm = train_opt["max_grad_norm"]
            self.D_ratio = train_opt["D_ratio"]

            ## buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])
    
    def feed_data(self, data):

        self.syn_hr = data["tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)

    def forward(self):

        self.fake_real_lr = self.netDeg(self.syn_hr)
        self.fake_syn_hr = self.netSR(self.fake_real_lr)
        # self.fake_real_hr = self.netSR(self.real_lr)

    def optimize_parameters(self, step):
        self.forward()
        
        loss_dict = OrderedDict()

        loss_G = 0

        if self.losses.get("lr_adv"):
            self.set_requires_grad(["netD1"], False)
            g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["lr_adv"], self.real_lr, self.fake_real_lr
            )
            loss_dict["g1_adv"] = g1_adv_loss.item()
            loss_G += self.loss_weights["lr_adv"] * g1_adv_loss

        if self.losses.get("lr_percep"):
            lr_percep, lr_style = self.losses["lr_percep"](self.real_lr, self.fake_real_lr)
            loss_dict["lr_percep"] = lr_percep.item()
            if lr_style is not None:
                loss_dict["lr_style"] = lr_style.item()
                loss_G += self.loss_weights["sr_percep"] * lr_style
            loss_G += self.loss_weights["sr_percep"] * lr_percep
        
        if self.losses.get("lr_color"):
            lr_color = self.losses["lr_color"](self.fake_real_lr, self.syn_hr)
            loss_dict["lr_color"] = lr_color.item()
            loss_G += self.loss_weights["lr_color"] * lr_color

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD2"], False)
            sr_adv = self.calculate_gan_loss_G(
                self.netD2, self.losses["sr_adv"], self.syn_hr, self.fake_syn_hr
            )
            loss_dict["sr_adv"] = sr_adv.item()
            loss_G += self.loss_weights["sr_adv"] * sr_adv

        if self.losses.get("sr_pix"):
            sr_pix = self.losses["sr_pix"](self.fake_syn_hr, self.syn_hr)
            loss_dict["sr_pix"] = sr_pix.item()
            loss_G += self.loss_weights["sr_pix"] * sr_pix

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](self.syn_hr, self.fake_syn_hr)
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                loss_G += self.loss_weights["sr_percep"] * sr_style
            loss_G += self.loss_weights["sr_percep"] * sr_percep
        
        if self.losses.get("sr_tv"):
            sr_tv = self.losses["sr_tv"](self.fake_real_hr)
            loss_dict["sr_tv"] = sr_tv.item()
            loss_G = self.loss_weights["sr_tv"] * sr_tv

        self.set_optimizer(names=["netDeg", "netSR"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(names=["netDeg", "netSR"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netDeg", "netSR"], operation="step")

        ## update D1, D2
        loss_D = 0

        if self.losses.get("lr_adv"):
            if step % self.D_ratio == 0:
                self.set_requires_grad(["netD1"], True)
                loss_d1 = self.calculate_gan_loss_D(
                    self.netD1, self.losses["lr_adv"], self.real_lr,
                    self.fake_lr_buffer.choose(self.fake_real_lr.detach())
                )
                loss_dict["d1_adv"] = loss_d1.item()
                loss_d1 = self.loss_weights["lr_adv"] * loss_d1

                self.set_optimizer(names=["netD1"], operation="zero_grad")
                loss_d1.backward()
                self.clip_grad_norm(["netD1"], norm=self.max_grad_norm)
                self.set_optimizer(names=["netD1"], operation="step")

        if self.losses.get("sr_adv"):
            if step % self.D_ratio == 0:
                self.set_requires_grad(["netD2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.netD2, self.losses["sr_adv"], self.syn_hr,
                    self.fake_sr_buffer.choose(self.fake_syn_hr.detach())
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_d2 = self.loss_weights["sr_adv"] * loss_d2

                self.set_optimizer(names=["netD2"], operation="zero_grad")
                loss_d1.backward()
                self.clip_grad_norm(["netD2"], norm=self.max_grad_norm)
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

    def test(self, data):
        self.real_lr = data["src"].to(self.device)
        self.set_network_state(["netSR"], "eval")
        with torch.no_grad():
            self.fake_real_hr = self.netSR(self.real_lr)
        self.set_network_state(["netSR"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.real_lr.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_real_hr.detach()[0].float().cpu()
        return out_dict


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return  images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images