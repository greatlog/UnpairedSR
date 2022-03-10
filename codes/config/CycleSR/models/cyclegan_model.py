import logging
from collections import OrderedDict
import random

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

        self.network_names = ["netG1", "netG2", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            "g1d1_adv",
            "g2d2_adv",
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
             # setup loss, optimizers, schedulers
            self.setup_train(train_opt)
        
            self.max_grad_norm = train_opt["max_grad_norm"]

            # buffer
            self.fake_src_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_tgt_buffer = ShuffleBuffer(train_opt["buffer_size"])

    def feed_data(self, data):

        self.src = data["src"].to(self.device)
        self.tgt = data["tgt"].to(self.device)
    
    def forward(self):

        self.fake_tgt = self.netG1(self.src)
        self.rec_src = self.netG2(self.fake_tgt)
        self.fake_src = self.netG2(self.tgt)
        self.rec_tgt = self.netG1(self.fake_src)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        self.forward()

        loss_G = 0
        # set D fixed
        self.set_requires_grad(["netD1", "netD2"], False)

        g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["g1d1_adv"], self.tgt, self.fake_tgt
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_G += self.loss_weights["g1d1_adv"] * g1_adv_loss

        g2_adv_loss = self.calculate_gan_loss_G(
            self.netD2, self.losses["g2d2_adv"], self.src, self.fake_src
        )
        loss_dict["g2_adv"] = g2_adv_loss.item()
        loss_G += self.loss_weights["g2d2_adv"] * g2_adv_loss

        if self.losses.get("g1_idt"):
            self.tgt_idt = self.netG1(self.tgt)
            g1_idt = self.losses["g1_idt"](self.tgt, self.tgt_idt)
            loss_dict["g1_idt"] = g1_idt.item()
            loss_G += self.loss_weights["g1_idt"] * g1_idt
        
        if self.losses.get("g2_idt"):
            self.src_idt = self.netG2(self.src)
            g2_idt = self.losses["g2_idt"](self.src, self.src_idt)
            loss_dict["g2_idt"] = g2_idt.item()
            loss_G += self.loss_weights["g2_idt"] * g2_idt

        g1g2_cycle = self.losses["g1g2_cycle"](self.rec_src, self.src)
        loss_dict["g1g2_cycle"] = g1g2_cycle.item()
        loss_G += self.loss_weights["g1g2_cycle"] * g1g2_cycle

        g2g1_cycle = self.losses["g2g1_cycle"](self.rec_tgt, self.tgt)
        loss_dict["g2g1_cycle"] = g2g1_cycle.item()
        loss_G += self.loss_weights["g2g1_cycle"] * g2g1_cycle

        self.set_optimizer(names=["netG1", "netG2"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(names=["netG1", "netG2"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netG1", "netG2"], operation="step")

        ## update D1, D2
        self.set_requires_grad(["netD1", "netD2"], True)

        loss_D = 0
        loss_d1 = self.calculate_gan_loss_D(
            self.netD1, self.losses["g1d1_adv"], self.tgt,
            self.fake_tgt_buffer.choose(self.fake_tgt.detach())
        )
        loss_dict["d1_adv"] = loss_d1.item()
        loss_D += loss_d1

        loss_d2 = self.calculate_gan_loss_D(
            self.netD2, self.losses["g2d2_adv"], self.src,
            self.fake_src_buffer.choose(self.fake_src)
        )
        loss_dict["d2_adv"] = loss_d2.item()
        loss_D += loss_d2

        self.set_optimizer(names=["netD1", "netD2"], operation="zero_grad")
        loss_D.backward()
        self.clip_grad_norm(names=["netD1","netD2"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netD1", "netD2"], operation="step")

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
        self.src = data["src"].to(self.device)
        self.netG1.eval()
        with torch.no_grad():
            self.fake_tgt = self.netG1(self.src)
        self.netG1.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
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
