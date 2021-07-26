import logging
from collections import OrderedDict
import random

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

        self.network_names = ["Decoder", "Encoder", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            "lr_adv",
            "sr_adv",
            "sr_pix",
            "lr_quant",
            "lr_gauss"
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
            self.quant = Quantization()
            self.D_ratio = train_opt["D_ratio"]

            ## buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])

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

        self.syn_hr = data["tgt"].to(self.device)
        self.real_lr = data["src"].to(self.device)

    def encoder_forward(self):
        noise = torch.randn_like(self.real_lr).to(self.device)

        (
            self.fake_real_lr,
            self.predicted_kernel,
            self.predicted_noise,
            self.predicted_jpeg
         ) = self.Encoder(self.syn_hr, noise)
        
        self.syn_sr = self.Decoder(self.fake_real_lr)
    
    def decoder_forward(self):
        self.syn_sr_quant = self.Decoder(self.quant(self.fake_real_lr).detach())
        if self.losses.get("sr_adv"):
            self.real_sr = self.Decoder(self.real_lr)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        # optimize trans
        ## update G
        self.set_requires_grad(["netD1", "Decoder"], False)
        self.encoder_forward()
        loss_G = 0

        g1_adv_loss = self.calculate_gan_loss_G(
            self.netD1, self.losses["lr_adv"],
            self.real_lr, self.fake_real_lr
        )
        loss_dict["g1_adv"] = g1_adv_loss.item()
        loss_G += self.loss_weights["lr_adv"] * g1_adv_loss

        sr_pix = self.losses["sr_pix"](self.syn_hr, self.syn_sr)
        loss_dict["sr_pix"] = sr_pix.item()
        loss_G += self.loss_weights["sr_pix"] * sr_pix * 10

        if self.losses.get("lr_quant"):
            lr_quant = self.losses["lr_quant"](
                self.fake_real_lr, self.quant(self.fake_real_lr)
                )
            loss_dict["lr_qunat"] = lr_quant.item()
            loss_G += self.loss_weights["lr_quant"] * lr_quant
        
        if self.losses.get("lr_gauss"):
            lr_gauss = self.losses["lr_gauss"](self.predicted_kernel)
            loss_dict["lr_gauss"] = lr_gauss.item()
            loss_G += self.loss_weights["lr_gauss"] * lr_gauss

        self.set_optimizer(names=["Encoder"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["Encoder"], self.max_grad_norm)
        self.set_optimizer(names=["Encoder"], operation="step")

        ## update D
        if step % self.D_ratio == 0:
            self.set_requires_grad(["netD1"], True)
            loss_d1 = self.calculate_gan_loss_D(
                self.netD1, self.losses["lr_adv"],
                self.real_lr, self.fake_lr_buffer.choose(self.fake_real_lr)
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_D = self.loss_weights["lr_adv"] * loss_d1
            self.optimizers["netD1"].zero_grad()
            loss_D.backward()
            self.clip_grad_norm(["netD1"], self.max_grad_norm)
            self.optimizers["netD1"].step()

        # optimize SR
        self.set_requires_grad(["Decoder"], True)
        self.decoder_forward()
        loss_G = 0

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD2"], False)
            sr_adv_loss = self.calculate_gan_loss_G(
                self.netD2, self.losses["sr_adv"], self.syn_hr, self.real_sr
            )
            loss_dict["sr_adv"] = sr_adv_loss.item()
            loss_G += self.loss_weights["sr_adv"] * sr_adv_loss

        sr_pix = self.losses["sr_pix"](self.syn_hr, self.syn_sr_quant)
        loss_dict["sr_pix"] = sr_pix.item()
        loss_G += self.loss_weights["sr_pix"] * sr_pix

        self.set_optimizer(names=["Decoder"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["Decoder"], self.max_grad_norm)
        self.set_optimizer(names=["Decoder"], operation="step")

        ## update D2
        if step % self.D_ratio == 0:
            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.netD2, self.losses["sr_adv"], self.syn_hr, self.quant(self.real_sr).detach()
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_D = self.loss_weights["sr_adv"] * loss_d2
                self.optimizers["netD2"].zero_grad()
                loss_D.backward()
                self.clip_grad_norm(["netD2"], self.max_grad_norm)
                self.optimizers["netD2"].step()

        self.log_dict = loss_dict
    
    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(self.quant(fake).detach())
        d_pred_real = netD(real)

        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(self.quant(fake))
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
        self.set_network_state(["Decoder"], "eval")
        with torch.no_grad():
            self.fake_tgt = self.Decoder(self.src)
        self.set_network_state(["Decoder"], "train")

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