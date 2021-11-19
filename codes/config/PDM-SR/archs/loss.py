import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as lp

from utils.registry import LOSS_REGISTRY

from .vgg import VGGFeatureExtractor


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    def __init__(self, penealty="L1Loss"):
        super().__init__()
        self.penealty = getattr(nn, penealty)()

    def forward(self, pred):
        y_diff = self.penealty(pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = self.penealty(pred[:, :, :, :-1], pred[:, :, :, 1:])

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class GaussGuided(nn.Module):
    def __init__(self, ksize, sigma):
        super().__init__()

        ax = torch.arange(0, ksize) - ksize//2
        xx, yy = torch.meshgrid(ax, ax)
        dis = (xx ** 2 + yy ** 2)
        dis = torch.exp(-dis / sigma ** 2)
        dis = dis / dis.sum()

        self.register_buffer("gauss", dis.view(1, ksize**2, 1, 1))
    
    def forward(self, kernel):

        return F.mse_loss(self.gauss, kernel)

@LOSS_REGISTRY.register()
class PerceptualLossLPIPS(nn.Module):
    def __init__(self, net="alex", normalize=True):
        super().__init__()
        self.fn = lp.LPIPS(net=net, spatial=True)
        for p in self.fn.parameters():
            p.requires_grad = False
        
        self.normalize = normalize
    
    def forward(self, res, ref):
        return self.fn(res, ref, normalize=self.normalize).mean(), None


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, res, ref):
        return F.mse_loss(res, ref)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, res, ref):
        return F.l1_loss(res, ref)


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ["wgan", "wgan_softplus"]:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss
