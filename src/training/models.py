import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from kornia.geometry import conv_soft_argmax2d
from kornia.filters import gaussian_blur2d

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  
project_root = os.path.dirname(os.path.dirname(current_dir))  
sys.path.append(project_root)

from src.utils import utils
from src.training import losses


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => [Dropout] => convolution => [BN] => ReLU)
    """
    def __init__(self, in_channels, out_channels,
                 mid_channels=None,
                 use_dropout=True,
                 use_batchnorm=True,
                 kernel_size=3,
                 dropout_prob=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.layers = nn.ModuleList()
        self.kernel_size = kernel_size

        self.layers.append(nn.Conv2d(in_channels, mid_channels,
                                     kernel_size=(self.kernel_size,self.kernel_size), padding=1))
        if use_batchnorm: self.layers.append(nn.BatchNorm2d(mid_channels))
        self.layers.append(nn.LeakyReLU(inplace=True))
        if use_dropout: self.layers.append(nn.Dropout(p=dropout_prob))
        self.layers.append(nn.Conv2d(mid_channels, out_channels,
                                     kernel_size=(self.kernel_size,self.kernel_size), padding=1))
        if use_batchnorm: self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Down(nn.Module):
    """Down-sampling layer => DoubleConv
    """
    def __init__(self, in_channels, out_channels,
                 down_layer="strided_conv",
                 drop=0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        # Choose the downsampling layer function
        if down_layer is "maxpool":
            self.layers.append(nn.MaxPool2d(2))
        elif down_layer is "avgpool":
            self.layers.append(nn.AvgPool2d(2))
        else:  # Otherwise use strided convolutions
            self.layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(2,2), padding=1))

        self.layers.append(DoubleConv(in_channels=in_channels, out_channels=out_channels,
                                      use_dropout=True,
                                      use_batchnorm=True,
                                      kernel_size=3,
                                      dropout_prob=drop))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UpsampleDeterministic(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upscale = scale_factor

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        """
        return x[:, :, :, None,:,None].expand(
            -1, -1, -1, self.upscale, -1, self.upscale).reshape(x.size(0), x.size(1), x.size(2) * self.upscale,
                                                                x.size(3) * self.upscale)


class Up(nn.Module):
    """Upsampling layer => DoubleConv
    """
    def __init__(self, in_channels, out_channels,
                 bilinear=False,
                 deterministic=False,
                 drop=0.2):
        # Changed default
        super().__init__()
        self.layers = nn.ModuleList()

        if not deterministic:
            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels=in_channels,
                                       out_channels=out_channels,
                                       mid_channels=in_channels//2,
                                       use_dropout=True,
                                       use_batchnorm=True,
                                       kernel_size=3,
                                       dropout_prob=drop)
            else:
                self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=in_channels // 2,
                                             kernel_size=(2,2),
                                             stride=(2,2))
                self.conv = DoubleConv(in_channels=in_channels,
                                       out_channels=in_channels//2,
                                       use_dropout=True,
                                       use_batchnorm=True,
                                       kernel_size=3,
                                       dropout_prob=drop)
        else:
            # Use deterministic upsampling layer
            # Note: See these issues, it may not work well!!
            # https://github.com/pytorch/pytorch/issues/12207
            # https://github.com/pytorch/pytorch/issues/7068
            # Here also the up layer does not do any reduction of channels,
            # so it has to be done by the DoubleConv layer
            self.up = UpsampleDeterministic(scale_factor=2)
            self.conv = DoubleConv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   mid_channels=in_channels//2,
                                   use_dropout=True,
                                   use_batchnorm=True,
                                   kernel_size=3,
                                   dropout_prob=drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Need to adjust/match the size of x1 and x2
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class Unet(nn.Module):
    """
    (Deterministic up to 25 decimal points which I think is good enough)
    Deterministic False, Bilinear = True : NOPE
    Deterministic False, Bilinear = False : YES (MOSTLY??)
    Deterministic TRUE: YES

    *************************************************************
    Deterministic behaviour of upsampling layer:
    ** When deterministic flag is False:
    -- When bilinear upsampling is used (bilinear=True) then it is not deterministic,
    but it is comparable (probably same values up to first 4 decimal points)
    -- When transpose convolutions are used (bilinear=False), then it is 'mostly' deterministic,
    this means it offers deterministic results but in some runs it may not be the same. This behaviour
    is not completely understood. But we can safely assume, it provides comparable results.

    ** When deterministic flag is True:
    -- The model offers completely deterministic behaviour due to the use of the deterministic
    upsampling layer. But the performance of this layer has to be benchmarked.
    """
    def __init__(self, input_channels=3,
                 num_classes=1,
                 down_layer="maxpool",
                 feature_list=None,
                 use_dropout=True,
                 use_batchnorm=True,
                 drop=None,
                 kernel_size=3,
                 bilinear=True,
                 deterministic=False):
        super(Unet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.down_layer = down_layer
        self.feature_list = feature_list if feature_list else [16, 32, 64, 128, 256]
        self.feature_list_flipped = np.flip(self.feature_list)

        if drop:
            self.drop = [drop] * len(self.feature_list) if isinstance(drop, int) else drop
        else:
            self.drop = [0.3, 0.37, 0.43, 0.5, 0.5]
        assert(isinstance(drop, list) and len(drop)==len(self.feature_list),
               "The dropout value has to be either an integer or a list of length of number of features")

        self.drop_flipped = np.flip(self.drop)
        self.bilinear = bilinear
        self.deterministic = deterministic
        self.factor = 2 if self.deterministic or self.bilinear else 1
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.conv = DoubleConv(in_channels=input_channels,
                               out_channels=self.feature_list[0],
                               use_dropout=use_dropout,
                               use_batchnorm=use_batchnorm,
                               kernel_size=kernel_size,
                               dropout_prob=self.drop[0])

        # Encoder path (downsampling path)
        for i in range(len(self.feature_list)-1):
            if i+1 == len(self.feature_list)-1:
                # For the last downsampling layer, need to include factor depending on bilinear flag
                self.down_layers.append(Down(in_channels=self.feature_list[i],
                                             out_channels=self.feature_list[i + 1]//self.factor,
                                             down_layer=self.down_layer,
                                             drop=self.drop[i + 1]))
            else:
                self.down_layers.append(Down(in_channels=self.feature_list[i],
                                             out_channels=self.feature_list[i+1],
                                             down_layer=self.down_layer,
                                             drop=self.drop[i+1]))

        # Decoder path (Upsampling path)
        for i in range(len(self.feature_list)-1):
            if i + 1 == len(self.feature_list_flipped)-1:
                self.up_layers.append(Up(in_channels=self.feature_list_flipped[i],
                                         out_channels=(self.feature_list_flipped[i + 1]),
                                         bilinear=self.bilinear,
                                         deterministic=self.deterministic,
                                         drop=self.drop_flipped[i]))
            else:
                self.up_layers.append(Up(in_channels=self.feature_list_flipped[i],
                                      out_channels=(self.feature_list_flipped[i+1]//self.factor),
                                      bilinear=self.bilinear,
                                      deterministic=self.deterministic,
                                      drop=self.drop_flipped[i]))

        self.outconv = OutConv(in_channels=self.feature_list[0],
                               out_channels=self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        outputs = []

        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i+1)])

        x = self.outconv(x)
        return x


class UnetGauss(Unet):
    def __init__(self, kernel=None,
                 **kwargs):
        super(UnetGauss, self).__init__(**kwargs)
        self.kernel=torch.ones(3, 3) if kernel is None else kernel
        self.gauss = gaussian_blur2d
        self.softargmax = conv_soft_argmax2d

    def forward(self, x):
        x = self.conv(x)
        outputs = []

        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i+1)])

        x = self.outconv(x)
        x = self.gauss(x, (3, 3), sigma=(1, 1))  # Local gaussian smoothing of the max values
        pred_coords, pred_mask = self.softargmax(x, (3, 3), (1, 1), (1, 1), output_value=True)
        return pred_mask, x


def make_unet(model_config, model=Unet):
    """
    Wrapper to create a unet model for image segmentation
    :param model_config: Key-value pairs for network params
    :return: pytorch model of unet
    """
    input_channels = model_config.get('input_channels')
    num_classes = model_config.get('num_classes')
    down_layer = model_config.get('down_layer')
    feature_list = model_config.get('feature_list')
    use_dropout = model_config.get('use_dropout')
    use_batchnorm = model_config.get('use_batchnorm')
    drop = model_config.get('drop')
    kernel_size = model_config.get('kernel_size')
    bilinear = model_config.get('bilinear')
    deterministic = model_config.get('deterministic')

    return model(input_channels=input_channels,
                 num_classes=num_classes,
                 down_layer=down_layer,
                 feature_list=feature_list,
                 use_dropout=use_dropout,
                 use_batchnorm=use_batchnorm,
                 drop=drop,
                 kernel_size=kernel_size,
                 bilinear=bilinear,
                 deterministic=deterministic)


# *******************************************************************************************************
"""
Utilities
"""

def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


class UNet_pl(pl.LightningModule):
    def __init__(self, n_channels, n_classes, kernel=None, bilinear=True):
        super(UNet_pl, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.kernel=torch.ones(3, 3) if kernel is None else kernel

        self.inc = DoubleConv(n_channels, 16, dropout_prob=0.3)
        self.down1 = Down(16, 32, down_layer="avgpool", drop=0.37)
        self.down2 = Down(32, 64, drop=0.43)
        self.down3 = Down(64, 128, drop=0.5)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, int(256 // factor), drop=0.4)
        self.up1 = Up(256, int(128 // factor), bilinear, drop=0.5)
        self.up2 = Up(128, int(64 // factor), bilinear, drop=0.43)
        self.up3 = Up(64, int(32 // factor), bilinear, drop=0.37)
        self.up4 = Up(32, 16, bilinear, drop=0.3)
        self.outc = OutConv(16, n_classes)

        '''
        Init loss functions
        '''
        self.mse = nn.MSELoss()
        self.dice_loss = losses.dice_coeff_loss

        '''
        Init metric function
        '''
        self.metric_fn = losses.dice_coeff

        self.save_hyperparameters()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch, batch_idx):
        image, mask, filename = batch
        pred_mask = self(image)
        loss = self.compute_losses(y_true=mask, y_pred=pred_mask)
        metric = self.metric_fn(pred=pred_mask, target=mask)

        # Define loggers
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("my_metric", metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        image, mask, filename = batch
        pred_mask = self(image)
        return pred_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def init_model(self):
        pass

    def compute_losses(self, y_true, y_pred):
        return self.mse(input=y_pred, target=y_true) + 1 - self.dice_loss(pred=y_pred, target=y_true)

    @staticmethod
    def add_model_specific_args(parent_parser):
        utils.set_cwd()
        project_root = os.getcwd()

        parser = parent_parser.add_argument_group('ModelConfig')
        parser.add_argument("--model_config_path", type=str, default="",
                            help="path to model config file")

        parser.add_argument('--input_channels', type=int, default=3,
                            help='number of channels of input data')

        parser.add_argument('--num_classes', type=int, default=1,
                            help='number of channels of output data')

        parser.add_argument("--down_layer", type=str, default="maxpool", choices=["maxpool", "avgpool"],
                            help="Type of downsampling layer")

        parser.add_argument("--feature_list", type=int, nargs="+", default=[16, 32, 64, 128, 256],
                            help="list of up and down layer filters")

        parser.add_argument('--use_dropout', action='store_true',
                            help='use dropout layers')

        parser.add_argument('--use_batchnorm', action='store_true',
                            help='use batch normalisation')

        parser.add_argument("--drop", type=int, nargs="+", default=[0.3, 0.37, 0.43, 0.5, 0.5],
                            help="list of dropout values")

        parser.add_argument('--kernel_size', type=int, default=3,
                            help='size of convolution kernel')

        parser.add_argument('--bilinear', action='store_true',
                            help='use bilinear upsampling layer')

        parser.add_argument('--deterministic_up', action='store_true',
                            help='use deterministic upsampling layer')
        return parent_parser








