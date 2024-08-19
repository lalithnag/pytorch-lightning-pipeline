import os
import sys
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(os.path.dirname(current_dir))  # ~/endoscopic-image-analysis
sys.path.append(project_root)

from src.training import models
from src.utils import utils
from src.training import losses
from src.evaluation import matching
from src.visualisation import vis
from torchvision.utils import save_image
from src.pl_modules.convlstm import ConvLSTM
from src.pl_modules.shared_encoder import *

from kornia.geometry import conv_soft_argmax2d
from kornia.filters import gaussian_blur2d


class Unet(pl.LightningModule):
    """
    Lightning class for the Unet module, that has configurable options of dropout, batchnorm and depth
    This unet returns one output, this can be further used in an extensible manner
    Saves all the hyperparams to the hparams.yaml in the model definition
    """
    def __init__(self, input_channels: int = 3,
                 num_classes: int = 1,
                 down_layer: str = "maxpool",
                 feature_list: list = None,
                 use_dropout: bool = True,
                 use_batchnorm: bool = True,
                 drop: list or int = None,
                 kernel_size: int = 3,
                 bilinear: bool = False,
                 deterministic_up: bool = False,
                 conv_block: nn.Module = models.DoubleConv,
                 down_conv: nn.Module = models.Down,
                 up_conv: nn.Module = models.Up,
                 out_conv: nn.Module = models.OutConv,
                 dice_coeff=losses.dice_coeff,
                 metric_fn=losses.dice_coeff,
                 init_type: str = "kaiming",
                 distribution: str = "normal",
                 lr: float = 0.001,
                 **kwargs):
        super(Unet, self).__init__()
        #model_hyperparams = {items: values for items, values in locals().items() if items!='kwargs'}

        '''
        Init input arguments
        '''
        #self.hparams = hparams
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.down_layer = down_layer
        self.feature_list = feature_list if feature_list else [16, 32, 64, 128, 256]
        self.feature_list_flipped = np.flip(self.feature_list)
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        if drop: self.drop = [drop] * len(self.feature_list) if isinstance(drop, int) else drop
        else: self.drop = [0.3, 0.37, 0.43, 0.5, 0.5]
        assert (isinstance(self.drop, list) and len(self.drop)==len(self.feature_list)) or isinstance(self.drop, int), \
            "The dropout value has to be either an integer or a list of length of number of features"
        self.drop_flipped = np.flip(self.drop)

        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self.deterministic_up = deterministic_up
        self.factor = 2 if self.deterministic_up or self.bilinear else 1

        '''
        Init model layer definitions
        '''
        self.conv = conv_block(in_channels=self.input_channels,
                               out_channels=self.feature_list[0],
                               use_dropout=self.use_dropout,
                               use_batchnorm=self.use_batchnorm,
                               kernel_size=self.kernel_size,
                               dropout_prob=self.drop[0])

        self.down_conv = down_conv
        self.up_conv = up_conv
        self.out_conv = out_conv
        # Using a PyTorch nn module list does not work with the lightning checkpoint
        # So we need to use a list and then unpack it with lightning sequential
        self.down_layers = []
        self.up_layers = []

        # Encoder path (downsampling path)
        for i in range(len(self.feature_list)-1):
            if i+1 == len(self.feature_list)-1:
                # For the last downsampling layer, need to include factor depending on bilinear flag
                self.down_layers.append(self.down_conv(in_channels=self.feature_list[i],
                                                       out_channels=self.feature_list[i + 1]//self.factor,
                                                       down_layer=self.down_layer,
                                                       drop=self.drop[i + 1]))
            else:
                self.down_layers.append(self.down_conv(in_channels=self.feature_list[i],
                                                       out_channels=self.feature_list[i+1],
                                                       down_layer=self.down_layer,
                                                       drop=self.drop[i+1]))
        self.down_layers = nn.Sequential(*self.down_layers)

        # Decoder path (Upsampling path)
        for i in range(len(self.feature_list)-1):
            if i + 1 == len(self.feature_list_flipped)-1:
                self.up_layers.append(self.up_conv(in_channels=self.feature_list_flipped[i],
                                                   out_channels=(self.feature_list_flipped[i + 1]),
                                                   bilinear=self.bilinear,
                                                   deterministic=self.deterministic_up,
                                                   drop=self.drop_flipped[i]))
            else:
                self.up_layers.append(self.up_conv(in_channels=self.feature_list_flipped[i],
                                                   out_channels=(self.feature_list_flipped[i+1]//self.factor),
                                                   bilinear=self.bilinear,
                                                   deterministic=self.deterministic_up,
                                                   drop=self.drop_flipped[i]))
        self.up_layers = nn.Sequential(*self.up_layers)

        self.outconv = self.out_conv(in_channels=self.feature_list[0],
                                     out_channels=self.num_classes)

        '''
        Init loss and metric functions
        '''
        self.mse = nn.MSELoss()
        self.dice_coeff = dice_coeff
        self.metric_fn = metric_fn

        self.init_type = init_type
        self.distribution = distribution
        self.lr = lr

        # Save all model hyperparams to the hparams.yaml file
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv(x)
        outputs = []

        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i + 1)])
        x = self.outconv(x)
        return x

    def training_step(self, batch, batch_idx):
        image, mask, filename = batch
        pred_mask = self(image)
        train_loss = self.compute_losses(y_true=mask, y_pred=pred_mask)
        train_metric = self.metric_fn(pred=pred_mask, target=mask)
        self.log('train_loss_step', train_loss)
        return {'loss': train_loss, 'metric': train_metric.detach()}

    def training_epoch_end(self, outputs):
        # Log the mean of the loss and metric at the end of every training epoch
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_train_metric = torch.stack([x['metric'] for x in outputs]).mean() * 100
        self.logger.experiment.add_scalars("loss", {"train_loss": avg_train_loss}, self.current_epoch+1)
        self.logger.experiment.add_scalar('metric', avg_train_metric, self.current_epoch+1)

    def validation_step(self, batch, batch_idx):
        image, mask, filename = batch
        pred_mask = self(image)
        val_loss = self.compute_losses(y_true=mask, y_pred=pred_mask)
        return val_loss

    def validation_epoch_end(self, outputs):
        # Log the validation loss at the end of every validation epoch
        avg_val_loss = torch.stack([x for x in outputs]).mean()
        self.logger.experiment.add_scalars("loss", {"val_loss": avg_val_loss}, self.current_epoch+1)

    def test_step(self, batch, batch_idx):
        image, gt_points, filename = batch
        pred_mask = self(image)
        pred_mask_np = pred_mask.detach().cpu().clone().squeeze(0).numpy().transpose((2, 1, 0))

        gt_points = self.convert_points_to_numpy(gt_points)
        mask_regions = gt_points
        pred_regions = matching.centres_of_mass(pred_mask, self.centre_of_mask_threshold)

        rel_folder, frame_number, side = self.test_dataset.get_split_filename(filename[0])

        print(self.save_pred_points, self.save_pred_mask, self.save_annotated)
        if self.save_pred_points:
            json_name = "{:06d}{}".format(int(frame_number), ".json")
            json_path = os.path.join(self.save_path, rel_folder,
                                     "pred_points_0{}".format(self.side_map[side]),
                                     json_name)
            vis.save_points(pred_mask_np[0, ...], json_path)

        if self.save_pred_mask:
            pred_mask_name = "{:06d}{}".format(int(frame_number), ".png")
            pred_mask_path = os.path.join(self.save_path, rel_folder,
                                          "pred_mask_0{}".format(self.side_map[side]),
                                          pred_mask_name)
            save_image(pred_mask, pred_mask_path)

        if self.save_annotated:
            annotated_image_name = "{:06d}{}".format(int(frame_number), ".png")
            annotated_path = os.path.join(self.save_path, rel_folder,
                                          "annotated_0{}".format(self.side_map[side]),
                                          annotated_image_name)
            save_image(image, annotated_path)
            image_cv = cv2.imread(annotated_path)
            annotated_image = vis.get_annotated_from_points(image=image_cv,
                                                            point=gt_points,
                                                            pred_mask=pred_mask_np[0, ..., 0])
            cv2.imwrite(annotated_path, annotated_image)

        matched = matching.match_keypoints(pred_regions, mask_regions, self.match_pixel_threshold)
        result = {
            "total": len(pred_regions),
            "TP": len(matched),
            "FP": len(pred_regions) - len(matched),
            "FN": len(mask_regions) - len(matched)
        }
        self.logger.experiment.add_scalars("test", result, batch_idx)
        return result

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        print("LR: ", self.lr)
        reduce_on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                                 patience=10,
                                                                                 min_lr=1e-10,
                                                                                 factor=0.1)
        return {"optimizer": optimizer,
                "lr_scheduler": { "scheduler": reduce_on_plateau_scheduler,
                                  "monitor": 'train_loss_step'},
                }

    def init_weights(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                if not ('BatchNorm' in module.__class__.__name__):
                    if self.distribution == "uniform":
                        if self.init_type == "kaiming":
                            # Default non-linearity for kaiming is leaky-relu
                            torch.nn.init.kaiming_uniform_(module.weight)
                        elif self.init_type == "xavier":
                            torch.nn.init.xavier_uniform_(module.weight)
                        else:
                            torch.nn.init.uniform_(module.weight)
                    else:
                        if self.init_type == "kaiming":
                            torch.nn.init.kaiming_normal_(module.weight)
                        elif self.init_type == "xavier":
                            torch.nn.init.xavier_normal_(module.weight)
                        else:
                            torch.nn.init.normal_(module.weight)
                else:
                    torch.nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def compute_losses(self, y_true, y_pred, *args, **kwargs):
        return self.mse(input=y_pred, target=y_true) + 1 - losses.dice_coeff(pred=y_pred, target=y_true)

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

        parser.add_argument("--init_type", type=str, default="kaiming", choices=["kaiming", "xavier"],
                            help="Type of init method to use for the model weights")

        parser.add_argument("--distribution", type=str, default="normal", choices=["uniform", "normal"],
                            help="Type of distirubtion to use for init'ing the model weights")
        return parent_parser


class UnetGaussian(Unet):
    """
    Lightning class for the modified Unet module that uses a Gaussian layer and
    a learnable softargmax layer from Kornia
    Saves all the hyperparams to the hparams.yaml in the model definition
    """
    def __init__(self, kernel=None,
                 sigma_2=2,
                 **kwargs):
        super(UnetGaussian, self).__init__(**kwargs)
        self.kernel = (3, 3) if kernel is None else kernel
        self.gauss = gaussian_blur2d
        self.softargmax = conv_soft_argmax2d
        self.sigma_2 = (sigma_2, sigma_2)

    def forward(self, x):
        x = self.conv(x)
        outputs = []

        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i + 1)])

        x = self.outconv(x)
        x = self.gauss(x, self.kernel, sigma=self.sigma_2)  # Local gaussian smoothing of the max values
        pred_coords, pred_mask = self.softargmax(x, output_value=True)
        return pred_mask, x

    def training_step(self, batch, batch_idx):
        image, mask, filename = batch
        final_layer_pred_mask, gaussian_layer_pred_mask  = self(image)
        train_loss = self.compute_losses(y_true_gaussian_layer=mask, y_pred_gaussian_layer=gaussian_layer_pred_mask,
                                         y_true=mask, y_pred=final_layer_pred_mask)
        train_metric = self.metric_fn(pred=final_layer_pred_mask, target=mask)
        self.log('train_loss_step', train_loss)
        return {'loss': train_loss, 'metric': train_metric.detach()}

    def validation_step(self, batch, batch_idx):
        image, mask, filename = batch
        final_layer_pred_mask, gaussian_layer_pred_mask  = self(image)
        val_loss = self.compute_losses(y_true_gaussian_layer=mask, y_pred_gaussian_layer=gaussian_layer_pred_mask,
                                       y_true=mask, y_pred=final_layer_pred_mask)
        return val_loss

    def test_step(self, batch, batch_idx):
        image, gt_points, filename = batch
        pred_mask = self(image)[1]
        pred_mask_np = pred_mask.detach().cpu().clone().squeeze(0).numpy()

        gt_points = self.convert_points_to_numpy(gt_points)
        mask_regions = gt_points
        pred_regions = matching.centres_of_mass(pred_mask_np.transpose((2, 1, 0))[..., 0], self.centre_of_mask_threshold)

        rel_folder, frame_number, side = filename[0].split()

        if self.save_pred_points:
            json_name = "{:06d}{}".format(int(frame_number), ".json")
            json_path = os.path.join(self.save_path, rel_folder,
                                     "pred_points_0{}".format(self.side_map[side]),
                                     json_name)
            vis.save_points(pred_mask_np[0, ...], json_path)

        if self.save_pred_mask:
            pred_mask_name = "{:06d}{}".format(int(frame_number), ".png")
            pred_mask_path = os.path.join(self.save_path, rel_folder,
                                          "pred_mask_0{}".format(self.side_map[side]),
                                          pred_mask_name)
            save_image(pred_mask, pred_mask_path)

        if self.save_annotated:
            annotated_image_name = "{:06d}{}".format(int(frame_number), ".png")
            annotated_path = os.path.join(self.save_path, rel_folder,
                                          "annotated_0{}".format(self.side_map[side]),
                                          annotated_image_name)
            save_image(image, annotated_path)
            image_cv = cv2.imread(annotated_path)
            annotated_image = vis.get_annotated_from_points(image=image_cv,
                                                              point=gt_points,
                                                              pred_mask=pred_mask_np[0, ...])
            cv2.imwrite(annotated_path, annotated_image)

        matched = matching.match_keypoints(pred_regions, mask_regions, self.match_pixel_threshold)
        result = {
            "total": len(pred_regions),
            "TP": len(matched),
            "FP": len(pred_regions) - len(matched),
            "FN": len(mask_regions) - len(matched)
        }
        self.logger.experiment.add_scalars("test", result, batch_idx)
        return result

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def compute_losses(self, y_true, y_pred,
                       y_true_gaussian_layer, y_pred_gaussian_layer, *args, **kwargs):
        gaussian_layer_loss = self.mse(input=y_pred_gaussian_layer, target=y_true_gaussian_layer) + \
                              1 - losses.dice_coeff(pred=y_pred_gaussian_layer, target=y_true_gaussian_layer)
        final_layer_loss = self.mse(input=y_pred, target=y_true) + \
                              1 - losses.dice_coeff(pred=y_pred, target=y_true)
        return 0.5 * gaussian_layer_loss + 0.5 * final_layer_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        utils.set_cwd()
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

        parser.add_argument("--init_type", type=str, default="kaiming", choices=["kaiming", "xavier"],
                            help="Type of init method to use for the model weights")

        parser.add_argument("--distribution", type=str, default="normal", choices=["uniform", "normal"],
                            help="Type of distirubtion to use for init'ing the model weights")

        parser.add_argument("--sigma_2", type=int, default=2,
                            help='sigma size of gaussian layer in model')
        return parent_parser


class ConvLSTMUnet(UnetGaussian):
    def __init__(self, lstm_layers=1, **kwargs):
        super(ConvLSTMUnet, self).__init__(**kwargs)
        self.lstm_layers = lstm_layers
        self.conv_lstm = ConvLSTM(input_dim=self.feature_list[-2],
                                  hidden_dim=self.feature_list[-2],
                                  kernel_size=(self.kernel_size, self.kernel_size),
                                  num_layers=self.lstm_layers)

    def forward(self, x):
        x = self.conv(x)
        outputs = []

        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)

        x = self.conv_lstm(x.unsqueeze(0))[0][0].squeeze(1)

        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i + 1)])

        x = self.outconv(x)
        x = self.gauss(x, self.kernel, sigma=self.sigma_2)  # Local gaussian smoothing of the max values
        pred_coords, pred_mask = self.softargmax(x, output_value=True)
        return pred_mask, x


class SharedEncoderUnet(UnetGaussian):
    def __init__(self, **kwargs):
        super(SharedEncoderUnet, self).__init__(**kwargs)
        self.down_layers = EncoderUnet(self.feature_list,
                                       self.conv,
                                       self.down_conv,
                                       self.down_layer,
                                       self.factor,
                                       self.drop)

        self.merge_layer = MergeUnet(self.feature_list,
                                     self.factor)

        self.up_layers = DecoderUnet(self.feature_list,
                                     self.up_conv,
                                     self.bilinear,
                                     self.deterministic_up,
                                     self.factor,
                                     self.drop_flipped)

    def forward(self, x):
        # Split the input data into A and B views by always grouping the 3 image channels
        split_x = [torch.split(stereo_image[0], 3) for stereo_image in torch.split(x, 1)]
        view_size = (x.size()[0], 3, x.size()[2], x.size()[3])

        x_A = torch.cat([stereo_image[0] for stereo_image in split_x], 0).reshape(view_size)
        x_B = torch.cat([stereo_image[1] for stereo_image in split_x], 0).reshape(view_size)

        # Encode both views seperately with the same shared encoder to share the weights
        x_A, outputs_A = self.down_layers(x_A)
        x_B, outputs_B = self.down_layers(x_B)

        # Merge their outputs to use in decoding
        x = torch.cat([x_A, x_B], 1)
        outputs = [torch.cat([output_A, output_B], 1) for output_A, output_B in zip(outputs_A, outputs_B)]
        x, outputs = self.merge_layer(x, outputs)

        # Decode
        x = self.up_layers(x, outputs)
        x = self.outconv(x)
        x = self.gauss(x, self.kernel, sigma=self.sigma_2)  # Local gaussian smoothing of the max values
        pred_coords, pred_mask = self.softargmax(x, output_value=True)
        return pred_mask, x
