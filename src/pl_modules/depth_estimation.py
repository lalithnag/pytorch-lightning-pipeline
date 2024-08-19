import os
import sys
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Append project dir to sys path so other modules are available
current_dir = os.getcwd() 
project_root = os.path.dirname(os.path.dirname(current_dir))  
sys.path.append(project_root)

from src.pl_modules import resnet_encoder
from src.pl_modules import depth_decoder
from src.pl_modules import layers

from src.training import models
from src.utils import utils
from src.training import losses
from src.evaluation import matching
from src.visualisation import vis
from torchvision.utils import save_image


class DepthNet(pl.LightningModule):
    """
    Lightning class for the Unet module, that has configurable options of dropout, batchnorm and depth
    This unet returns one output, this can be further used in an extensible manner
    Saves all the hyperparams to the hparams.yaml in the model definition
    """
    def __init__(self,
                 num_layers: int=18,
                 weights_init: str="pretrained",
                 scales=[0],
                 lr: float=0.001,
                 batch_size=batch_size,
                 height=448,
                 width=448,
                 **kwargs):
        super(DepthNet, self).__init__()
        #model_hyperparams = {items: values for items, values in locals().items() if items!='kwargs'}

        '''
        Init input arguments
        '''
        #self.hparams = hparams
        self.num_layers = num_layers
        self.weights_init = weights_init
        self.scales = scales
        self.encoder = resnet_encoder.ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
        self.parameters_to_train += list(self.encoder.parameters())

        self.depth_decoder = depth_decoder.DepthDecoder(self.encoder.num_ch_enc, self.scales)
        self.parameters_to_train += list(self.depth_decoder.parameters())

        self.height = height
        self.width = width
        self.batch_size = self.batch_size
        self.backproject_depth = layers.BackprojectDepth(self.batch_size, self.height, self.width)
        self.project_3d = layers.Project3D(self.batch_size, self.height, self.width)
        '''
        Init loss and metric functions
        '''
        self.lr = lr

        self.mse = nn.MSELoss()
        self.dice_coeff = dice_coeff
        self.metric_fn = metric_fn

        self.save_hyperparameters()

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

