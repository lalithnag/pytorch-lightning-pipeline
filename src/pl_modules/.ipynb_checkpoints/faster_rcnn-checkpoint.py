import os
import sys
import numpy as np
import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.detection import rpn
import torchvision.models as models
#import training

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(os.path.dirname(current_dir))  # ~/endoscopic-image-analysis
sys.path.append(project_root)

from src.utils import utils
from src.training import losses


class ResnetRegionProposalNetwork(torch.nn.Module):
    def __init__(self):
        super(ResnetRegionProposalNetwork, self).__init__()

        # --------- Backbone ---------------- #
        # Define backbone with the trainable layers
        self.resnet_backbone = torch.nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-3])
        non_trainable_backbone_layers = 5
        counter = 0
        for child in self.resnet_backbone:
            if counter < non_trainable_backbone_layers:
                for param in child.parameters():
                    param.requires_grad = False
                counter += 1
            else:
                break

        # ------ Anchor generation ---------- #
        # Define all the anchors
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.rpn_anchor_generator = rpn.AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        out_channels = 1024
        self.rpn_head = rpn.RPNHead(
            out_channels, self.rpn_anchor_generator.num_anchors_per_location()[0]
        )

        # --------- Proposal layer ---------- #
        rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}
        rpn_post_nms_top_n = {"training": 2000, "testing": 1000}
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.2
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5

        self.rpn = rpn.RegionProposalNetwork(
            self.rpn_anchor_generator, self.rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)


    def forward(self,
                images,       # type: ImageList
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        feature_maps = self.resnet_backbone(images)
        features = {"0": feature_maps}
        image_sizes = getImageSizes(images)  # Feed the image sizes but if its std # 488x211
        image_list = il.ImageList(images, image_sizes)  # ???
        return self.rpn(image_list, features, targets)  # ???