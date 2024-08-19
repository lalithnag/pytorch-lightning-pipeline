# Imports
import os
import sys

import cv2
import torch
import random
import numpy as np

import pytorch_lightning as pl
import albumentations as alb

from torch.utils.data import DataLoader
import albumentations.augmentations.transforms as alb_tr

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(project_root)

from src.utils import utils
from src.data.gen_suture_masks import GenSutureMasks
from src.dataset_preproc.dataloader import *


def seed_worker(worker_id):
    """Function to initialize the seed for each worker
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ConditionError(Exception):
    pass


class EndoDataModule(pl.LightningDataModule):
    """
    Lightning data module for the endoscopic dataset, in the simulator and the OR domains
    The data is present either in the kitti or the adaptor data format
    It is the class' responsibility to save the parameters that is created in the hyperparams yaml file
    """
    def __init__(self, dataroot: str,
                 split_dir: str,
                 data_split: str,
                 height: int = 448,
                 width: int = 448,
                 aug: bool = False,
                 aug_prob: float = 0.5,
                 data_format: str = "kitti",
                 image_ext: str = ".png",
                 in_memory: bool = False,
                 fold: int = 1,
                 num_workers: int = 32,
                 batch_size: int = 1,
                 fake: bool = False,
                 load_from_disk: bool = False,
                 mask_path_suffix: str = "",
                 binary: bool = False,
                 blur_func: str = "gaussian",
                 spread: int = 2,
                 **kwargs):

        super(EndoDataModule, self).__init__()

        self.dataroot = dataroot
        self.split_dir = split_dir
        self.data_split = data_split
        self.height = height

        self.width = width
        self.aug = aug
        self.aug_prob = aug_prob
        self.data_format = data_format
        self.image_ext = image_ext
        self.in_memory = in_memory
        self.fold = ("fold_" + str(fold)) if not fake else ""
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.fake = fake
        self.load_from_disk = load_from_disk
        self.mask_path_suffix = mask_path_suffix
        self.binary = binary
        self.blur_func = blur_func
        self.spread = spread

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.image_aug = None
        self.image_mask_aug = None
        self.GT = True
        self.split_file_path = os.path.join(split_dir, data_split, self.fold, "{}_files.txt")

        if self.aug:
            # Define image specific augmentation
            self.image_aug = alb.Compose([alb.Resize(height=self.height, width=self.width),
                                          alb_tr.ColorJitter(brightness=0.2,
                                                             contrast=(0.3, 1.5),
                                                             saturation=(0.5, 2),
                                                             hue=0.1,
                                                             p=self.aug_prob)])
            # Define image+mask specific augmentation
            self.image_mask_aug = alb.Compose([alb.Rotate(limit=(-60, 60),
                                                          p=self.aug_prob),
                                               alb.IAAAffine(translate_percent=10, shear=0.1,
                                                             p=self.aug_prob),
                                               alb.HorizontalFlip(p=self.aug_prob),
                                               alb.VerticalFlip(p=self.aug_prob)])

        # Save image augmentations to config file
        self.aug_dict = {"image_aug": alb.to_dict(self.image_aug) if self.image_aug else None,
                         "image_mask_aug": alb.to_dict(self.image_mask_aug) if self.image_mask_aug else None}

        # ::TODO: Save only the parameters created by the parser in this class, instead of all
        #self.save_hyperparameters()

    def setup(self, stage=None):
        """
        Get all the filenames from the split path
        Define the PyTorch datasets that either load image and mask or create a mask on the fly
        depending on if masks are available
        Define image specific and image+mask specific augmentations
        """

        if stage in (None, "fit"):
            # Get filenames from split-path
            train_filenames = utils.read_lines_from_text_file(self.split_file_path.format("train"))

            # Try-except block for the train data generator
            try:
                # If the condition (load from disk) is not satisfied or if there is an exception,
                # then on-the-fly dataloader is used
                if not self.load_from_disk: raise ConditionError
                self.train_dataset = EndoMaskDataset(data_root_folder=self.dataroot,
                                                     filenames=train_filenames,
                                                     height=self.height,
                                                     width=self.width,
                                                     image_aug=self.image_aug,
                                                     aug_prob=self.aug_prob,
                                                     image_ext=self.image_ext,
                                                     data_format=self.data_format,
                                                     image_mask_aug=self.image_mask_aug,
                                                     mask_path_suffix=self.mask_path_suffix)
                # Try to get a data sample from dataloader
                sample_train_output = self.train_dataset.__getitem__(0)

            except(ConditionError, FileNotFoundError):
                self.train_dataset = SutureDataset(data_root_folder=self.dataroot,
                                                   filenames=train_filenames,
                                                   height=self.height,
                                                   width=self.width,
                                                   image_aug=self.image_aug,
                                                   aug_prob=self.aug_prob,
                                                   image_ext=self.image_ext,
                                                   data_format=self.data_format,
                                                   image_mask_aug=self.image_mask_aug,
                                                   binary=self.binary,
                                                   blur_func=self.blur_func,
                                                   spread=self.spread)

                # Try-except block for the validation data generator
                if os.path.isfile(self.split_file_path.format("val")):
                    val_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                    try:
                        # If the condition (load from disk) is not satisfied or if there is an exception,
                        # then on-the-fly dataloader is used
                        if not self.load_from_disk: raise ConditionError
                        self.val_dataset = EndoMaskDataset(data_root_folder=self.dataroot,
                                                           filenames=val_filenames,
                                                           height=self.height,
                                                           width=self.width,
                                                           image_aug=None,
                                                           image_ext=self.image_ext,
                                                           data_format=self.data_format,
                                                           image_mask_aug=None,
                                                           mask_path_suffix=self.mask_path_suffix)
                        # Try to get a data sample from dataloader
                        sample_val_output = self.val_dataset.__getitem__(0)

                    except(ConditionError, FileNotFoundError):
                        self.val_dataset = SutureDataset(data_root_folder=self.dataroot,
                                                         filenames=val_filenames,
                                                         height=self.height,
                                                         width=self.width,
                                                         image_aug=None,
                                                         image_ext=self.image_ext,
                                                         data_format=self.data_format,
                                                         image_mask_aug=None,
                                                         binary=self.binary,
                                                         blur_func=self.blur_func,
                                                         spread=self.spread)

            if stage in (None, "test"):
                test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                self.test_dataset = SutureDataset(data_root_folder=self.dataroot,
                                                  filenames=test_filenames,
                                                  height=self.height,
                                                  width=self.width,
                                                  image_aug=None,
                                                  image_ext=self.image_ext,
                                                  data_format=self.data_format,
                                                  image_mask_aug=None,
                                                  binary=self.binary,
                                                  blur_func=self.blur_func,
                                                  spread=self.spread,
                                                  test=True)

            if stage in (None, "predict"):
                test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                self.test_dataset = EndoDataset(data_root_folder=self.dataroot,
                                                filenames=test_filenames,
                                                height=self.height,
                                                width=self.width,
                                                image_aug=None,
                                                image_ext=self.image_ext,
                                                data_format=self.data_format)

    # Set up stage dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=False,
                          worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          worker_init_fn=seed_worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker)

    def get_aug_dict(self):
        return {"image_aug": alb.to_dict(self.image_aug),
                "image_mask_aug": alb.to_dict(self.image_mask_aug)}

    @staticmethod
    def add_data_specific_args(parent_parser):
        utils.set_cwd()
        project_root = os.getcwd()

        parser = parent_parser.add_argument_group('EndoDataConfig')
        parser.add_argument("--data_config_path", type=str, default="",
                            help="path to data config file")

        parser.add_argument("--dataroot", type=str, default="/home/lalith/data/11_mkr_challenge_dataset",
                            help="path where the datasets are located")

        parser.add_argument("--split_dir", type=str, default=os.path.join(project_root, "splits"),
                            help="split directory")

        parser.add_argument("--data_split", type=str, default="mkr_challenge_lr",
                            help="name of split located in split dir")

        parser.add_argument("--height", type=int, default=288,
                            help="input image height")

        parser.add_argument("--width", type=int, default=512,
                            help="input image width")

        parser.add_argument("--aug", action="store_true",
                            help="if set, trains from raw KITTI png files (instead of jpgs)")

        parser.add_argument('--aug_prob', type=float, default=0.5,
                            help='Probability to apply image+mask augmentations')

        parser.add_argument("--data_format", type=str, default="kitti", choices=["kitti", "mono"],
                            help="kitti or mono dataset format")

        parser.add_argument("--image_ext", type=str, default=".png", choices=[".npy", ".png"],
                            help="image extension")

        parser.add_argument("--in_memory", action="store_true",
                            help="if set, trains from raw KITTI png files (instead of jpgs)")

        parser.add_argument('--fold', type=int, default=1,
                            help='CV fold')

        parser.add_argument("--num_workers", type=int, default=32,
                            help="number of workers for parallel processing")

        parser.add_argument('--batch_size', type=int, default=1,
                            help='size of the batches')

        parser.add_argument('--fake', action='store_true',
                            help='Loads split from fake data without fold')

        parser.add_argument('--load_from_disk', action='store_true',
                            help='Loads masks from disk, else generates masks on-the-fly')

        parser.add_argument('--mask_path_suffix', type=str, default="_gaussian_2",
                            help='Suffix for mask path from where the masks are to be loaded'
                                 'Useful only if loading from disk')

        parser.add_argument('--binary', action='store_true',
                            help='If set, generates on-the-fly binary masks')

        parser.add_argument('--blur_func', type=str, default="gaussian", choices=["gaussian", "tanh"],
                            help='The blur func to be applied to the on-the-fly generated masks')

        parser.add_argument('--spread', type=int, default=2,
                            help='The spread of the blur func to be applied to on-the-fly generated masks')
        return parent_parser

    @staticmethod
    def get_parser_arg_names(parser):
        return set(vars(parser.parse_known_args()[0]).keys())


class StereoEndoDataModule(EndoDataModule):
    def setup(self, stage=None):
        """
        Get all the filenames from the split path
        Define the PyTorch datasets that either load image and mask or create a mask on the fly
        depending on if masks are available
        Define image specific and image+mask specific augmentations
        """

        if stage in (None, "fit"):
            # Get filenames from split-path
            train_filenames = utils.read_lines_from_text_file(self.split_file_path.format("train"))

            # Try-except block for the train data generator
            try:
                # If the condition (load from disk) is not satisfied or if there is an exception,
                # then on-the-fly dataloader is used
                if not self.load_from_disk: raise ConditionError
                self.train_dataset = StereoEndoMaskDataset(data_root_folder=self.dataroot,
                                                           filenames=train_filenames,
                                                           height=self.height,
                                                           width=self.width,
                                                           image_aug=self.image_aug,
                                                           aug_prob=self.aug_prob,
                                                           image_ext=self.image_ext,
                                                           data_format=self.data_format,
                                                           image_mask_aug=self.image_mask_aug,
                                                           mask_path_suffix=self.mask_path_suffix)
                # Try to get a data sample from dataloader
                sample_train_output = self.train_dataset.__getitem__(0)

            except(ConditionError, FileNotFoundError):
                self.train_dataset = StereoSutureDataset(data_root_folder=self.dataroot,
                                                         filenames=train_filenames,
                                                         height=self.height,
                                                         idth=self.width,
                                                         image_aug=self.image_aug,
                                                         aug_prob=self.aug_prob,
                                                         image_ext=self.image_ext,
                                                         data_format=self.data_format,
                                                         image_mask_aug=self.image_mask_aug,
                                                         binary=self.binary,
                                                         blur_func=self.blur_func,
                                                         spread=self.spread)

            # Try-except block for the validation data generator
            if os.path.isfile(self.split_file_path.format("val")):
                val_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                try:
                    # If the condition (load from disk) is not satisfied or if there is an exception,
                    # then on-the-fly dataloader is used
                    if not self.load_from_disk: raise ConditionError
                    self.val_dataset = StereoEndoMaskDataset(data_root_folder=self.dataroot,
                                                             filenames=val_filenames,
                                                             height=self.height,
                                                             width=self.width,
                                                             image_aug=None,
                                                             image_ext=self.image_ext,
                                                             data_format=self.data_format,
                                                             image_mask_aug=None,
                                                             mask_path_suffix=self.mask_path_suffix)
                    # Try to get a data sample from dataloader
                    sample_val_output = self.val_dataset.__getitem__(0)

                except(ConditionError, FileNotFoundError):
                    self.val_dataset = StereoSutureDataset(data_root_folder=self.dataroot,
                                                           filenames=val_filenames,
                                                           height=self.height,
                                                           width=self.width,
                                                           image_aug=None,
                                                           image_ext=self.image_ext,
                                                           data_format=self.data_format,
                                                           image_mask_aug=None,
                                                           binary=self.binary,
                                                           blur_func=self.blur_func,
                                                           spread=self.spread)

        if stage in (None, "test"):
            test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
            self.test_dataset = StereoSutureDataset(data_root_folder=self.dataroot,
                                                    filenames=test_filenames,
                                                    height=self.height,
                                                    width=self.width,
                                                    image_aug=None,
                                                    image_ext=self.image_ext,
                                                    data_format=self.data_format,
                                                    image_mask_aug=None,
                                                    binary=self.binary,
                                                    blur_func=self.blur_func,
                                                    spread=self.spread,
                                                    test=True)

        if stage in (None, "predict"):
            test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
            self.test_dataset = StereoEndoDataset(data_root_folder=self.dataroot,
                                                  filenames=test_filenames,
                                                  height=self.height,
                                                  width=self.width,
                                                  image_aug=None,
                                                  image_ext=self.image_ext,
                                                  data_format=self.data_format)


class EndoDataModuleAB(EndoDataModule):
    def __init__(self, dataroot_A:str,
                 dataroot_B:str,
                 data_split_a: str,
                 data_split_b: str,
                 randomise:bool,
                 **kwargs):
        super(EndoDataModuleAB, self).__init__(**kwargs)

        self.dataroot_A = dataroot_A
        self.dataroot_B = dataroot_B
        self.data_split_A = data_split_a
        self.data_split_B = data_split_b
        self.randomise = randomise

        self.split_file_path_A = os.path.join(self.split_dir, self.data_split_A, self.fold, "{}_files.txt")
        self.split_file_path_B = os.path.join(self.split_dir, self.data_split_B, self.fold, "{}_files.txt")

        if self.aug:
            # Define image specific augmentation
            self.image_aug = alb.Compose([alb.Resize(height=self.height, width=self.width),
                                          alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            # Define image+mask specific augmentation
            self.image_mask_aug = alb.Compose([alb.Rotate(limit=(-60, 60), p=self.aug_prob),
                                               alb.HorizontalFlip(p=0.5),
                                               alb.VerticalFlip(p=0.5)])

        # Save image augmentations to config file
        self.aug_dict = {"image_aug": alb.to_dict(self.image_aug) if self.image_aug else None,
                         "image_mask_aug": alb.to_dict(self.image_mask_aug) if self.image_mask_aug else None}

        def setup(self, stage=None):
            """
            Get all the filenames from the split path
            Define the PyTorch datasets that either load image and mask or create a mask on the fly
            depending on if masks are available
            Define image specific and image+mask specific augmentations
            """

            if stage in (None, "fit"):
                # Get filenames from split-path
                train_filenames_A = utils.read_lines_from_text_file(self.split_file_path_A.format("train"))
                train_filenames_B = utils.read_lines_from_text_file(self.split_file_path_B.format("train"))

                # Try-except block for the train data generator
                try:
                    # If the condition (load from disk) is not satisfied or if there is an exception,
                    # then on-the-fly dataloader is used
                    if not self.load_from_disk: raise ConditionError
                    self.train_dataset = ConcatDataset(EndoDataset(data_root_folder=dataroot_A,
                                                                   filenames=train_filenames_A,
                                                                   height=self.height,
                                                                   width=self.width,
                                                                   image_aug=self.image_aug,
                                                                   aug_prob=self.aug_prob,
                                                                   image_ext=self.image_ext,
                                                                   data_format=self.data_format),

                                                       EndoDataset(data_root_folder=dataroot_B,
                                                                   filenames=train_filenames_B,
                                                                   height=self.height,
                                                                   width=self.width,
                                                                   image_aug=self.image_aug,
                                                                   aug_prob=self.aug_prob,
                                                                   image_ext=self.image_ext,
                                                                   data_format=self.data_format))

                    # Try to get a data sample from dataloader
                    sample_train_output = self.train_dataset.__getitem__(0)

                except(ConditionError, FileNotFoundError):
                    self.train_dataset = ConcatDataset(SutureDataset(data_root_folder=self.dataroot_A,
                                                                     filenames=train_filenames_A,
                                                                     height=self.height,
                                                                     width=self.width,
                                                                     image_aug=self.image_aug,
                                                                     aug_prob=self.aug_prob,
                                                                     image_ext=self.image_ext,
                                                                     data_format=self.data_format,
                                                                     image_mask_aug=self.image_mask_aug,
                                                                     binary=self.binary,
                                                                     blur_func=self.blur_func,
                                                                     spread=self.spread),

                                                       SutureDataset(data_root_folder=self.dataroot_B,
                                                                     filenames=train_filenames_B,
                                                                     height=self.height,
                                                                     width=self.width,
                                                                     image_aug=self.image_aug,
                                                                     aug_prob=self.aug_prob,
                                                                     image_ext=self.image_ext,
                                                                     data_format=self.data_format,
                                                                     image_mask_aug=self.image_mask_aug,
                                                                     binary=self.binary,
                                                                     blur_func=self.blur_func,
                                                                     spread=self.spread))

                    # Try-except block for the validation data generator
                    if os.path.isfile(self.split_file_path.format("val")):
                        val_filenames_A = utils.read_lines_from_text_file(self.split_file_path_A.format("val"))
                        val_filenames_B = utils.read_lines_from_text_file(self.split_file_path_B.format("val"))

                        try:
                            # If the condition (load from disk) is not satisfied or if there is an exception,
                            # then on-the-fly dataloader is used
                            if not self.load_from_disk: raise ConditionError

                            self.val_dataset = ConcatDataset(EndoMaskDataset(data_root_folder=self.dataroot_A,
                                                                             filenames=val_filenames,
                                                                             height=self.height,
                                                                             width=self.width,
                                                                             image_aug=None,
                                                                             image_ext=self.image_ext,
                                                                             data_format=self.data_format,
                                                                             image_mask_aug=None,
                                                                             mask_path_suffix=self.mask_path_suffix),

                                                             EndoMaskDataset(data_root_folder=self.dataroot,
                                                                             filenames=val_filenames,
                                                                             height=self.height,
                                                                             width=self.width,
                                                                             image_aug=None,
                                                                             image_ext=self.image_ext,
                                                                             data_format=self.data_format,
                                                                             image_mask_aug=None,
                                                                             mask_path_suffix=self.mask_path_suffix))

                            # Try to get a data sample from dataloader
                            sample_val_output = self.val_dataset.__getitem__(0)

                        except(ConditionError, FileNotFoundError):
                            self.val_dataset = ConcatDataset(SutureDataset(data_root_folder=self.dataroot,
                                                                           filenames=val_filenames,
                                                             height=self.height,
                                                             width=self.width,
                                                             image_aug=None,
                                                             image_ext=self.image_ext,
                                                             data_format=self.data_format,
                                                             image_mask_aug=None,
                                                             binary=self.binary,
                                                             blur_func=self.blur_func,
                                                             spread=self.spread),

                                                             SutureDataset(data_root_folder=self.dataroot,
                                                                           filenames=val_filenames,
                                                                           height=self.height,
                                                                           width=self.width,
                                                                           image_aug=None,
                                                                           image_ext=self.image_ext,
                                                                           data_format=self.data_format,
                                                                           image_mask_aug=None,
                                                                           binary=self.binary,
                                                                           blur_func=self.blur_func,
                                                                           spread=self.spread))

                if stage in (None, "test"):
                    test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                    self.test_dataset = SutureDataset(data_root_folder=self.dataroot,
                                                      filenames=test_filenames,
                                                      height=self.height,
                                                      width=self.width,
                                                      image_aug=None,
                                                      image_ext=self.image_ext,
                                                      data_format=self.data_format,
                                                      image_mask_aug=None,
                                                      binary=self.binary,
                                                      blur_func=self.blur_func,
                                                      spread=self.spread,
                                                      test=True)

                if stage in (None, "predict"):
                    test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                    self.test_dataset = EndoDataset(data_root_folder=self.dataroot,
                                                    filenames=test_filenames,
                                                    height=self.height,
                                                    width=self.width,
                                                    image_aug=None,
                                                    image_ext=self.image_ext,
                                                    data_format=self.data_format)


class StereoDataModule(EndoDataModule):
    def setup(self, stage=None):
        if stage in (None, "fit"):
            # Get filenames from split-path
            train_filenames = utils.read_lines_from_text_file(self.split_file_path.format("train"))
            self.train_dataset = StereoImageDataset(data_root_folder=self.dataroot,
                                                    filenames=train_filenames,
                                                    height=self.height,
                                                    width=self.width,
                                                    mage_aug=self.image_aug,
                                                    aug_prob=self.aug_prob,
                                                    image_ext=self.image_ext,
                                                    data_format=self.data_format)

            # Try-except block for the validation data generator
            if os.path.isfile(self.split_file_path.format("val")):
                val_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
                self.val_dataset = StereoImageDataset(data_root_folder=self.dataroot,
                                                      filenames=val_filenames,
                                                      height=self.height,
                                                      width=self.width,
                                                      image_aug=None,
                                                      image_ext=self.image_ext,
                                                      data_format=self.data_format)

        if stage in (None, "test"):
            test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
            self.test_dataset = StereoImageDataset(data_root_folder=self.dataroot,
                                                   filenames=test_filenames,
                                                   height=self.height,
                                                   width=self.width,
                                                   image_aug=None,
                                                   image_ext=self.image_ext,
                                                   data_format=self.data_format,
                                                   test=True)

        if stage in (None, "predict"):
            test_filenames = utils.read_lines_from_text_file(self.split_file_path.format("val"))
            self.test_dataset = EndoDataset(data_root_folder=self.dataroot,
                                            filenames=test_filenames,
                                            height=self.height,
                                            width=self.width,
                                            image_aug=None,
                                            image_ext=self.image_ext,
                                            data_format=self.data_format)
