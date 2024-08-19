# Imports
import os
import sys

import cv2
import random
import numpy as np
from PIL import Image

from torchvision.transforms import functional as func
from torchvision import transforms
from torch.utils.data import Dataset

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils import utils as ut
from src.data.gen_suture_masks import GenSutureMasks


class EndoDataset(Dataset):
    """ The endoscopic dataset requires the following folder structure:
    Surgery --> Video --> images which contain the images to be loaded

    This mono class, works with split files which are text files that contain the
    relative path and the image name. (It is a mono class because it returns a single image)
    The class reads the image file paths that are specified in the text file and loads the images.
    It applies the specified transforms to the image, else it just converts it into a tensor.
    :returns Pre-processed and augmented image as a tensor
    """

    def __init__(self, data_root_folder=None,
                 filenames=None,
                 height=448,
                 width=448,
                 image_aug=None,
                 aug_prob=0.5,
                 camera="left",
                 image_ext='.png',
                 data_format="mono"):
        super(EndoDataset).__init__()
        self.data_root_folder = data_root_folder
        self.filenames = filenames
        self.height = height
        self.width = width
        self.image_ext = image_ext
        self.camera = camera
        self.format = data_format

        # Pick image loader based on image format
        self.image_loader = np.load if self.image_ext == '.npy' else ut.pil_loader
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.cam_to_side = {"left": "l", "right": "r"}

        # Image pre-processing options
        self.image_aug = image_aug
        self.aug_prob = aug_prob

        # Pick resize function based on image format
        if self.image_ext == '.png':
            # Output: Resized PIL Image
            self.resize = transforms.Resize((self.height, self.width), interpolation=Image.LINEAR)
            # Resize to dims slightly larger than given dims
            # Sometimes useful for aug together with crop function
            self.resize_bigger = transforms.Resize((int(self.height * 1.2),
                                                    int(self.width * 1.2)))
        elif self.image_ext == '.npy':
            self.resize = lambda x: cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            self.resize_bigger = lambda x: cv2.resize(x, (self.width*1.2, self.height*1.2),
                                                      interpolation=cv2.INTER_NEAREST)

    def get_split_filename(self, filename):
        """ Splits a filename string comprising of "relative_path <space> image_name"
        :param filename A string comprising of "relative_path <space> image_name"
        :return split_filename- "relative_path, image_name"
        """
        split_filename = filename.split()
        if self.format == "mono": return split_filename  # folder, image_name
        elif self.format == "kitti":
            # folder, frame_num, side
            if len(split_filename) == 2: return split_filename[0], split_filename[1], self.cam_to_side[self.camera]
            else: return split_filename

    def make_image_path(self, data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of image
        """
        if self.format == "mono": return os.path.join(data_root, rel_folder, "images", image_name+self.image_ext)
        elif self.format == "kitti":
            frame_name = "{:06d}{}".format(int(image_name), self.image_ext)
            return os.path.join(data_root, rel_folder,
                                "image_0{}".format(self.side_map[side]), frame_name)

    def preprocess(self, image):
        image = self.resize(image)
        # if self.image_aug and random.random() > self.aug_prob: image = self.image_aug(image)
        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]  # alb needs np input
        return func.to_tensor(image)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns the image with pre-proc transforms + aug applied to it"""
        args = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        image = self.preprocess(image)
        return image, self.filenames[index]


class EndoMaskDataset(EndoDataset):
    """ Loads an image and its corresponding mask
    Some aug is performed common to both image and mask
    Some aug like color aug is performed only on the image
    :returns Pre-proc+aug image, mask
    """

    def __init__(self, mask_transform=None,
                 image_mask_aug=None,
                 mask_path_suffix="",
                 **kwargs):
        super(EndoMaskDataset, self).__init__(**kwargs)
        self.mask_path_suffix = mask_path_suffix
        self.image_mask_aug = image_mask_aug
        self.mask_loader = ut.mask_np_loader if self.image_ext == '.npy' else ut.mask_loader
        self.mask_transform = transforms.ToTensor() if mask_transform is None else transforms.Compose(mask_transform)

    def make_mask_path(self, data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of mask
        """
        if self.format == "mono": return os.path.join(data_root, rel_folder, "masks"+self.mask_path_suffix,
                                                      image_name+self.image_ext)
        elif self.format == "kitti":
            frame_name = "{:06d}{}".format(int(image_name), self.image_ext)
            return os.path.join(data_root, rel_folder,
                                "mask_0{}".format(self.side_map[side])+self.mask_path_suffix, frame_name)

    def preprocess_image_mask(self, image, mask):
        image = self.resize(image)
        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]
        if self.image_mask_aug:
            augmented = self.image_mask_aug(image=np.asarray(image), mask=np.asarray(mask))
            # alb needs np input
            image = augmented["image"]
            mask = augmented["mask"]
        image = func.to_tensor(np.array(image))
        mask = func.to_tensor(np.array(mask))
        return image, mask

    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        mask = self.mask_loader(self.make_mask_path(self.data_root_folder, *args))
        image, mask = self.preprocess_image_mask(image=image, mask=mask)
        return image, mask, self.filenames[index]


class SutureDataset(EndoMaskDataset):
    """ Note: this may not be efficient, need to load everything offline to be efficient
    This is only for rapid experimentation
        :returns Pre-proc+aug image, mask
        """

    def __init__(self, binary="False",
                 blur_func="gaussian",
                 spread=1,
                 corresponding_only=False,
                 test=False,
                 **kwargs):
        super(SutureDataset, self).__init__(**kwargs)

        self.binary = binary
        self.blur_func = blur_func
        self.spread = spread
        self.corresponding_only = corresponding_only
        self.json_loader = ut.json_loader
        self.side_to_mask_list_index = {"l": 0, "r": 1}
        self.test = test
        self.gen_suture_masks = GenSutureMasks(target_height=self.height,
                                               target_width=self.width,
                                               binary=self.binary,
                                               blur_func=self.blur_func,
                                               data_format=self.format,
                                               spread=self.spread,
                                               corresponding_only=self.corresponding_only)

    @staticmethod
    def make_json_path(data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of mask
        """
        return os.path.join(data_root, rel_folder, "point_labels", image_name+".json")

    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        labels = self.json_loader(self.make_json_path(self.data_root_folder, *args))

        if self.test:
            l_r_points = self.gen_suture_masks.get_points(labels)
            side = args[-1] if len(args) == 3 else 'l'  # If there is a side, otherwise take left
            points = l_r_points[self.side_to_mask_list_index[side]]
            image = self.preprocess(image=image)
            return image, points, self.filenames[index]

        else:
            mask = self.gen_suture_masks.labels_to_mask(labels)
            # If mono then list contains just one mask, if kitti then [left_mask, right_mask]
            # This part is not really efficient because for each frame both left and right masks are "created"
            # But this should work for now, for experimentation purposes
            # For efficiency, the masks need to be loaded offline anyway
            mask = mask[0] if self.format=="mono" else mask[self.side_to_mask_list_index[args[-1]]]
            image, mask = self.preprocess_image_mask(image=image, mask=mask)
            return image, mask, self.filenames[index]


class StereoEndoDataset(EndoDataset):
    def preprocess(self, image_A, image_B):
        image_A = self.resize(image_A)
        image_B = self.resize(image_B)
        # if self.image_aug and random.random() > self.aug_prob: image = self.image_aug(image)
        if self.image_aug:
            augmented_images = self.image_aug(image=np.asarray(image_A), other_image=np.asarray(image_B))
            image_A = augmented_images["image"]
            image_B = augmented_images["other_image"]
        image = np.concatenate(
            (np.array(image_A), np.array(image_B)), axis=2)
        return func.to_tensor(image)

    def __getitem__(self, index):
        """Returns the image with pre-proc transforms + aug applied to it"""
        args = self.get_split_filename(self.filenames[index])
        image_A = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        image_B = self.image_loader(self.make_image_path(self.data_root_folder, *[self.other_cam[arg]
                                                                                  if idx == 2 else arg for idx, arg in enumerate(args)]))
        image = self.preprocess(image_A, image_B)
        return image, self.filenames[index]


class StereoEndoMaskDataset(EndoMaskDataset, StereoEndoDataset):
    def preprocess_image_mask(self, image_A, image_B, mask_A, mask_B):
        image_A = self.resize(image_A)
        image_B = self.resize(image_B)
        if self.image_aug:
            augmented_images = self.image_aug(image=np.asarray(image_A), other_image=np.asarray(image_B))
            image_A = augmented_images["image"]
            image_B = augmented_images["other_image"]
        if self.image_mask_aug:
            augmented_images_and_mask = self.image_mask_aug(image=np.asarray(image_A),
                                                            mask=np.asarray(mask_A), other_image=np.asarray(image_B))
            image_A = augmented_images_and_mask["image"]
            mask_A = augmented_images_and_mask["mask"]
            image_B = augmented_images_and_mask["other_image"]
        mask_A = np.array(mask_A)[..., np.newaxis]
        image = func.to_tensor(np.concatenate((np.array(image_A), np.array(image_B)), axis=2))
        mask_A = func.to_tensor(np.array(mask_A))
        return image, mask_A

    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image_A = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        mask_A = self.mask_loader(self.make_mask_path(self.data_root_folder, *args))
        image_B = self.image_loader(self.make_image_path(
            self.data_root_folder, *[self.other_cam[arg] if idx == 2 else arg for idx, arg in enumerate(args)]))
        mask_B = self.mask_loader(self.make_mask_path(
            self.data_root_folder, *[self.other_cam[arg] if idx == 2 else arg for idx, arg in enumerate(args)]))
        image, mask = self.preprocess_image_mask(image_A, image_B, mask_A, mask_B)
        return image, mask, self.filenames[index]


class StereoSutureDataset(SutureDataset, StereoEndoMaskDataset):
    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image_A = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        image_B = self.image_loader(self.make_image_path(
            self.data_root_folder, *[self.other_cam[arg] if idx == 2 else arg for idx, arg in enumerate(args)]))
        labels = self.json_loader(self.make_json_path(self.data_root_folder, *args))

        if self.test:
            l_r_points = self.gen_suture_masks.get_points(labels)
            points = l_r_points[self.side_to_mask_list_index[args[2]]]
            image = self.preprocess(image_A, image_B)
            return image, points, self.filenames[index]

        else:
            mask = self.gen_suture_masks.labels_to_mask(labels)
            # If mono then list contains just one mask, if kitti then [left_mask, right_mask]
            # This part is not really efficient because for each frame both left and right masks are "created"
            # But this should work for now, for experimentation purposes
            # For efficiency, the masks need to be loaded offline anyway
            image, mask = self.preprocess_image_mask(image_A, image_B, mask[self.side_to_mask_list_index[args[2]]],
                                                     mask[self.side_to_mask_list_index[self.other_cam[args[2]]]])
            return image, mask, self.filenames[index]


class EndoMaskDatasetAB(EndoMaskDataset):
    """ Load unpaired images from two domains
    Together with the mask
    Same image aug is applied to images from both domains
    Similarly, same mask aug applied to masks from both domains
    Note: The default behaviour is aligned=True! This is because aligned=False generates
    random numbers by using randint, which leads to many repeating values
    """

    def __init__(self,
                 data_root_folder_A=None,
                 data_root_folder_B=None,
                 filenames_A=None,
                 filenames_B=None,
                 randomise=False,
                 **kwargs):
        super(EndoMaskDatasetAB, self).__init__(**kwargs)

        self.randomise = randomise
        self.data_root_folder_A = data_root_folder_A
        self.data_root_folder_B = data_root_folder_B
        self.filenames_A = filenames_A
        self.filenames_B = filenames_B
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __getitem__(self, index):
        # Get the relative path, frame number and side : need for both mask and image
        args_A = self.get_split_filename(self.filenames_A[index % len(self.filenames_A)])
        # mod only prevents index overshoot that may be caused due to differing lengths returned by max
        if self.randomise:
            # Note: randomise may lead to repeating samples in B domain. It is recommended to use randomise=False
            args_B = self.get_split_filename(self.filenames_B[random.randint(0, len(self.filenames_B) - 1)])
        else: args_B = self.get_split_filename(self.filenames_B[index % len(self.filenames_B)])

        image_A, image_B = map(self.image_loader, (self.make_image_path(self.data_root_folder_A, *args_A),
                                                   self.make_image_path(self.data_root_folder_B, *args_B)))

        mask_A, mask_B = map(self.mask_loader, (self.make_mask_path(self.data_root_folder_A, *args_A),
                                                self.make_mask_path(self.data_root_folder_B, *args_B)))

        [(image_A, mask_A), (image_B, mask_B)] = [self.preprocess_image_mask(image=image, mask=mask)
                                                  for (image, mask) in [(image_A, mask_A), (image_B, mask_B)]]
        return image_A, mask_A, image_B, mask_B

    def __len__(self):
        return max(len(self.filenames_A), len(self.filenames_B))


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class StereoImageDataset(EndoDataset):
    """Loads an image and its corresponding stereo view
    :returns image, corresponding stereo-image
    """

    def __init__(self, **kwargs):
        super(StereoImageDataset, self).__init__(**kwargs)
        self.cam_to_opp_side = {"left": "r", "right": "l", "l":"r", "r":"l"}
        self.crop = transforms.CenterCrop((self.height, self.width))  # (h,w)

    def preprocess_image_mask(self, image, opp_image):
        #image = self.resize(image)
        #opp_image = self.resize(opp_image)

        image = self.crop(image)
        opp_image = self.crop(opp_image)
        return func.to_tensor(image), func.to_tensor(opp_image)

    def make_image_path(self, data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of image
        """
        if self.format == "mono": return os.path.join(data_root, rel_folder, "images", image_name+self.image_ext)
        elif self.format == "kitti":
            frame_name = (image_name+self.image_ext)
            return os.path.join(data_root, rel_folder,
                                "image_0{}".format(self.side_map[side]), frame_name)

    def __getitem__(self, index):
        dir, frame_num = (self.filenames[index]).split()[:2]#, (self.filenames[index]).split()[1]
        im_view = self.image_loader(self.make_image_path(self.data_root_folder, dir, frame_num,
                                                         self.cam_to_side[self.camera]))
        opp_view = self.image_loader(self.make_image_path(self.data_root_folder, dir, frame_num,
                                                          self.cam_to_opp_side[self.camera]))

        im_view, opp_view = self.preprocess_image_mask(im_view, opp_view)
        return im_view, opp_view, self.filenames[index]
