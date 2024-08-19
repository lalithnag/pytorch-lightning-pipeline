# Set up imports
import os
import sys
import cv2
import warnings
import random
import glob
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(os.path.dirname(current_dir))  # ~/endoscopic-image-analysis
sys.path.append(project_root)

from src.utils import utils as ut
from src.tests.custom_tests import test_data_leak


class GenSplits:
    """
    Class to get a data folder with specific surgeries or sessions (called "ops")
    and split them for cross-validation using sklearn
    Sometimes the root directory may have other unnecessary folders, in this case the op_dirs can
    be provided as an argument to this class as a list
    """
    def __init__(self, data_root=None,
                 splits_root="../splits",
                 splits_name=None,
                 num_folds = 3,
                 op_dirs=None,
                 append_lr=True,
                 save_vis=True,
                 cv=GroupKFold,
                 cmap_data=plt.cm.Paired,
                 cmap_cv=plt.cm.coolwarm):
        super(GenSplits).__init__()
        self.data_root = data_root
        self.splits_root = splits_root
        self.append_lr = append_lr
        self.splits_name = splits_name
        self.split_path = os.path.join(self.splits_root, self.splits_name)

        self.num_folds = num_folds
        self.cv = cv(n_splits=self.num_folds)
        self.cv.get_n_splits()
        self.op_dirs = op_dirs if op_dirs else ut.get_sub_dirs(self.data_root, paths=False)
        self.image_paths, self.surgery_name_groups, self.surgery_group_number = self.prepare_data_paths()

        self.save_vis = save_vis
        self.cmap_data = cmap_data
        self.cmap_cv = cmap_cv
        self.save_path = os.path.join(self.split_path, "{}.png")

    def get_images_in_surgery_folder(self, surgery_folder):
        """
        In a specific surgery dir like '2019_xx_xx_aa1111', it returns all the image_paths as a list
        :param surgery_folder:
        :return: indices, surgery_names
        The surgery names are '2019_xx_xxxx' will be used to group the data so that all the images belonging
        to one op remain in the same split created by sklearn
        """
        indices, surgery_names = [], []
        video_folders = ut.get_sub_dirs(os.path.join(self.data_root, surgery_folder), paths=False)
        for video in video_folders:
            image_filepaths = natsorted(glob.glob(os.path.join(self.data_root, surgery_folder, video,
                                                               'image_02', '*')))
            # Prepare text file information
            rel_path_name = os.path.join(surgery_folder, video)  # Folder name
            frame_indices = [os.path.basename(os.path.splitext(path)[0]) for path in image_filepaths]
            newline_batch = [' '.join((rel_path_name, frame_index)) for frame_index in frame_indices]
            indices += newline_batch
            # Just append surgery name along as the 'group' variable
            surgery_names += [surgery_folder] * len(image_filepaths)
        return indices, surgery_names

    @staticmethod
    def append_lr_to_filenames(indices):
        appended_indices = []
        for line in indices:
            for side in ["l", "r"]:
                index = ' '.join((line, side))
                appended_indices.append(index)
        return appended_indices

    def prepare_data_paths(self):
        """
        Gets all the imapge paths, surgery names from the dirs specified by op_dirs
        Additionally, assigns a group number for each surgery name as a list, so that this
        can be used for visualisation of groups
        :return:
        """
        image_paths, surgery_name_groups, surgery_num = [], [], []
        for i, surgery in enumerate(self.op_dirs):
                images, surgery_names = self.get_images_in_surgery_folder(surgery_folder=surgery)
                image_paths += images
                surgery_name_groups += surgery_names
                surgery_num += [i]*len(surgery_names)
        return image_paths, surgery_name_groups, surgery_num

    def visualize_data_groups(self, groups):
        """
        Visualise the splits of groups
        :param groups: The list of group numbers for each data sample
        :return: If save flag is set then it saves the created figure
        When this function is called from a notebook, the fig is displayed inline
        """
        fig, ax = plt.subplots()
        ax.scatter(range(len(groups)), [.5] * len(groups), c=groups, marker='_', lw=50, cmap=self.cmap_data)
        # ::TODO:: Annotate this graph with the fold number and number of images
        ax.set(yticklabels=['Data\ngroup'], xlabel="Sample index")
        ax.set_title('Data split of different ops', fontsize=15)
        if self.save_vis: plt.savefig(self.save_path.format(self.splits_name+"_data_split"))

    def plot_cv_indices(self, cv, X, group, ax, n_splits, lw=10):
        """Create a sample plot for indices of a cross-validation object."""
        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=X, groups=group)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices, marker='_', lw=lw, cmap=self.cmap_cv,
                       vmin=-.2, vmax=1.2)
        ax.scatter(range(len(group)), [ii + 1.5] * len(group),
                   c=group, marker='_', lw=lw, cmap=self.cmap_data)

        # Formatting
        yticklabels = list(range(n_splits)) + ['group']
        ax.set(yticks=np.arange(n_splits + 0.1) + .5, yticklabels=yticklabels,
               xlabel='Sample index', ylabel="CV iteration",
               ylim=[n_splits + 1.2, -.2], xlim=[0, len(group)])
        ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
        return ax

    def split_save_data(self):
        """
        The master function that creates and saves vis if flag is set, else when called from a notebook these
        vis are displayed inline. Then this function splits the data according to the params provided, appends lr if
        the flag is set (this helps in using images from both the cameras), shuffles them, and finally writes them to
        disk by creating a folder for each fold. Each of these folders will contain a train and val split.
        """
        print("\nStarting cross-validation splitting...")
        print("\nThe surgeries used for training in this dataset are: ")
        ut.print_elements_of_list(self.op_dirs)

        ut.check_and_create_folder(self.split_path)  # splits/mkr_mono/fold_1

        # Visualise split of data groups
        if self.save_vis:
            self.visualize_data_groups(groups=self.surgery_group_number)  # Data visualisation
            # CV visualisation
            fig, ax = plt.subplots()
            self.plot_cv_indices(self.cv, X=self.image_paths, group=self.surgery_group_number,
                                 ax=ax, n_splits=self.num_folds)
            plt.savefig(self.save_path.format(self.splits_name+"_fold_split"))

        # Splits the data and save to disk
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.image_paths, self.image_paths,
                                                                   self.surgery_name_groups)):
            success = ut.check_and_create_folder(
                os.path.join(self.split_path, "fold_"+str(fold+1)))  # splits/mkr_mono/fold_1

            train_paths = [self.image_paths[idx] for idx in train_idx]
            val_paths = [self.image_paths[idx] for idx in test_idx]

            if self.append_lr:
                [train_paths, val_paths] = map(self.append_lr_to_filenames, [train_paths, val_paths])

            random.shuffle(train_paths)
            random.shuffle(val_paths)

            f_writepath = os.path.join(self.split_path, "fold_{}", "{}_files.txt")
            ut.write_list_to_text_file(save_path=f_writepath.format(fold + 1, "train"),
                                       text_list=train_paths,
                                       verbose=False)

            ut.write_list_to_text_file(save_path=f_writepath.format(fold + 1, "val"),
                                       text_list=val_paths,
                                       verbose=False)

            print("Fold {}: Extracted {} training files and {} validation files "
                  "and wrote them to disk at {}".format(str(fold + 1), len(train_paths), len(val_paths),
                                                        self.split_path))


parser = argparse.ArgumentParser('Generate splits of data for cross-validation')
parser.add_argument("--dataroot", type=str, help="path where the datasets are located")
parser.add_argument("--splits_root", type=str, help="split directory", default="../splits")
parser.add_argument("--splits_name", type=str, help="name of split")

parser.add_argument('--op_dirs', type=str)
parser.add_argument('--num_folds', type=int, default=3, help='number of folds to be split into')
parser.add_argument("--append_lr", help="Take images from both cameras?", action="store_true")
parser.add_argument("--save_vis", help="Save vis of data and fold split?", action="store_true")


if __name__ == '__main__':

    args = parser.parse_args()
    args.op_dirs = list(args.op_dirs.split(',')) if args.op_dirs else None
    args.splits_name = args.splits_name+"_lr" if args.append_lr else args.splits_name+"_mono"
    gen_splits = GenSplits(data_root=args.dataroot,
                           splits_root=args.splits_root,
                           splits_name=args.splits_name,
                           num_folds=args.num_folds,
                           op_dirs=args.op_dirs,
                           append_lr=args.append_lr,
                           save_vis=args.save_vis)

    gen_splits.split_save_data()

    print("Running tests to ensure no data leak...")
    test_data_leak(folds_path=os.path.join(args.splits_root, args.splits_name),
                   num_splits=args.num_folds)
