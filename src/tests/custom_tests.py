# Set up imports
import os
import sys
import warnings

import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(os.path.dirname(current_dir))  # ~/endoscopic-image-analysis
sys.path.append(project_root)

import src.utils.utils as ut


class DataLeakError(Exception):
    pass


def test_data_leak(folds_path=None, num_splits=None):
    folds_filepath = os.path.join(folds_path, "fold_{}", "{}_files.txt")
    for fold in range(num_splits):
        train_paths, val_paths = [ut.read_lines_from_text_file(folds_filepath.format(fold+1, split))
                                  for split in ["train", "val"]]

        train_rel_paths, val_rel_paths = [[path.split()[0] for path in paths] for paths in [train_paths, val_paths]]
        train_surgery_folders, val_surgery_folders = [[rel_path.split('/')[0] for rel_path in rel_paths] for rel_paths
                                                      in [train_rel_paths, val_rel_paths]]
        train_surgeries, val_surgeries = [set(folders) for folders in [train_surgery_folders, val_surgery_folders]]

        if train_surgeries in val_surgeries:
            raise DataLeakError("Train data found in validation data split!")
        else: print("Test passed for Fold {}: No data leak between training and evaluation folds".format(fold+1))