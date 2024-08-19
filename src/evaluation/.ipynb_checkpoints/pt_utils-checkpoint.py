# Imports
import os
import sys
import json
import cv2
import random
import numpy as np
from copy import deepcopy

from natsort import natsorted
from PIL import Image

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(os.path.dirname(current_dir))  # ~/endoscopic-image-analysis
sys.path.append(project_root)

from src.utils import utils as ut


def interpolate_point(h_original=None,
                      w_original=None,
                      h_resized=None,
                      w_resized=None,
                      x=0, y=0):
    """
    Interpolate a point from original size to target size
    :param h_original: Original height of image in which points were labeled
    :param w_original: Original width of mask in which points were labeled
    :param h_resized: Target height to be resized to
    :param w_resized: Target width to be resized to
    :param x: x coordinate of point to be resized
    :param y: y coordinate of point to be resized
    :return: Interpolated x,y
    """
    if not (h_resized or w_resized or h_original or w_original):
        raise ValueError("Please specify valid values for interpolation")
    if (h_original == h_resized) and (w_original == w_resized): return x,y

    factor_h = h_resized / h_original
    factor_w = w_resized / w_original
    interpolated_x = round(x * factor_w)
    interpolated_y = round(y * factor_h)

    # Handling rounding errors
    if interpolated_x > w_resized - 1: interpolated_x = w_resized - 1
    if interpolated_y > h_resized - 1: interpolated_y = h_resized - 1
    return interpolated_x, interpolated_y


def extract_points_id_from_json_kitti(json_labels, height=288, width=512):
    coordinates = {"x": 0, "y": 1}
    corresponding_only = False

    points_dict = json_labels['shapes']
    left_points_list, right_points_list = [], []

    h_original = json_labels["imageHeight"] // 2
    w_original = json_labels["imageWidth"]

    left_points_dict = {}
    right_points_dict = {}

    for i in range(len(points_dict)):
        if points_dict[0]['label'] == 'None':
            pass
        else:
            if points_dict[i]['shape_type'] == 'None':
                pass
            elif points_dict[i]['shape_type'] == 'line':
                # Case 1: The first labelled point file_items["shapes"][n]['points'][0] --> [x,y]
                # The y coordinate lies in the lower half of the image
                # which means y coordinate>im_h/2; then first point belongs to the bottom image
                # else first point belongs to top image
                # They are then matched to left and right depending on how it was recorded
                if points_dict[i]['points'][0][1] > json_labels["imageHeight"] // 2:
                    side = {"bottom": 0, "top": 1}
                else:
                    side = {"top": 0, "bottom": 1}

                top_x = points_dict[i]['points'][side["top"]][coordinates["x"]]
                top_y = points_dict[i]['points'][side["top"]][coordinates["y"]]
                bottom_x = points_dict[i]['points'][side["bottom"]][coordinates["x"]]
                bottom_y = points_dict[i]['points'][side["bottom"]][coordinates["y"]] \
                           - json_labels["imageHeight"] // 2

                # Assign top and bottom to left and right
                point_pair = {"top": (top_x, top_y),
                              "bottom": (bottom_x, bottom_y)}

                left_point = point_pair[json_labels['lr_to_top_down']["left"]]
                right_point = point_pair[json_labels['lr_to_top_down']["right"]]

                left_point = interpolate_point(h_original=h_original,
                                               w_original=w_original,
                                               h_resized=height,
                                               w_resized=width,
                                               x=left_point[coordinates["x"]],
                                               y=left_point[coordinates["y"]])

                right_point = interpolate_point(h_original=h_original,
                                                w_original=w_original,
                                                h_resized=height,
                                                w_resized=width,
                                                x=right_point[coordinates["x"]],
                                                y=right_point[coordinates["y"]])

                left_points_dict[points_dict[i]['label']] = left_point
                right_points_dict[points_dict[i]['label']] = right_point

            elif points_dict[i]['shape_type'] == 'point':
                # If correspondences_only flag is set to False, these single points are also included
                x = points_dict[i]['points'][0][0]
                y = points_dict[i]['points'][0][1]

                point = (x, y)
                point = interpolate_point(h_original=h_original,
                                          w_original=w_original,
                                          h_resized=height,
                                          w_resized=width,
                                          x=point[coordinates["x"]],
                                          y=point[coordinates["y"]])

                # Check if the y-coordinate belongs to top or bottom image
                # And then match it to left or right; append accordingly
                if points_dict[i]['points'][0][1] < json_labels["imageHeight"] // 2:
                    point = (x, y)
                    point = interpolate_point(h_original=h_original,
                                              w_original=w_original,
                                              h_resized=height,
                                              w_resized=width,
                                              x=point[coordinates["x"]],
                                              y=point[coordinates["y"]])

                    if json_labels["lr_to_top_down"]["left"] == "top":
                        left_points_dict[points_dict[i]['label']] = point
                    else:
                        right_points_dict[points_dict[i]['label']] = point

                else:
                    point = (x, y - json_labels["imageHeight"] // 2)
                    point = interpolate_point(h_original=h_original,
                                              w_original=w_original,
                                              h_resized=height,
                                              w_resized=width,
                                              x=point[coordinates["x"]],
                                              y=point[coordinates["y"]])

                    if json_labels["lr_to_top_down"]["left"] == "bottom":
                        left_points_dict[points_dict[i]['label']] = point
                    else:
                        right_points_dict[points_dict[i]['label']] = point

    return [left_points_dict, right_points_dict]


def get_union_of_n_lists(list_of_lists):
    list_of_sets = [set(list_elem) for list_elem in list_of_lists]
    return list(set.union(*list_of_sets))


def interpolate_list(list_of_points):
    copied_list = deepcopy(list_of_points)
    for i, point in enumerate(copied_list):
        if i > 0:  # Start checking from the first element
            if not point:
                try:
                    prev_value = ut.get_last_non_zero_elem(copied_list[:i])  # get last non-zero element
                    # to get next element, reverse the list and get last non-zero element
                    next_value = ut.get_last_non_zero_elem(list(reversed(copied_list[i:])))
                    # interpolate between the two values
                    if prev_value and next_value:
                        copied_list[i] = (
                        int((prev_value[0] + next_value[0]) / 2), int((prev_value[1] + next_value[1]) / 2))
                except:
                    print('Excepting at :', i, 'and exiting')
                    break
    return copied_list


def get_outliers_arr(arr):
    q1 = np.quantile(arr, 0.25)  # finding the 1st quartile
    q3 = np.quantile(arr, 0.75)  # finding the 3rd quartile
    med = np.median(arr)
    iqr = q3 - q1  # finding the iqr region
    upper_bound = int(q3 + (1.5 * iqr))  # finding upper and lower whiskers
    lower_bound = int(q1 - (1.5 * iqr))
    return [i for i, k in enumerate(arr) if (k <= lower_bound) | (k >= upper_bound)]


def get_outlier_indices(list_of_points):
    x = [point[0] for point in list_of_points if point]
    y = [point[1] for point in list_of_points if point]
    x_outliers = get_outliers_arr(x)
    y_outliers = get_outliers_arr(y)
    x_y_outliers = np.unique(x_outliers + y_outliers)
    # Find the indices of all the outliers in the non-squeezed array
    indices = [i for i, k in enumerate(list_of_points) for indx in x_y_outliers if
               k == (x[indx], y[indx])]
    return indices
