# Imports
import os
import sys
import json
import cv2
import random
import numpy as np
from copy import deepcopy

from scipy.optimize import linear_sum_assignment
import math

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


def extract_points_id_from_json_kitti_as_lists(json_labels, height=288, width=512):
    coordinates = {"x": 0, "y": 1}
    corresponding_only = False

    points_dict = json_labels['shapes']
    left_points_list, right_points_list = [], []

    h_original = json_labels["imageHeight"] // 2
    w_original = json_labels["imageWidth"]

    left_points_label_list= []
    left_points_list = []

    right_points_label_list = []
    right_points_list = []

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

                left_points_label_list.append(points_dict[i]['label'])
                left_points_list.append(left_point)

                right_points_label_list.append(points_dict[i]['label'])
                right_points_list.append(right_point)
                #left_points_dict[points_dict[i]['label']] = left_point
                #right_points_dict[points_dict[i]['label']] = right_point

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
                        #left_points_dict[points_dict[i]['label']] = point
                        left_points_label_list.append(points_dict[i]['label'])
                        left_points_list.append(point)
                    else:
                        right_points_label_list.append(points_dict[i]['label'])
                        right_points_list.append(point)
                        #right_points_dict[points_dict[i]['label']] = point

                else:
                    point = (x, y - json_labels["imageHeight"] // 2)
                    point = interpolate_point(h_original=h_original,
                                              w_original=w_original,
                                              h_resized=height,
                                              w_resized=width,
                                              x=point[coordinates["x"]],
                                              y=point[coordinates["y"]])

                    if json_labels["lr_to_top_down"]["left"] == "bottom":
                        left_points_label_list.append(points_dict[i]['label'])
                        left_points_list.append(point)
                        #left_points_dict[points_dict[i]['label']] = point
                    else:
                        right_points_label_list.append(points_dict[i]['label'])
                        right_points_list.append(point)
                        #right_points_dict[points_dict[i]['label']] = point

    #return [left_points_dict, right_points_dict]
    return [left_points_label_list, right_points_label_list, left_points_list, right_points_list]


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
                        copied_list[i] = (int((prev_value[0] + next_value[0]) / 2),
                                          int((prev_value[1] + next_value[1]) / 2))
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


def get_outlier_mask(id_points_dict_for_all_ids, point_list):
    selected_id_points_dict = [k for k in id_points_dict_for_all_ids if k.keys()[0] in point_list]
    print(selected_id_points_dict)


def get_global_matched_id_dict(trackers_source_frame, trackers_new_frame, radius=20):
    trackers_source_frame = dict(sorted(trackers_source_frame.items()))
    trackers_new_frame = dict(sorted(trackers_new_frame.items()))
    if not trackers_source_frame: return trackers_new_frame, trackers_source_frame
    # If the source frame is an empty dict, then just return the same IDs

    source_frame_points = [value for value in trackers_source_frame.values()]
    new_frame_points = [value for value in trackers_new_frame.values()]

    distance = get_distance_frame(gt_points_frame=source_frame_points,
                                            tracker_points_frame=new_frame_points,
                                            radius=radius,
                                            max_distance=588)
    match_rows, match_cols = linear_sum_assignment(distance)  # Hungarian matching

    global_id_list = [id_num for id_num in trackers_source_frame.keys()]

    new_frame_id_dict = {}
    for pr_id, point in trackers_new_frame.items():
        col_index = list(match_cols).index(
            pr_id) if pr_id in match_cols else None  # Get's the index of this ID in match_cols
        source_index = match_rows[
            col_index] if col_index != None else None  # The corresponding matched row is the ID as per source frame
        distance_in_radius = distance[source_index][pr_id] <= radius if source_index != None else False
        if distance_in_radius:
            new_frame_id_dict[list(trackers_source_frame.items())[source_index][0]] = point  # Add that ID to this point
        else:
            current_last_id = max(global_id_list) if global_id_list else -1
            new_frame_id_dict[current_last_id + 1] = point
            global_id_list.append(current_last_id + 1)
            trackers_source_frame[current_last_id + 1] = point
    return new_frame_id_dict, trackers_source_frame


def get_global_matched_trackers(sequence, radius=20):
    """Sequence is a list where each frame is a dict of tracker IDs"""

    # If there is just one element in the sequence then return that element
    if len(sequence) <= 1: return sequence
    global_frame = sequence[0]
    target_frame = sequence[1]  # Set target frame as first element
    matched_trackers = [global_frame]  # Initialising the tracked sequence with 0th element

    for i in range(len(sequence) - 1):
        matched_target_frame, global_frame = get_global_matched_id_dict(trackers_source_frame=global_frame,
                                                                        trackers_new_frame=target_frame,
                                                                        radius=radius)
        for id_ in global_frame.keys():
            if id_ in matched_target_frame.keys():
                x_mean = np.mean([global_frame[id_][0], matched_target_frame[id_][0]])
                y_mean = np.mean([global_frame[id_][1], matched_target_frame[id_][1]])
                global_frame[id_] = (x_mean, y_mean)

        matched_trackers.append(matched_target_frame)
        try:
            target_frame = sequence[i + 2]  # The new frame is the target frame
        except IndexError:
            break
    return matched_trackers


def get_distance_point(gt_point, tracker_point):
    return abs(math.sqrt((gt_point[0] - tracker_point[0]) ** 2 + (gt_point[1] - tracker_point[1]) ** 2))


def get_distance_frame(gt_points_frame, tracker_points_frame, radius=6, max_distance=588):
    distance = np.zeros((len(gt_points_frame), len(tracker_points_frame)))
    for i, gt_point in enumerate(gt_points_frame):
        for j, tracker_point in enumerate(tracker_points_frame):
            norm = get_distance_point(gt_point=gt_point,
                                      tracker_point=tracker_point)
            distance[i][j] = norm if norm <= radius else max_distance+1
    return distance


def compute_distance(p1, p2):
    if list(p2) == [None, None]:
        return None
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
