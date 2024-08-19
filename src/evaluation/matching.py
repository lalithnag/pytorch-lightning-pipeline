import random
import datetime
import os
import sys
import skimage.draw
import skimage.measure
import skimage.morphology
from scipy import ndimage
from copy import deepcopy
import math
import json
import traceback

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  
project_root = os.path.dirname(os.path.dirname(current_dir))  
sys.path.append(project_root)

from src.utils import utils as ut
from src.evaluation import pt_utils


class MatchFrame:
    def __init__(self,
                 gt_frame_dicts_seq,
                 pred_frame_dicts_seq,
                 threshold=6):
        super(MatchFrame, self).__init__()

        self.gt_frame_dicts_seq = gt_frame_dicts_seq
        self.pred_frame_dicts_seq = pred_frame_dicts_seq
        self.threshold = threshold

    @staticmethod
    def matching_naive(pred_points, gt_points, threshold):
        return match_keypoints(pred_points, gt_points, threshold)

    @staticmethod
    def matching_evaluate(pred_points, gt_points, threshold):
        _, _, _, tp_pred = evaluate(pred_points, gt_points, threshold)
        return [[pred_points[pred['prediction_index']], gt_points[pred['label_index']]] for pred in tp_pred]

    @staticmethod
    def matching_hungarian(pred_points, gt_points, threshold):
        _,_,_, dist_mat, match_rows, match_cols = hungarian_matching(pred_points, gt_points, threshold)
        matched_regions = []
        for i, j in zip(match_rows, match_cols):
            if dist_mat[i][j] <= 6 : matched_regions.append([pred_points[j], gt_points[i]])
        return matched_regions

    def frame_to_seq(self, matching_func):
        """Wrapper function to apply a matching function for every frame of the sequence
        """
        def wrapper(*args):
            return [matching_func(list(pred_frame.values()), list(gt_frame.values()), self.threshold)
                    for pred_frame, gt_frame in zip(self.pred_frame_dicts_seq, self.gt_frame_dicts_seq)]
        return wrapper

    def get_matched_pred_dicts(self, matched_regions_seq):
        tp_gt_pts_gt_ids_seq = []
        tp_pred_pts_gt_ids_seq, tp_pred_pts_pred_ids_seq = [], []
        fn_gt_pts_gt_ids_seq, fp_pred_pts_pred_ids_seq = [], []
        tp_seq, fp_seq, fn_seq = [], [], []

        for gt_frame, pred_frame, matched_regions_frame in zip(self.gt_frame_dicts_seq,
                                                               self.pred_frame_dicts_seq,
                                                               matched_regions_seq):
            tp_gt_pts_gt_ids_frame = {}
            tp_pred_pts_gt_ids_frame, tp_pred_pts_pred_ids_frame = {}, {}
            fn_gt_pts_gt_ids_frame, fp_pred_pts_pred_ids_frame = {}, {}

            for i, (pred_match, gt_match) in enumerate(matched_regions_frame):
                matched_gt_label = list(gt_frame.keys())[list(gt_frame.values()).index(gt_match)]
                matched_pred_label = list(pred_frame.keys())[list(pred_frame.values()).index(pred_match)]

                tp_gt_pts_gt_ids_frame[matched_gt_label] = gt_match
                tp_pred_pts_gt_ids_frame[matched_gt_label] = pred_match
                tp_pred_pts_pred_ids_frame[matched_pred_label] = pred_match

                fn_gt_pts_gt_ids_frame[matched_gt_label] = ()
                fp_pred_pts_pred_ids_frame[matched_pred_label] = ()

            for label, point in gt_frame.items():
                if label not in tp_gt_pts_gt_ids_frame.keys():
                    tp_gt_pts_gt_ids_frame[label] = ()
                    fn_gt_pts_gt_ids_frame[label] = point

            for label, point in pred_frame.items():
                if label not in tp_pred_pts_pred_ids_frame:
                    fp_pred_pts_pred_ids_frame[label] = point
                    tp_pred_pts_pred_ids_frame[label] = ()

            tp_seq.append(len(matched_regions_frame))  # number of pred_matches
            fp_seq.append(len(list(pred_frame.keys())) - len(matched_regions_frame))
            fn_seq.append(len(list(gt_frame.keys())) - len(matched_regions_frame))

            tp_gt_pts_gt_ids_seq.append(tp_gt_pts_gt_ids_frame)
            tp_pred_pts_gt_ids_seq.append(tp_pred_pts_gt_ids_frame)
            tp_pred_pts_pred_ids_seq.append(tp_pred_pts_pred_ids_frame)
            fn_gt_pts_gt_ids_seq.append(fn_gt_pts_gt_ids_frame)
            fp_pred_pts_pred_ids_seq.append(fp_pred_pts_pred_ids_frame)

        return {'tp_gt_pts_gt_ids_seq': tp_gt_pts_gt_ids_seq,
                'tp_pred_pts_gt_ids_seq': tp_pred_pts_gt_ids_seq,
                'tp_pred_pts_pred_ids_seq': tp_pred_pts_pred_ids_seq,
                'fn_gt_pts_gt_ids_seq': fn_gt_pts_gt_ids_seq,
                'fp_pred_pts_pred_ids_seq': fp_pred_pts_pred_ids_seq,
                'tp_seq': tp_seq, 'fp_seq': fp_seq, 'fn_seq': fn_seq}

    def get_matches(self, matching_method='evaluate'):
        match_dict = {'naive': self.matching_naive,
                      'secondary': self.matching_evaluate,
                      'hungarian': self.matching_hungarian,
                      'hota': self.matching_hota}
        if matching_method=='hota':
            self.pred_frame_dicts_seq = self.get_global_matched_trackers(self.pred_frame_dicts_seq)
        return self.get_matched_pred_dicts(match_dict[matching_method]()) if matching_method == 'hota' else \
            self.get_matched_pred_dicts(self.frame_to_seq(match_dict[matching_method])())

    def get_global_matched_id_dict(self, trackers_source_frame, trackers_new_frame, radius=20):
        trackers_source_frame = dict(sorted(trackers_source_frame.items()))
        trackers_new_frame = dict(sorted(trackers_new_frame.items()))
        if not trackers_source_frame: return trackers_new_frame, trackers_source_frame
        # If the source frame is an empty dict, then just return the same IDs

        source_frame_points = [value for value in trackers_source_frame.values()]
        new_frame_points = [value for value in trackers_new_frame.values()]

        distance = pt_utils.get_distance_frame(gt_points_frame=source_frame_points,
                                               tracker_points_frame=new_frame_points,
                                               radius=self.threshold,
                                               max_distance=588)
        match_rows, match_cols = linear_sum_assignment(distance)  # Hungarian matching
        global_id_list = list(trackers_source_frame.keys())

        new_frame_id_dict = {}
        for pr_id, point in trackers_new_frame.items():
            # Get the index of this ID in match_cols
            col_index = list(match_cols).index(pr_id) if pr_id in match_cols else None
            # The corresponding matched row is the ID as per source frame
            source_index = match_rows[col_index] if col_index is not None else None
            distance_in_radius = distance[source_index][pr_id] <= radius if source_index is not None else False
            if distance_in_radius:
                # Add that ID to this point
                new_frame_id_dict[list(trackers_source_frame.items())[source_index][0]] = point
            else:
                current_last_id = max(global_id_list) if global_id_list else -1
                new_frame_id_dict[current_last_id + 1] = point
                global_id_list.append(current_last_id + 1)
                trackers_source_frame[current_last_id + 1] = point
        return new_frame_id_dict, trackers_source_frame

    def get_global_matched_trackers(self, sequence, radius=20):
        """Sequence is a list where each frame is a dict of tracker IDs"""
        # If there is just one element in the sequence then return that element
        if len(sequence) <= 1: return sequence
        global_frame = sequence[0]
        target_frame = sequence[1]  # Set target frame as first element
        matched_trackers = [global_frame]  # Initialising the tracked sequence with 0th element

        for i in range(len(sequence) - 1):
            matched_target_frame, global_frame = self.get_global_matched_id_dict(trackers_source_frame=global_frame,
                                                                                 trackers_new_frame=target_frame,
                                                                                 radius=radius)
            for id_ in global_frame.keys():
                if id_ in matched_target_frame.keys():
                    x_mean = np.mean([global_frame[id_][0], matched_target_frame[id_][0]])
                    y_mean = np.mean([global_frame[id_][1], matched_target_frame[id_][1]])
                    global_frame[id_] = (x_mean, y_mean)
            matched_trackers.append(matched_target_frame)
            try: target_frame = sequence[i + 2]  # The new frame is the target frame
            except IndexError: break
        return matched_trackers

    def get_temporal_assigned_gt_frame_dicts_seq(self):
        temporal_union_gt_ids = pt_utils.get_union_of_n_lists([list(frame_dict.keys())
                                                               for frame_dict in self.gt_frame_dicts_seq])
        temporal_gt_id_dict = {label: id_ for id_, label in enumerate(temporal_union_gt_ids)}
        # converts 'Ausstich1': (123, 21) into 0:(123, 21) where 0 is assigned to the list of temporal union of GT IDs
        return [{temporal_gt_id_dict[frame_dict_id]: frame_dict_label
                 for frame_dict_id, frame_dict_label in frame_dict.items()}
                for frame_dict in self.gt_frame_dicts_seq]

    def matching_hota(self):
        res = dict()

        gt_dets = self.get_temporal_assigned_gt_frame_dicts_seq()
        tracker_dets = self.get_global_matched_trackers(self.pred_frame_dicts_seq)

        [gt_ids, tracker_ids] = map(lambda x: [list(xi.keys()) for xi in x], [gt_dets, tracker_dets])

        num_gt_ids = len(list(set().union(*gt_ids)))
        num_tracker_ids = len(list(set().union(*tracker_ids)))

        potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
        gt_id_count = np.zeros((num_gt_ids, 1))
        tracker_id_count = np.zeros((1, num_tracker_ids))

        matched_gt_dets = [[]]*len(gt_ids)  # Length of sequence
        matched_tracker_dets = [[]]*len(tracker_ids)  # Length of sequence

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):

            gt_ids_t = np.asarray(gt_ids_t)
            tracker_ids_t = np.asarray(tracker_ids_t)

            gt_points_t = [gt_dets[t][index] for index in gt_ids_t]
            tracker_points_t = [tracker_dets[t][index] for index in tracker_ids_t]

            similarity = pt_utils.get_distance_frame(gt_points_frame=gt_points_t,
                                                     tracker_points_frame=tracker_points_t)
            similarity[similarity > 6] = 588
            sim_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim = np.zeros_like(similarity)
            sim_mask = sim_denom > 0 + np.finfo('float').eps
            sim[sim_mask] = similarity[sim_mask] / sim_denom[sim_mask]

            if len(gt_ids_t) and len(tracker_ids_t):
                potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim
            if len(gt_ids_t):
                gt_id_count[np.asarray(gt_ids_t)] += 1
            if len(tracker_ids_t):
                tracker_id_count[0, np.asarray(tracker_ids_t)] += 1

        ####################################### (3)
        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_count = np.zeros_like(potential_matches_count)

        metric = {'HOTA_TP': 0,
                  'HOTA_FN': 0,
                  'HOTA_FP': 0,
                  'LocA': 0}

        matches = []
        ####################################### (4)
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            gt_ids_t = np.asarray(gt_ids_t)
            tracker_ids_t = np.asarray(tracker_ids_t)

            gt_points_t = [gt_dets[t][index] for index in gt_ids_t]
            tracker_points_t = [tracker_dets[t][index] for index in tracker_ids_t]

            ####################################### (5)
            match_rows_th6 = []
            match_cols_th6 = []
            num_matches = 0

            similarity = pt_utils.get_distance_frame(gt_points_frame=gt_points_t,
                                                      tracker_points_frame=tracker_points_t)
            similarity[similarity > 6] = 588
            # print(similarity)

            if len(gt_ids_t) and len(tracker_ids_t):
                score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
                match_rows, match_cols = linear_sum_assignment(score_mat)

                actually_matched_mask = similarity[match_rows, match_cols] <= 6
                match_rows_th6 = match_rows[actually_matched_mask]
                match_cols_th6 = match_cols[actually_matched_mask]
                num_matches = len(match_rows_th6)

                matched_gt_dets[t] = [gt_dets[t][id_] for id_ in gt_ids_t[match_rows_th6]]
                matched_tracker_dets[t] = [tracker_dets[t][id_] for id_ in tracker_ids_t[match_cols_th6]]

            matches.append([match_rows_th6, match_cols_th6])

            #metric['HOTA_TP'] += num_matches
            #metric['HOTA_FN'] += len(gt_ids_t) - num_matches
            #metric['HOTA_FP'] += len(tracker_ids_t) - num_matches

            #if num_matches > 0:
            #    metric['LocA'] += sum(similarity[match_rows_th6, match_cols_th6])
            #    if len(match_rows_th6) and len(match_cols_th6):
            #        matches_count[gt_ids_t[match_rows_th6], tracker_ids_t[match_cols_th6]] += 1

        matched_regions = [[(tracker_point, gt_point) for tracker_point, gt_point in zip(tracker_det, gt_det)]
                           for tracker_det, gt_det in zip(matched_tracker_dets, matched_gt_dets)]

        ####################################### (7)
        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        #ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        #metric['AssA'] = np.sum(matches_count * ass_a) / np.maximum(1, metric['HOTA_TP'])
        #ass_re = matches_count / np.maximum(1, gt_id_count)
        #metric['AssRe'] = np.sum(matches_count * ass_re) / np.maximum(1, metric['HOTA_TP'])
        #ass_pr = matches_count / np.maximum(1, tracker_id_count)
        #metric['AssPr'] = np.sum(matches_count * ass_pr) / np.maximum(1, metric['HOTA_TP'])

        # Calculate final scores
        #metric['LocA'] = np.maximum(1e-10, metric['LocA']) / np.maximum(1e-10, metric['HOTA_TP'])
        #metric['DetRe'] = metric['HOTA_TP'] / np.maximum(1, metric['HOTA_TP'] + metric['HOTA_FN'])
        #metric['DetPr'] = metric['HOTA_TP'] / np.maximum(1, metric['HOTA_TP'] + metric['HOTA_FP'])
        #metric['DetA'] = metric['HOTA_TP'] / np.maximum(1, metric['HOTA_TP'] + metric['HOTA_FN'] + metric['HOTA_FP'])

        #metric['HOTA'] = np.sqrt(metric['DetA'] * metric['AssA'])
        #metric['RHOTA'] = np.sqrt(metric['DetRe'] * metric['AssA'])

        return matched_regions

    @staticmethod
    def get_temporal_match_pred_dict_frame(matched_pred_dicts, gt_id, gt_pts, pred_pts, frame_num):
        tp_pred_pts_gt_ids = matched_pred_dicts['tp_pred_pts_gt_ids_seq']
        tp_pred_pts_pred_ids = matched_pred_dicts['tp_pred_pts_pred_ids_seq']

        try:
            pred_pt = tp_pred_pts_gt_ids[frame_num][gt_id]
            gt_pt = matched_pred_dicts['tp_gt_pts_gt_ids_seq'][frame_num][gt_id]
        except KeyError: return None

        pred_id = list(tp_pred_pts_pred_ids[frame_num].keys())[
            list(tp_pred_pts_pred_ids[frame_num].values()).index(pred_pt)]

        tpa_pred_pts = [pred_pts_gt_ids_frame[gt_id] if gt_id in list(pred_pts_gt_ids_frame.keys())
                                                        and pred_id in list(pred_pts_pred_ids_frame.keys())
                                                        and pred_pts_gt_ids_frame[gt_id] == pred_pts_pred_ids_frame[
                                                            pred_id] else ()
                        for pred_pts_gt_ids_frame, pred_pts_pred_ids_frame in
                        zip(tp_pred_pts_gt_ids, tp_pred_pts_pred_ids)]

        tpa_gt_pts_gt_id = {gt_id: [gt_pts_gt_ids_frame[gt_id] if gt_id in list(gt_pts_gt_ids_frame.keys())
                                                                and tpa_pred_pts[i] else ()
                                    for i, gt_pts_gt_ids_frame in enumerate(matched_pred_dicts['tp_gt_pts_gt_ids_seq'])]}

        fna_gt_pts_gt_id = {gt_id: [gt_pt_frame if not tpa_pred_pts[i] else ()
                                    for i, gt_pt_frame in enumerate(gt_pts)]}

        fpa_pred_pts_pred_id = {pred_id: [pred_pt_frame if not tpa_pred_pts[i] else ()
                                          for i, pred_pt_frame in enumerate(pred_pts[pred_id])]}

        return {'tpa_pred_pts_gt_id': {gt_id: tpa_pred_pts},
                'tpa_pred_pts_pred_id': {pred_id: tpa_pred_pts},
                'tpa_gt_pts_gt_id': tpa_gt_pts_gt_id,
                'fna_gt_pts_gt_id': fna_gt_pts_gt_id,
                'fpa_pred_pts_pred_id': fpa_pred_pts_pred_id,
                'gt_pt': gt_pt,
                'pred_pt': pred_pt}

    @staticmethod
    def get_multi_view_match_pred_dict(matched_pred_dicts, matched_pred_dicts_opp, gt_id, frame_num):
        tp_pred_pts_pred_ids_left = matched_pred_dicts['tp_pred_pts_gt_ids_seq']
        tp_pred_pts_pred_ids_right = matched_pred_dicts_opp


def centres_of_mass(mask, threshold):
    mask_bin = deepcopy(mask)
    mask_bin[mask >= threshold] = 1
    mask_bin[mask < threshold] = 0

    label_regions, num_regions = skimage.measure.label(mask_bin, background=0, return_num=True)
    indexlist = [item for item in range(1, num_regions + 1)]
    return ndimage.measurements.center_of_mass(mask_bin, label_regions, indexlist)


def match_keypoints(kplist1, kplist2, threshold):
    """
    compare two lists of keypoints and match the keypoints.

    :param list kplist1: list containing keypoints
    :param list kplist2: list containing keypoints
    :param float threshold: maximal distance for matching
    :return: list of matching keypoints with length of kplist1
    """
    labellist_matched = []
    for elem1 in kplist1:
        match = 0
        labeldistance = threshold
        for elem2 in kplist2:
            if pt_utils.compute_distance(elem1, elem2) < labeldistance:
                labeldistance = pt_utils.compute_distance(elem1, elem2)
                # print(labeldistance, elem1, elem2)
                match = elem2
        if match:
            labellist_matched.append([elem1, match])
            # kplist1.remove(elem1)
            # kplist2.remove(match)
        # else:
        #     labellist_matched.append([elem1, [None, None]])
    # from scipy.spatial import distance
    # m2 = distance.cdist(pred_regions, mask_regions, 'euclidean') --> Matrix
    return remove_dupes(labellist_matched)


def remove_dupes(matchlist):
    """
    removes duplicated second elements in matchlist
    m = [[(1, 2), (3, 4)], [(4, 2), (3, 7)], [(2, 8), (3, 7)], [(5, 8), (3, 7)], [(10, 2), (3, 4)], [(6, 3), (2, 4)]]
    m_c = [[(1, 2), (3, 4)], [(2, 8), (3, 7)], [(6, 3), (2, 4)]]
    :return:
    """
    dupe_idx = []
    for ind1, row1 in enumerate(matchlist):
        for ind2, row2 in enumerate(matchlist):
            if (row1[1] == row2[1]) and (ind1 != ind2):
                if [ind2, ind1] not in dupe_idx:
                    dupe_idx.append([ind1, ind2])

    m_cleared = deepcopy(matchlist)
    for dupe in dupe_idx:
        pair1 = matchlist[dupe[0]]
        pair2 = matchlist[dupe[1]]
        d1 = pt_utils.compute_distance(pair1[0], pair1[1])
        d2 = pt_utils.compute_distance(pair2[0], pair2[1])
        if d1 < d2:
            if pair2 in m_cleared:
                m_cleared.remove(pair2)
        else:
            if pair1 in m_cleared:
                m_cleared.remove(pair1)
    return m_cleared


def evaluate(predictions, labels, radius):
    """
    Evaluate an array of predictions based on an array of labels and their given radius
    True positive gets calculated by matching predictions with labels from shortest distance to longest distance.
    False positive are all predictions without a label.
    False negative are all label without a prediction.
    :param predictions: an array of predictions with x and y coordinates
    :param labels: an array of labels with x and y coordinates
    :param radius: the radius around the labels within which a prediction is still correct
    :returns: the amount of true positive (TP) (label with prediction), false positive (FP) (prediction with no label) and false negative (FN) (label with no prediction) labels
    """
    # count all labels in radius of each prediction
    labels_in_radius_of_all_predictions = []

    # iterate all predictions
    for prediction_index, prediction in enumerate(predictions):
        labels_in_radius_of_prediction = []
        # for each label
        for label_index, label in enumerate(labels):
            # get the distance to all close labels for each prediction
            distance = abs(math.sqrt((label[0] - prediction[0])**2 + (label[1] - prediction[1])**2))
            # save all close labels of the prediction
            if distance <= radius:
                labels_in_radius_of_prediction.append(
                    {"prediction_index": prediction_index, "label_index": label_index, "distance": distance})
        labels_in_radius_of_all_predictions.append(labels_in_radius_of_prediction)

    # all true positive predictions with labels and distance
    true_positive_predictions = []
    # check if any predictions have close labels
    # find all matching pairs of predictions and labels starting with the closest pair
    while max([len(_) for _ in labels_in_radius_of_all_predictions], default=0) >= 1:
        # the closest pair of any prediction and any label
        closest_prediction_label_pair = None
        # iterate the predictions
        for labels_in_radius_of_prediction in labels_in_radius_of_all_predictions:
            # choose the prediction and label with the shortest distance
            for close_label in labels_in_radius_of_prediction:
                if closest_prediction_label_pair == None or \
                        close_label["distance"] <= closest_prediction_label_pair["distance"]:
                    closest_prediction_label_pair = close_label
        # the best prediction is a true positive prediction
        true_positive_predictions.append(closest_prediction_label_pair)
        # make sure this prediction does not get picked again
        labels_in_radius_of_all_predictions[closest_prediction_label_pair["prediction_index"]] = []
        # make sure this label does not get picked again
        for index, labels_in_radius_of_prediction in enumerate(labels_in_radius_of_all_predictions):
            # remove the label of the best prediction from all other predictions
            labels_in_radius_of_all_predictions[index] = [
                close_label for close_label in labels_in_radius_of_prediction
                if close_label["label_index"] != closest_prediction_label_pair["label_index"]]

    # the amount of true positives is just the amount of found predictions and labels matches
    true_positive = len(true_positive_predictions)
    # the amount of false positives is the amount of predictions not found in the predictions and labels matches
    false_positive = len([prediction for index, prediction in enumerate(predictions) if len(
        [tp_prediction for tp_prediction in true_positive_predictions
         if tp_prediction["prediction_index"] == index]) == 0])
    # the amount of false negatives is the amount of labels not found in the predictions and labels matches
    false_negative = len([label for index, label in enumerate(labels) if len(
        [tp_prediction for tp_prediction in true_positive_predictions if tp_prediction["label_index"] == index]) == 0])
    return true_positive, false_positive, false_negative, true_positive_predictions


def hungarian_matching(predictions, labels, radius):
    # Get the matches using hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    def get_distance(label, prediction):
        return abs(math.sqrt((label[0] - prediction[0]) ** 2 + (label[1] - prediction[1]) ** 2))
    # compute similarity
    distance = np.zeros((len(labels), len(predictions)))
    for i, gt_point in enumerate(labels):
        for j, pred_point in enumerate(predictions):
            distance[i][j] = get_distance(label=gt_point, prediction=pred_point)
    match_rows, match_cols = linear_sum_assignment(distance)
    # Define TP, FP, and FN
    tp = 0
    p = len(match_rows)
    for i, j in zip(match_rows, match_cols):
        if distance[i][j] <= radius:
            tp += 1
    fn = len(labels) - tp
    fp = p - tp
    return tp, fp, fn, distance, match_rows, match_cols
