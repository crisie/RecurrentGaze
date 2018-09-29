# !/usr/bin/env python
# title           :data_utils.py
# description     :Script with data utilites
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

import numpy as np
import os
from keras.preprocessing import image
import re
import cv2 as cv
import copy
import random
from enum import Enum
from collections import namedtuple
import tensorflow as tf
from keras import backend as K


class input_type(Enum):
    """
    Types of network inputs
    """
    FACE = 'face'
    EYES = 'eyes'
    LANDMARKS = 'landmarks'


""" 
Data structure containing:
- x: img directories
- y: labels
- feats: information needed to preprocess the data
- indxs: data indexes
- parts: validation participants (if any)
"""
DataTuple = namedtuple('DataE', 'x y feats idxs parts')


def prepare_tensor_dims(img: np.ndarray):
    """
    Modify tensor dimensions so that it's compatible with network, and change channel order.
    :param img: input image
    :return: ready-to-go image
    """
    if img.ndim < 4:
        img = np.expand_dims(img, axis=0)
    img = img[:, :, :, ::-1]
    return img


def unison_shuffled_copies(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    Get shuffled copies of input data in unison
    :param a: input data a
    :param b: input data b
    :param c: input data c
    :return: unison shuffled copies of input data
    """
    assert len(a) == len(b)
    assert len(b) == len(c)
    p = np.random.permutation(len(a))
    return [a[i] for i in p], [b[i] for i in p], [c[i] for i in p]


def unison_shuffled_copy(data: DataTuple):
    """
    Get shuffled copies of DataTuple elements x y and feats. It does not shuffle Indexes.
    :param data: DataTuple element
    :return: DataTuple unison shuffled copy of input data
    """
    a, b, c = unison_shuffled_copies(data.x, data.y, data.feats)
    data = data._replace(x=a)
    data = data._replace(y=b)
    data = data._replace(feats=c)
    return data


def matches_key(key: str, message: str):
    """
    Find key in message
    :param key: key to find in message
    :param message: message
    :return: matched key
    """
    return len(re.findall(r'(?<!\d)' + key, message)) > 0


def compute_min_max_landmarks_fold(train_feats: list, normalize: bool=True):
    """
    Compute min and maximum landmarks from training landmarks.
    :param train_feats: features including landmarks
    :param normalize: True if landmarks have to be converted to normalized space
    :return: compute min and max landmarks
    """
    min_lndmk = np.array([9999.0, 9999.0, 9999.0], dtype=np.float32)
    max_lndmk = np.array([-9999.0, -9999.0, -9999.0], dtype=np.float32)
    for fram in train_feats:
        frame = copy.deepcopy(fram)
        landmarks = format_map_to_vector(frame, 13, 81)

        mean_face = np.mean(landmarks, axis=0)
        if normalize:
            norm_matrix = get_face_conv(frame)
            mean_face, landmarks = transform_landmarks(landmarks, norm_matrix, mean_face)

        lndmks = landmarks - mean_face
        min_landmark = np.min(lndmks, axis=0)
        max_landmark = np.max(lndmks, axis=0)
        min_lndmk = np.array([np.min([min_landmark[i], min_lndmk[i]]) for i, d in enumerate(min_landmark)])
        max_lndmk = np.array([np.max([max_landmark[i], max_lndmk[i]]) for i, d in enumerate(max_landmark)])

    # Take into account that horizontal flip is possible, so adjust x axis accordingly
    if abs(min_lndmk[0] > abs(max_lndmk[0])):
        max_lndmk[0] = abs(min_lndmk[0])
    elif abs(min_lndmk[0] < abs(max_lndmk[0])):
        min_lndmk[0] = - max_lndmk[0]

    return min_lndmk, max_lndmk


def add_minxmax_landmarks_values(face_features: list, min_lndmk: list, max_lndmk: list):
    """
    Add min and max landmarks values to face_features list
    :param face_features: list of features needed to preprocess data
    :param min_lndmk: minimum landmark coordinate
    :param max_lndmk: maximum landmark coordinate
    :return: modified face_features including min and max landmarks at the end of each row
    """
    feats = np.hstack([face_features, np.zeros((len(face_features), 6))])
    for frame in feats:
        frame[-6:-3] = min_lndmk
        frame[-3:] = max_lndmk
    return feats


def train_valtest_split(data: np.ndarray, gt: np.ndarray, face_features: np.ndarray, test_folders: list,
                        validation_parts: int = 0):
    """
    Get training/test split, based on input test_folders. Since the purpose of the code is to have person-independent
    models, the function assumes that participants included in test folders are not included in training whatsoever.
    If validation_parts > 0, #validation_parts participants would be extracted from training split, thus having
    training/validation/test split, but in this case only training/validation splits would be returned. See
    get_validation_split to know more details about this extraction process. The function will return empty (None)
    Validation/test split if there is no validation/test data.

    :param data: image paths
    :param gt: gaze vector ground truth
    :param face_features: face features
    :param test_folders: list of test folders
    :param validation_parts: number of participants used for validation
    :return: DataTuple containing all data/gt/feats for each training and validation/test splits. Note: validation/test
            DataTuple will be None if there is no validation/test data.
    """
    validation_participants = None
    validation_idxs = []
    test_idxs = []
    if len(test_folders) > 0:
        if validation_parts > 0:
            validation_idxs, validation_participants = get_validation_split(validation_parts, data, test_folders)

        for folder in test_folders[0]:
            test_idxs += [i for i, item in enumerate(data) if matches_key(folder, item)]

        if validation_parts == 0:
            validation_idxs = test_idxs

        validation = DataTuple(x=[data[i] for i in validation_idxs],
                               y=[gt[i] for i in validation_idxs],
                               feats=[face_features[i] for i in validation_idxs],
                               idxs=validation_idxs,
                               parts=(validation_participants if validation_participants is not None else test_folders))
    else:
        validation = None

    train_idxs = [i for i in range(len(data)) if i not in test_idxs and i not in validation_idxs]

    train = DataTuple(x=[data[i] for i in train_idxs],
                      y=[gt[i] for i in train_idxs if len(gt) > 0],
                      feats=[face_features[i] for i in train_idxs],
                      idxs=train_idxs,
                      parts=None)

    return train, validation


def get_validation_split(validation_parts: int, data: np.ndarray, test_folders: list):
    """
    Get random validation split from participants not included in test
    :param validation_parts: number of validation participants
    :param data: image paths
    :param test_folders: list of test folders
    :return: validation indexes and selected validation participants
    """

    test_participants = set([int(i.split("_")[0]) for i in test_folders])
    all_participants = set(range(1, 17))
    train_participants = list(all_participants - test_participants)
    validation_participants = random.sample(train_participants, validation_parts)
    validation_participants_str = [str(s) + "_" for s in validation_participants]

    validation_idxs = []
    for part in validation_participants_str:
        validation_idxs += [i for i, item in enumerate(data) if matches_key(part, item)]
    return validation_idxs, validation_participants


def arrange_sequences(data: DataTuple, look_back: int=25, max_look_back: int=25):
    """
    Re-arrange data so that there's one array of images per sequence, by applying a sliding window of stride 1 among the
    data. Care is taken so that jumps between frames are not included in same sequence (that is, only contiguous frames
    are included in a sequence).
    :param data: input DataTuple data
    :param look_back: sequence length
    :param max_look_back: Used for validation/test purposes. If different sequence lengths are being compared in
            different experiments, to be fair they have to be compared using the same validation sequences.
            This value specifies the maximum look back used in those experiments, so that all models are validated
            with the same sequences.
    :return: modified DataTuple data with re-arranged data
    """
    x, y, feats, idxs = [], [], [], []
    seqs = np.array([i[0] for i in data.feats])
    seqs_length = np.bincount(np.array(seqs, dtype="int64"))
    for s in range(len(seqs_length)):
        if seqs_length[s] >= max_look_back:
            idxs_tmp = np.where(seqs == s)[0]
            for i in range(idxs_tmp[0] + max_look_back - look_back, idxs_tmp[-1] - (look_back - 1) + 1):
                x.append(data.x[i:i + look_back])
                y.append(data.y[i:i + look_back])
                feats.append(data.feats[i:i + look_back])
                idxs.append(data.idxs[i:i + look_back])

    return DataTuple(x, y, feats, idxs, data.parts)


def transform_landmarks(landmarks: np.ndarray, conv_mat: np.ndarray, mean_face: np.ndarray):
    """
    Convert landmarks to normalized space
    :param landmarks: list of 3D landmarks
    :param conv_mat: conversion matrix
    :param mean_face: mean face coordinates
    :return: normalized landmarks and mean face point
    """
    landmarks = conv_mat.dot(landmarks.transpose()).transpose()
    mean_face = conv_mat.dot(mean_face)
    return landmarks, mean_face


def preprocess_oface(img: np.ndarray, bb: list):
    """
    Preprocess original face image, so that image limits are valid.
    :param img: original image
    :param bb: bounding box
    :return: cropped original face patch
    """
    # add some borders to the defined BB to allow for some horizontal/vertical shift when augmenting
    bb_uptemp = update_borders(bb, (int(bb[3] + bb[3] * 0.05), int(bb[2] + int(bb[2] * 0.05))))

    # check if the borders of the bb exceed the limits of the original image. If so, correct it and save the
    # changes for initial x and y in changed (which are the pixels below 0)
    bb_temp, changed = check_valid_bb(bb_uptemp, img.shape)
    # Create face crop with new checked bb coordinates
    oface = img[int(bb_temp[1]):int(bb_temp[1] + bb_temp[2]),
            int(bb_temp[0]):int(bb_temp[0] + bb_temp[3]), :]
    # If the original crop had some value below 0, create new face crop with the limits of bb_uptemp
    # and invalid coordinates (below 0) with 0 value.
    if changed is not None:
        temp = np.zeros((int(bb_uptemp[2]), int(bb_uptemp[3]), 3), img.dtype)
        temp[int(changed[1]):int(changed[1] + bb_temp[2]), int(changed[0]):int(changed[0] + bb_temp[3]), :] = oface
        oface = temp

    return oface


def warp_image(img: np.ndarray, warp_mat: np.ndarray, roi_size: list):
    """
    Warp given image according to warp_mat and roi_size.
    :param img: original image
    :param warp_mat: transformation matrix
    :param roi_size: output image size
    :return: warped image
    """
    warped_image = cv.warpPerspective(img, warp_mat, (int(roi_size[0]), int(roi_size[1])))
    return warped_image


def denormalize_gaze(gaze_conv_mat: np.ndarray, y_2d):
    """
    De-normalize gaze back to original space and convert to angles.
    :param gaze_conv_mat: gaze conversion matrix
    :param y_2d: normalized 2D gaze angles
    :return: 2D gaze in original space
    """
    y_3d = numpy_angles2vector(y_2d)#[np.newaxis, :])
    gaze_conv_inv = np.linalg.inv(gaze_conv_mat)
    rec = gaze_conv_inv.dot(np.transpose(y_3d))
    rec = rec / np.linalg.norm(np.transpose(rec))

    #rec = vector2angles(rec)
    return rec#[0, :]


def update_borders(bb: list, roi_size: list):
    """
    Update bounding box borders according to roi_size. Borders are updated from center of image.
    :param bb: original bounding box
    :param roi_size: output bounding box size
    :return: modified bounding box
    """
    mid_x = bb[0] + bb[2] / 2
    mid_y = bb[1] + bb[3] / 2

    new_x = int(mid_x - roi_size[1] / 2)
    new_y = int(mid_y - roi_size[0] / 2)

    return [new_x, new_y, roi_size[0], roi_size[1]]


def crop_image_from_center(img: np.ndarray, output_size: list):
    """
    Crop given image starting counting from center of image.
    :param img: original image
    :param output_size: output size
    :return: cropped image
    """
    rows = output_size[0]
    cols = output_size[1]

    mid_x = int(img.shape[1] / 2)
    mid_y = int(img.shape[0] / 2)

    new_x = int(mid_x - cols / 2)
    new_y = int(mid_y - rows / 2)

    return img[new_y:new_y + rows, new_x:new_x + cols]


#def get_max_bb(bbs):
#    dims = [99999, 99999, 0, 0]
#    final_bb = np.array(dims)

#    final_bb[0] = np.min(bbs[:, 0])
#    final_bb[1] = np.min(bbs[:, 1])
#    max_tx = np.max(bbs[:, 0] + bbs[:, 3])
#    max_ty = np.max(bbs[:, 1] + bbs[:, 2])
#    final_bb[2] = max_ty - final_bb[1]
#    final_bb[3] = max_tx - final_bb[0]
#    if final_bb[3] > final_bb[2]:
#        final_bb[2] = final_bb[3]
#    elif final_bb[2] > final_bb[3]:
#        final_bb[3] = final_bb[2]

#    return final_bb


def check_valid_bb(bbo: list, img_shape: list):
    """
    Check bounding box limits are valid (within image). If not, correct them.
    :param bbo: original bounding box
    :param img_shape: original (big) image shape
    :return: modified bounding box, changes made
    """
    bb = np.array(bbo, copy=True)
    invalid = False
    changed = np.zeros((2,))
    if int(bbo[0]) < 0:
        bb[0] = 0
        bb[3] = bb[3] + bbo[0]
        changed[0] = -bbo[0]
        invalid = True
    if int(bbo[1]) < 0:
        bb[1] = 0
        bb[2] = bb[2] + bbo[1]
        changed[1] = -bbo[1]
        invalid = True
    if int(bb[1] + bb[2]) > img_shape[0]:
        bb[2] = img_shape[0] - bb[1] - 1
        invalid = True
    if int(bb[0] + bb[3]) > img_shape[1]:
        bb[3] = img_shape[1] - bb[0] - 1
        invalid = True

    if not invalid:
        changed = None

    return bb, changed


def format_map_to_vector(feature_lists: list, start: int, end: int):
    """
    Convert map into vector
    :param feature_lists: list of features, whose elements are maps.
    :param start: starting element within list
    :param end: ending element within list
    :return: vector (list) of selected features
    """
    nrange = np.arange(start, end)
    feat_vec = []
    for i in nrange:
        triplet = np.array(list(feature_lists[i]))
        feat_vec.append(triplet)

    feat_vec = np.array(feat_vec)
    return feat_vec


def min_max_normalization(data: np.ndarray, minimum: np.ndarray, maximum: np.ndarray, factor: float=0., eps: float =0.0000001):
    """
    Apply min/max normalization to data given minimum and maximum values. Final values are within range [0,factor].
    :param data: Original data
    :param minimum: Minimum value of data
    :param maximum: Maximum value of data
    :param factor: Final values are within range [0,factor].
    :param eps: eps
    :return: Normalized data
    """
    den = (maximum - minimum) + eps
    if factor > 0.:
        den = den * factor
    return (data - minimum) / den


def arrange_array(array_size: list):
    """
    Create numpy empty array given size
    :param array_size: array size
    :return: created empty array
    """
    return np.zeros(array_size, dtype=np.float32)


def arrange_array_size(batch_size: int, array_size: tuple):
    """
    Decide array size given batch size and array size [batch_size, height, width, channel].
    :param batch_size: batch size
    :param array_size: array (img) size
    :return: final array size
    """
    output = list(array_size)
    output.insert(0, batch_size)
    return output


def arrange_sequence_array_size(batch_size: int, array_size: tuple, seq_len: int):
    """
    Decide array size given batch size, array size and sequence length [batch_size, seq_len, height, width, channel].
    :param batch_size: batch size
    :param array_size: array (img) size
    :param seq_len: sequence length
    :return: final array size
    """
    output = arrange_array_size(batch_size, array_size)
    output.insert(1, seq_len)
    return output


def load_image(img: str):
    """
    Load image
    :param img: image directory
    :return: loaded image
    """
    return image.img_to_array(image.load_img(img)).astype(np.float32, copy=False)


def copy_face_features(feats: list):
    """
    Performs deep copy of feats
    :param feats: list of features
    :return: deep-copied features
    """
    return copy.deepcopy(feats)


def get_face_conv(feature: list):
    """
    Get face normalization matrix, to convert 3D face to normalized space.
    :param feature: list of features
    :return: 3D face normalization matrix
    """
    return np.array(list(feature[3])).reshape((3, 3))


def get_gaze_conv(feature: list):
    """
    Get gaze normalization matrix, to convert 3D gaze vector to normalized space.
    :param feature: list of features
    :return: 3D gaze normalization matrix
    """
    return np.array(list(feature[4])).reshape((3, 3))


def get_face_roi_size(feature: list):
    """
    Get face patch size
    :param feature: list of features
    :return: face patch size
    """
    return list(feature[11])


def get_eyes_roi_size(feature: list):
    """
    Get eye patch size
    :param feature: list of features
    :return: eye patch size
    """
    return list(feature[12])


def get_face_warp(feature: list):
    """
    Get transformation matrix to warp face image.
    :param feature: list of features
    :return: face warping matrix
    """
    return np.array(list(feature[2])).reshape((3, 3))


def get_leye_warp(feature: list):
    """
    Get transformation matrix to warp left eye
    :param feature: list of features
    :return: left eye warping matrix
    """
    return np.array(list(feature[5])).reshape((3, 3))


def get_reye_warp(feature: list):
    """
    Get transformation matrix to warp right eye
    :param feature: list of features
    :return: right eye warping matrix
    """
    return np.array(list(feature[8])).reshape((3, 3))


def get_bb(feature: list):
    """
    Get face bounding box
    :param feature: list of features
    :return: face bounding box
    """
    return list(feature[1])


def get_landmarks(feature: list):
    """
    Get 3D landmarks
    :param feature: list of features
    :return: 3D landmarks
    """
    return format_map_to_vector(feature, 13, 81)


def get_min_max_landmarks(feature: list):
    """
    Get minimum and maximum landmarks from training set
    :param feature: list of features
    :return: min/max landmarks
    """
    return feature[-6:-3], feature[-3:]


def normalize_gaze(gaze: np.ndarray, norm_mat: np.ndarray):
    """
    Convert gaze to normalized space.
    :param gaze: 3D unit gaze vector
    :param norm_mat: normalization matrix
    :return: normalized unit gaze vector
    """
    y = norm_mat.dot(gaze)
    y = y / np.linalg.norm(y)
    return y


def resize_oface(img: np.ndarray, bb: list, output_shape: tuple):
    """
    Crop and resize not-normalized image to given output shape.
    :param img: original face image
    :param bb: face bounding box
    :param output_shape: final face patch size
    :return: final face patch
    """
    oface = crop_image_from_center(img, (int(bb[3]), int(bb[2])))
    oface = cv.resize(oface, (output_shape[0], output_shape[1]))
    return oface


def resize_nface(img: np.ndarray, output_shape: tuple):
    """
    Crop normalized face
    :param img: original normalized face
    :param output_shape: final face patch size
    :return: final face patch
    """
    nface = crop_image_from_center(img, output_shape)
    return nface


def resize_eyes(leye: np.ndarray, reye: np.ndarray, output_shape: tuple):
    """
    Resize eye patches and convert them into one image with both eyes side to side.
    :param leye: original left eye patch
    :param reye: original right eye patch
    :param output_shape: final eyes patch size
    :return: final eyes patch
    """
    leye = crop_image_from_center(leye, (int(output_shape[0]), int(output_shape[1] / 2)))
    reye = crop_image_from_center(reye, (int(output_shape[0]), int(output_shape[1] / 2)))
    eyes = np.zeros((leye.shape[0], leye.shape[1] + reye.shape[1], leye.shape[2]))
    eyes[0:leye.shape[0], 0:leye.shape[1]] = leye
    eyes[0:reye.shape[0], leye.shape[1]:] = reye
    return eyes


def add_dimension(data: DataTuple):
    """
    Add dimension to data to be compatible with data format for sequences
    :param data: data
    :return: modified data
    """
    data = data._replace(x=np.expand_dims(data.x, axis=1))
    data = data._replace(y=np.expand_dims(data.y, axis=1))
    data = data._replace(feats=np.expand_dims(data.feats, axis=1))
    data = data._replace(idxs=np.expand_dims(data.idxs, axis=1))
    return data


def read_data_file(file: str):
    """
    Load specific data file (in this case only used to read data_files)
    :param file: data file
    :return: data file content
    """
    data_tmp = None
    if file != "":
        with open(file) as fp:
            data_tmp = fp.read().split("\n")
            data_tmp = data_tmp[:-1]
    return data_tmp


def read_input(data_files: list, path: str, gt_files: list = None, vector_gt_files: list = None,
               head_files: list = None):
    """
    Load and read input files, and convert their content into numpy arrays
    :param data_files: files containing the directories of input images
    :param path: path of input files
    :param gt_files: files containing 2D gaze ground truth (yaw and pitch) - just for legacy purposes
    :param vector_gt_files: files containing 3D ground truth (x,y,z unit vectors)
    :param head_files: files containing 2D head pose - just for legacy purposes
    :return: all input numpy arrays
    """

    # Data
    assert (len(data_files) > 0)
    data = []
    for n in data_files[0]:
        if n != "":
            data_tmp = read_data_file(n)
            data = np.concatenate((data, data_tmp), axis=0)
    data = np.array([os.path.join(path, filepath) for filepath in data])
    if os.name == "posix":
        data = np.array([str.replace(filepath, "\\", "/") for filepath in data])

    # Gaze GT (angle)
    gt = None
    if gt_files is not None:
        assert (len(gt_files) > 0)
        gt = np.empty((0, 2))
        for n in gt_files[0]:
            if n != "":
                gt_tmp = list(np.loadtxt(n))
                gt = np.concatenate((gt, gt_tmp), axis=0)
        assert (len(data) == len(gt))

    # Gaze GT (3D vector)
    vgt = None
    if vector_gt_files is not None:
        assert (len(vector_gt_files) > 0)
        vgt = np.empty((0, 3))
        for n in vector_gt_files[0]:
            if n != "":
                vgt_tmp = list(np.loadtxt(n))
                vgt = np.concatenate((vgt, vgt_tmp), axis=0)
        assert (len(data) == len(vgt))

    # Head GT (angle)
    head_gt = None
    if head_files is not None:
        head_gt = np.empty((0, 2))
        for n in head_files[0]:
            if n != "":
                head_gt_tmp = list(np.loadtxt(n))
                head_gt = np.concatenate((head_gt, head_gt_tmp), axis=0)
        assert (len(gt) == len(head_gt))

    return data, gt, vgt, head_gt


def read_face_features_file(files: list):
    """
    Reads and parses "face features" file. This file contains the following 81 elements
    (followed by their respective notation in BMVC paper):
    # 0 seq_num (sequence number); 1 bb (bounding box);
    # 2 face patch warp matrix (W); 3 face patch normalization mat. (M); 4 face patch gaze normalization matrix (R);
    # 5 left eye patch warp matrix (W); 6 left eye patch norm. mat. (M); 7 left eye patch gaze norm. mat. (R);
    # 8 right eye patch warp matrix (W); 9 right eye patch norm. mat. (M); 10 right eye patch gaze norm. mat. (R);
    # 11 face ROI size; 12 eye ROI size; 13 - 80 3d landmarks
    :param files: list of face features files
    :return: numpy array containing one row per frame, with elements parsed as maps (for legacy purposes).
    """
    convert = lambda x: map(float, x.decode('UTF-8').split(','))
    seq_num = 0
    face_feats = np.empty((0, 81))
    for f in files:
        me = dict([(n, convert) for n in range(0, 81)])
        if f != "":
            face_feats_tmp = np.genfromtxt(f, delimiter=';', dtype=None, skip_header=0, converters=me)
            for fr in face_feats_tmp:
                row = copy.deepcopy(fr)
                seq = list(row[0])
                fr[0] = int(seq[0] + seq_num)
            seq_num = np.array(face_feats_tmp[-1], copy=True)[0] + 1
            face_feats = np.concatenate((face_feats, face_feats_tmp), axis=0)
    return face_feats


def vector2angles(gaze_vector: np.ndarray):
    """
    Transforms a gaze vector into the angles yaw and elevation/pitch.
    :param gaze_vector: 3D unit gaze vector
    :return: 2D gaze angles
    """
    gaze_angles = np.empty((1, 2), dtype=np.float32)
    gaze_angles[0, 0] = np.arctan(-gaze_vector[0]/-gaze_vector[2])  # phi= arctan2(x/z)
    gaze_angles[0, 1] = np.arcsin(-gaze_vector[1])  # theta= arcsin(y)
    return gaze_angles


def angles2vector(angles):
    """
    Convert 2D angle (yaw and pitch) to 3D unit vector
    :param angles: list of 2D angles
    :return: computed 3D vectors
    """
    x = (-1.0) * K.sin(angles[:, 0]) * K.cos(angles[:, 1])
    y = (-1.0) * K.sin(angles[:, 1])
    z = (-1.0) * K.cos(angles[:, 0]) * K.cos(angles[:, 1])
    vec = K.transpose(K.concatenate([[x], [y], [z]], axis=0))
    return vec


def euclidean_distance(y_true, y_pred):
    """
    Compute average euclidean distance
    :param y_true: list of ground truth labels
    :param y_pred: list of predicted labels
    :return: euclidean distance
    """
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True)))


def angle_error(gt, pred):
    """
    Average angular error computed by cosine difference
    :param gt: list of ground truth label
    :param pred: list of predicted label
    :return: Average angular error
    """
    vec_gt = angles2vector(gt)
    vec_pred = angles2vector(pred)

    x = K.np.multiply(vec_gt[:, 0], vec_pred[:, 0])
    y = K.np.multiply(vec_gt[:, 1], vec_pred[:, 1])
    z = K.np.multiply(vec_gt[:, 2], vec_pred[:, 2])

    dif = K.np.sum([x, y, z], axis=0) / (tf.norm(vec_gt, axis=1) * tf.norm(vec_pred, axis=1))

    clipped_dif = K.clip(dif, np.float(-1.0), np.float(1.0))
    loss = (tf.acos(clipped_dif) * 180) / np.pi
    return K.mean(loss, axis=-1)


def numpy_angles2vector(angles):
    """
    Numpy version of angles2vector. Convert 2D angle (yaw and pitch) to 3D unit vector
    :param angles: list of 2D angles
    :return: computed 3D vectors
    """
    x = (-1.0)*np.sin(angles[:, 0]) * np.cos(angles[:, 1])
    y = (-1.0)*np.sin(angles[:, 1])
    z = (-1.0)*np.cos(angles[:, 0]) * np.cos(angles[:, 1])
    vec = np.transpose(np.concatenate([[x], [y], [z]], axis=0))
    return vec


def numpy_angle_error(gt, pred):
    """
    Numpy version of angle_error. Average angular error computed by cosine difference
    :param gt: list of ground truth label
    :param pred: list of predicted label
    :return: Average angular error
    """
    vec_gt = numpy_angles2vector(gt)
    vec_pred = numpy_angles2vector(pred)

    x = np.multiply(vec_gt[:, 0], vec_pred[:, 0])
    y = np.multiply(vec_gt[:, 1], vec_pred[:, 1])
    z = np.multiply(vec_gt[:, 2], vec_pred[:, 2])

    dif = np.sum([x, y, z], axis=0) / (np.linalg.norm(vec_gt, axis=1) * np.linalg.norm(vec_pred, axis=1))

    clipped_dif = np.clip(dif, np.float(-1.0), np.float(1.0))
    loss = (np.arccos(clipped_dif) * 180) / np.pi
    return np.mean(loss, axis=-1)


def get_normalized_data(mean_face, R, roi_size, calib):
    """
    Compute data needed to convert original space to normalized spacce (see paper for more details).
    :param mean_face: mean face coordinates
    :param R: rotation matrix (3x3)
    :param roi_size: size of final patch
    :param calib: intrinsic camera matrix
    :return: matrices needed to transform the data: 1) patch_conv: transformation matrix to convert 3D face to normalized space,
            2) patch_warp: transformation matrix to warp 2D image so that roll is removed, 3) patch_gaze: transformation matrix
            to convert 3D gaze vector to normalized space
    """
    focal_new = 960  # new focal distance
    distance_new = 600  # distance in normalized space from camera to reference point of the face (mean face)

    distance = np.linalg.norm(mean_face)
    z_scale = distance_new / distance

    cam_new = np.array([[focal_new, 0, roi_size[0] / 2], [0, focal_new, roi_size[1] / 2], [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]])

    R2 = R.transpose()
    hRx = R2[:, [0]]
    hRx = hRx.transpose()
    forward = mean_face / distance
    down = np.cross(forward, hRx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    warp_rot = np.row_stack((right, down, forward)).astype(np.float32)
    patch_conv = scale_mat.dot(warp_rot)
    patch_warp = cam_new.dot(patch_conv).dot(np.linalg.inv(calib['intrinsics']))
    patch_gaze = warp_rot

    return patch_conv, patch_warp, patch_gaze


def write_vector_to_file(file, vector):
    """
    Write 3x3 matrix in vector format into file (1,2,3;4,5,6;7,8,9)
    :param file: file object
    :param vector: matrix to write
    """
    for j in range(0, 9):
        if j is not 8:
            file.write(str(vector[j, 0]) + ',')
        else:
            file.write(str(vector[j, 0]) + ';')
