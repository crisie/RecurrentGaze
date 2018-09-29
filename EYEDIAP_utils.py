# !/usr/bin/env python
# title           :EYEDIAP_utils.py
# description     :Script with utility methods to read EYEDIAP files. Adapted from scripts provided by EYEDIAP.
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

import numpy as np
import math

import cv2 as cv

import tensorflow as tf
from keras import backend as K
from itertools import chain


def read_screen_track_file(screen_track_file):
    """
    Read the ground truth values, i.e. the 3D position of the screen target
    """
    screen_track_vals = np.loadtxt(screen_track_file, skiprows=1, delimiter=';')[:, -3:]
    return screen_track_vals


def read_ball_track_file(ball_track_file):
    """
    Read the ground truth values, i.e. the 3D position of the floating target
    """
    ball_track_vals = np.loadtxt(ball_track_file, skiprows=1, delimiter=';')[:, -3:]
    return ball_track_vals


def read_head_track_file(head_track_file):
    """
    Read the head pose parameters: the frame-by-frame rotation and translation
    """
    head_track_vals = np.loadtxt(head_track_file, skiprows=1, delimiter=';')[:, 1:]
    R = head_track_vals[:, :9].reshape(-1, 3, 3)
    T = head_track_vals[:, 9:12]
    return R, T


def readCalibrationFile(calibration_file):
    """
    Reads the calibration parameters
    """
    cal = {}
    fh = open(calibration_file, 'r')
    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(';')]
    cal['size'] = cal['size'][0], cal['size'][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['intrinsics'] = np.array(vals).reshape(3, 3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['R'] = np.array(vals).reshape(3, 3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['T'] = np.array(vals).reshape(3, 1)
    fh.close()
    return cal






