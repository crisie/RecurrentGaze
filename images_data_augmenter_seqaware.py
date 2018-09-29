# !/usr/bin/env python
# title           :images_data_augmenter_seqaware.py
# description     :Script with online data augmenter class and related methods
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

from keras.preprocessing.image import flip_axis, apply_transform, \
    transform_matrix_offset_center

import numpy as np
import cv2 as cv


def apply_transform_matrix(self, img: np.ndarray, transform_matrix):
    """
    Apply transformation matrix to image img
    :param self: self
    :param img: image
    :param transform_matrix: transformation matrix
    :return: transformed image
    """
    h, w = img.shape[0], img.shape[1]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    img = apply_transform(img, transform_matrix, channel_axis=2, fill_mode=self.fill_mode, cval=self.cval)
    return img


def modify_illumination(images: list, ilrange: list, random_bright: float=None):
    """
    Convert images to HSV color space, modify Value channel according to random brightness value and convert back to
    RGB. If random_bright is None, the random brightness value is uniformly sampled from ilrange tuple, otherwise
    random_bright is directly used. This brightness value is multiplied to the original Value channel.
    :param images: list of images
    :param ilrange: illumination range (min, max) from which the brightness value is uniformly sampled if random_bright is None.
    :param random_bright: optional value specifying the brightness multiplier.
    :return: transformed images, random_bright value
    """
    if random_bright is None:
        random_bright = np.random.uniform(ilrange[0], ilrange[1])
    new_images = []
    for image in images:
        image1 = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1[:, :, 2] = np.clip(image1[:, :, 2], 0., 255.)
        image1 = cv.cvtColor(image1, cv.COLOR_HSV2RGB)
        new_images.append(image1)
    return new_images, random_bright


def add_gaussian_noise(images: list, var: list, random_var: float=None, gauss_noise: list=None):
    """
    Add gaussian noise to input images. If random_var and gauss_noise are given, use them to compute the final images.
    Otherwise, compute random_var and gauss_noise.
    :param images: list of images
    :param var: variance range from which the variance value is uniformly sampled if random_var is None.
    :param random_var: optional value specifying the variance multiplier.
    :param gauss_noise: optional value specifying the additive gaussian noise per image.
    :return: transformed image, random_var value, gauss_noise_out list
    """
    if random_var is None:
        random_var = np.random.uniform(var[0], var[1])
    mean = 0
    new_images = []
    gauss_noise_out = []
    for i,image in enumerate(images):
        row, col, c = image.shape
        if gauss_noise is None or \
                (gauss_noise is not None and row*col*c !=
                 gauss_noise[i].shape[0]*gauss_noise[i].shape[1] * gauss_noise[i].shape[2]):
            gauss = np.random.normal(mean, random_var * 127.5, (row, col, c))
        else:
            gauss = gauss_noise[i]
        gauss_noise_out.append(gauss)
        gauss = gauss.reshape(row, col, c)
        image1 = np.clip(image + gauss, 0., 255.)
        new_images.append(image1)
    return new_images, random_var, gauss_noise_out


class ImageDataAugmenter(object):
    """
    ImageDataAugmenter class.
    Prepared to apply the same augmentation to a list of images/data.
    Sequence-aware: the current augmentation state can be returned to the calling class, so that all frames of a
    sequence are augmented in the same way.
    """
    def __init__(self,
                 rotation_range: float=0.,
                 width_shift_range: float=0.,
                 height_shift_range: float=0.,
                 zoom_range=0., #float, tuple or list of two floats
                 fill_mode: str='nearest',
                 cval: float=0.,
                 horizontal_flip: bool=False,
                 vertical_flip: bool=False,
                 gaussian_noise_range=0.,
                 illumination_range=1., #float, tuple or list of two floats
                 rotation_prob: float=0.75,
                 shift_prob: float=0.75,
                 zoom_prob: float=0.75,
                 flip_prob: float=0.5,
                 illumination_prob: float=0.75,
                 gaussian_noise_prob: float=0.75):
        """
        Initialize ImageDataAugmenter class
        :param rotation_range: degrees of min/max rotation.
        :param width_shift_range: maximum number of pixels the image would be shifted horizontally.
        :param height_shift_range: maximum number of pixels the image would be shifted vertically.
        :param zoom_range: zoom range (float, tuple or list of two floats). Values are % multipliers (for instance, 0.98
                would zoom in the image such that 98% of the image is visible; 1.02 would zoom out). Check check_ranges.
        :param fill_mode: see keras.preprocessing.image.apply_transform
        :param cval: see keras.preprocessing.image.apply_transform
        :param horizontal_flip: True if horizontal flip activated
        :param vertical_flip: True if vertical flip activated
        :param gaussian_noise_range: min and max range of variance for gaussian noise
        :param illumination_range: illumination range (float, tuple or list of two floats). Values are multipliers to be
                applied to the VALUE channel of HSV color space.
        :param rotation_prob: probability of rotation happening.
        :param shift_prob: probability of shift happening.
        :param zoom_prob: probability of zooming happening.
        :param flip_prob: probability of flipping image happening.
        :param illumination_prob: probability of modifying illumination happening.
        :param gaussian_noise_prob: probability of adding gaussian noise happening.
        """

        self.fill_mode = fill_mode
        self.cval = cval
        self.rotation_prob = rotation_prob
        self.shift_prob = shift_prob
        self.zoom_prob = zoom_prob
        self.flip_prob = flip_prob
        self.illumination_prob = illumination_prob
        self.gaussian_noise_prob = gaussian_noise_prob

        self.dict = {'rotation_range': rotation_range,
                     'width_shift_range': width_shift_range, 'height_shift_range': height_shift_range,
                     'zoom_range': zoom_range,
                     'horizontal_flip': horizontal_flip, 'vertical_flip': vertical_flip,
                     'gaussian_noise_range': gaussian_noise_range, 'illumination_range': illumination_range}

        self.check_ranges('zoom_range', zoom_range)
        self.check_ranges('illumination_range', illumination_range)
        self.check_ranges('gaussian_noise_range', gaussian_noise_range)

        # Current supported changes that save state along frames of same sequence
        self.last_state = {'vertical_flip': False,
                           'horizontal_flip': False,
                           'rotation': 0,
                           'shift_x': 0,
                           'shift_y': 0,
                           'zoom': 1,
                           'illumination': 1,
                           'gauss_var': 0,
                           'gauss_noise': []}

    def reset_state(self):
        """
        Reset state to default.
        :return: default state
        """
        return dict(self.last_state)

    def augment(self, face_img: np.ndarray, nface_img: np.ndarray, leye_img: np.ndarray, reye_img: np.ndarray,
                lndmks: np.ndarray , y: np.ndarray=None, keep: bool=False, last_state: dict=None):
        """
        Main function, used to augment a series of images and data. If keep is True, use last_state to define
        the augmentations to apply; otherwise, compute new augmentations.
        :param face_img: Original (not-normalized) face image
        :param nface_img: Normalized face image
        :param leye_img: Normalized left eye image
        :param reye_img: Normalized right eye image
        :param lndmks: 3D Landmarks
        :param y: label
        :param keep: If true, use last_state to define the augmentations to apply; otherwise, compute new augmentations.
        :param last_state: Optional dictionary specifying the last augmentations applied.
        :return: augmented images and data, and current_state
        """

        # Check if we use existing augmentations or we have to compute new ones
        if keep:
            assert(set(last_state) == set(self.last_state))
        else:
            last_state = dict(self.last_state)

        # Vertical flip
        if self.dict['vertical_flip']:
            if (not keep and np.random.random() < self.flip_prob) \
                    or (keep and last_state['vertical_flip']):
                face_img = flip_axis(face_img, 0)
                nface_img = flip_axis(nface_img, 0)
                leye_img = flip_axis(leye_img, 0)
                reye_img = flip_axis(reye_img, 0)
                lndmks[:, 1] = - lndmks[:, 1]
                y[1] = -y[1]
                last_state['vertical_flip'] = True
            else:
                last_state['vertical_flip'] = False

        # Horizontal flip
        if self.dict['horizontal_flip']:
            if (not keep and np.random.random() < self.flip_prob) \
                    or (keep and last_state['horizontal_flip']):
                face_img = flip_axis(face_img, 1)
                nface_img = flip_axis(nface_img, 1)
                leye_img_t = flip_axis(leye_img, 1) # change left-right order
                reye_img_t = flip_axis(reye_img, 1)
                leye_img = reye_img_t
                reye_img = leye_img_t
                lndmks[:, 0] = - lndmks[:, 0]
                y[0] = -y[0]
                last_state['horizontal_flip'] = True
            else:
                last_state['horizontal_flip'] = False

        # Rotation
        if keep:
            theta = last_state['rotation']
        else:
            if self.dict['rotation_range'] > 0. and np.random.random() < self.rotation_prob:
                theta = np.pi / 180 * np.random.uniform(-self.dict['rotation_range'], self.dict['rotation_range'])
            else:
                theta = 0
        last_state['rotation'] = theta

        # Translation in y (in pixels)
        if keep:
            tx = last_state['shift_x']
        else:
            if self.dict['height_shift_range'] > 0. and np.random.random() < self.shift_prob:
                tx = np.random.uniform(-self.dict['height_shift_range'], self.dict['height_shift_range'])
            else:
                tx = 0
        last_state['shift_x'] = tx

        # Translation in x (in pixels)
        if keep:
            ty = last_state['shift_y']
        else:
            if self.dict['width_shift_range'] > 0. and np.random.random() < self.shift_prob:
                ty = np.random.uniform(-self.dict['width_shift_range'], self.dict['width_shift_range'])
            else:
                ty = 0
        last_state['shift_y'] = ty

        # Zoom
        if keep:
            z = last_state['zoom']
        else:
            if self.dict['zoom_range'][0] != 1 and self.dict['zoom_range'][1] != 1 and \
                    np.random.random() < self.zoom_prob:
                z = np.random.uniform(self.dict['zoom_range'][0], self.dict['zoom_range'][1])
            else:
                z = 1
        last_state['zoom'] = z

        # Apply composition of transformations
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if z != 1:
            zoom_matrix = np.array([[z, 0, 0],
                                    [0, z, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            face_img = apply_transform_matrix(self, face_img, transform_matrix)
            nface_img = apply_transform_matrix(self, nface_img, transform_matrix)
            leye_img = apply_transform_matrix(self, leye_img, transform_matrix)
            reye_img = apply_transform_matrix(self, reye_img, transform_matrix)

        # Illumination
        if self.dict['illumination_range'][0] > 0. and self.dict['illumination_range'][0] != 1 \
                and self.dict['illumination_range'][1] != 1:
            if not keep and np.random.random() < self.illumination_prob:
                [face_img, nface_img, leye_img, reye_img], last_state['illumination'] = \
                    modify_illumination([face_img, nface_img, leye_img, reye_img], self.dict['illumination_range'])
            elif keep and last_state['illumination'] != 1:
                [face_img, nface_img, leye_img, reye_img], last_state['illumination'] = \
                    modify_illumination([face_img, nface_img, leye_img, reye_img], self.dict['illumination_range'],
                                        last_state['illumination'])

        # Additive gaussian noise
        if self.dict['gaussian_noise_range'][1] > 0.:
            if not keep and np.random.random() < self.gaussian_noise_prob:
                [face_img, nface_img, leye_img, reye_img], last_state['gauss_var'], last_state['gauss_noise'] = \
                    add_gaussian_noise([face_img, nface_img, leye_img, reye_img], self.dict['gaussian_noise_range'])
            elif keep and last_state['gauss_noise'] != []:
                [face_img, nface_img, leye_img, reye_img], last_state['gauss_var'], last_state['gauss_noise'] = \
                    add_gaussian_noise([face_img, nface_img, leye_img, reye_img], self.dict['gaussian_noise_range'],
                                       last_state['gauss_var'], last_state['gauss_noise'])

        return [face_img, nface_img, leye_img, reye_img, lndmks], y, last_state

    def check_ranges(self, param_name, param_value):
        """
        Check if given input ranges are valid. Raise Error otherwise
        :param param_name: parameter name
        :param param_value: parameter values
        :return: checked ranges.
        """

        if np.isscalar(param_value):
            if param_value < 0. or param_value > 1.:
                raise ValueError((param_name, ' should be within range [0,1]'))
            else:
                if param_name == 'zoom_range' or param_name == 'illumination_range':
                    if param_value == 1:
                        self.dict[param_name] = [1, 1]
                    else:
                        self.dict[param_name] = [1 - param_value, 1 + param_value]
                else:
                    self.dict[param_name] = [0., param_value]

        elif len(param_value) == 2:

            if param_name != 'zoom_range' and param_name != 'illumination_range' \
                    and (param_value[0] < 0. or param_value[0] > 1. or param_value[1] < 0. or param_value[1] > 1.):
                raise ValueError((param_name, ' should be within range [0,1]'))
            else:
                self.dict[param_name] = [param_value[0], param_value[1]]
        else:
            raise ValueError((param_name, ' should be a float or '
                                          'a tuple or list of two floats. '
                                          'Received arg: '), param_value)

