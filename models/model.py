# !/usr/bin/env python
# title           :model.py
# description     :Base/Factory class for model architectures
# author          :Cristina Palmero
# date            :30092018
# version         :1.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

from adamaccum import Adam_accumulate
from keras.optimizers import adam
from data_utils import angle_error, euclidean_distance


class Model(object):
    """
    Model base class.
    """
    def __init__(self):
        """
        Initialize model with custom objects.
        """
        self.model = None
        self.custom_objects = {'euclidean_distance': euclidean_distance,
                               'eucl_dist_ang': angle_error}  # Called eucl_dist_ang for backward compatibility

    def define(self, *args):
        """
        Define model architecture.
        :param args: arguments
        """
        pass

    def compile(self,  lr: float=0.01, accum: bool=False):
        """
        Compiles model using specific loss (euclidean distance, optimizer (ADAM) and metrics (angular error)
        :param lr: learning rate
        :param accum: True if wait for several mini-batch to update.
        """
        if accum:
            opt = Adam_accumulate(lr=lr, accum_iters=8)
        else:
            opt = adam(lr)

        if self.model is None:
            raise ValueError('Only defined models can be compiled.')
        else:
            # Use mean euclidean distance as loss and angular error and mse as metric
            self.model.compile(loss=euclidean_distance,
                          optimizer=opt,
                              metrics=[angle_error, 'mse'])

    def fit_generator(self, generator, args):
        """
        Fits the model using the given generator (see keras fit_generator)
        :param generator: Generator that inherits from keras.utils.Sequence
        :param args: fit_generator arguments (see keras fit_generator)
        :return: history
        """
        hist = self.model.fit_generator(generator, **args)
        return hist

    def save(self, filename: str, weights_only: bool=False):
        """
        Save model or model weights.
        :param filename: Model/model weights file name.
        :param weights_only: True if only weights are saved (not model definition).
        """
        if weights_only:
            self.model.save_weights(filename)
        else:
            self.model.save(filename)

    def load_weights(self, filename: str):
        self.model.load_weights(filename)

    def predict(self, input: list):
        return self.model.predict(input)

