# !/usr/bin/env python
# title           :base_model.py
# description     :Factory/Base class for loading base models (i.e. VGGFace, Alexnet...)
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

from keras_vggface.vggface import VGGFace as OVGGFace
from data_utils import input_type


class BaseModel(object):
    """
    Class that contains all used base models and their characteristics. It contains methods to register new base models
    (as subclasses) and to gather them based on their name. This class implements defaults methods to be overridden by
    subclasses.

    Base model is an architecture (and associated weights) that are used to create other models on top of them
    (i.e. VGGFace, Alexnet, etc).
    """

    base_models = {}

    def __init__(self):
        self.input_size = dict({input_type.FACE: None, input_type.EYES: None, input_type.LANDMARKS: (68 * 3,)})

    @classmethod
    def register_subclass(cls, name: str):
        """
        Register base model to list of available experiments
        (there can be subclasses not registered that cannot be used)
        :param name: base model name
        """
        def decorator(subclass):
            cls.base_models[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_base_model(cls, name: str, *args, **kwargs):
        """
        Gather base model subclass by name
        :param name: base model name
        :param args: arguments
        :param kwargs:
        :return: Retrieved experiment class (if it exists)
        """
        name = name.upper()

        if name not in cls.base_models:
            raise ValueError('{} is not a valid base model'.format(name))

        return cls.base_models[name](*args, **kwargs)

    def load_model(self, input_tensor=None, weights=None, include_top=False):
        pass

    def mean_center(self, x):
        pass


@BaseModel.register_subclass('VGGFACE')
class VGGFace(BaseModel):
    """
    Class for VGGFace model, from by keras_vggface (https://github.com/rcmalli/keras-vggface)
    """
    def __init__(self):
        """
        Initialize class
        """
        super().__init__()
        self.input_size[input_type.FACE] = (224, 224, 3)
        self.input_size[input_type.EYES] = (48, 120, 3)

    def load_model(self, input_tensor=None, weights=None, include_top=False):
        """
        Loads model (see keras_vggface)
        :param input_tensor: input tensor dimensions
        :param weights: weights to load
        :param include_top: True to include top (FC) layers
        :return: VGGFace model
        """
        return OVGGFace(input_tensor=input_tensor, include_top=include_top)

    def mean_center(self, x):
        """
        Pre-processing mean image values provided by VGG group
        :param x: image with 4 dimensions (i.e. [batch_size, height, width, channels], the last one corresponding to the channel.
        """
        x[:, :, :, 0] -= 93.5940  # B
        x[:, :, :, 1] -= 104.7624  # G
        x[:, :, :, 2] -= 129.1863  # R

        return x

