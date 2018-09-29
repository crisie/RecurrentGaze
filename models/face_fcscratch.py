# !/usr/bin/env python
# title           :face_fcscratch.py
# description     :Model architecture for face only as input, with option of adding metadata (landmarks).
#                  Conv. blocks fine-tuned from base model, FCs trained from scratch.
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

import os
os.environ['KERAS_BACKEND']='tensorflow'
from models import Model as MyModel
from keras.initializers import glorot_uniform
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Dropout, concatenate

from data_utils import input_type


class Facefcscratch(MyModel):
    """
    Model architecture for face only as input, using specific base model for convolutional blocks.
    FCs trained from scratch. Metadata can be also used, which would be concatenated to flattened face features.
    """
    def __init__(self):
        """
        Initialize class
        """
        super().__init__()

    def define(self, n_output: int=2, dropout: float=1., hidden_dim: int=4096,
               base_model=None, use_metadata: bool=False):
        """
        Define model architecture for face_fcscratch. If use_metadata is True, landmarks are concatenated to
        flattened face features.
        :param n_output: number of network outputs
        :param dropout: dropout value
        :param hidden_dim: number of hidden dimensions of FC layers
        :param base_model: Base model whose architecture and weights are used for convolutional blocks.
        :param use_metadata: add metadata (landmarks) to model
        """

        image_input = Input(shape=base_model.input_size[input_type.FACE], name='input-'+input_type.FACE.value)
        weight_init = glorot_uniform(seed=3)

        # Load model with FC layers
        base = base_model.load_model(input_tensor=image_input, include_top=False)

        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)

        if use_metadata:
            metadata_input = Input(shape=base_model.input_size[input_type.LANDMARKS],
                                   name='input-'+input_type.LANDMARKS.value)
            x = concatenate([x, metadata_input])

        x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init,  name='fc6')(x)
        if dropout < 1.:
            x = Dropout(dropout, seed=0, name='dp6')(x)
        x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc7')(x)
        if dropout < 1.:
            x = Dropout(dropout, seed=1, name='dp7')(x)
        out = Dense(n_output, kernel_initializer=weight_init, name='fc8')(x)

        # Freeze first conv layers
        for layer in base.layers[:4]:
            layer.trainable = False

        if use_metadata:
            self.model = Model([image_input, metadata_input], out)
        else:
            self.model = Model(image_input, out)

        # Print model summary
        self.model.summary()
