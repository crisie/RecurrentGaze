# !/usr/bin/env python
# title           :eyes_fcscratch.py
# description     :Model architecture for eyes only as input.
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
from keras.engine import Model
from keras.initializers import glorot_uniform
from keras.layers import Flatten, Dense, Input, Dropout

from data_utils import input_type


class Eyesfcscratch(MyModel):
    """
    Model architecture for eyes only as input, using specific base model for convolutional blocks.
    FCs are trained from scratch.
    """
    def __init__(self):
        """
        Initialize class
        """
        super().__init__()

    def define(self, n_output: int=2, dropout: float=1., base_model=None):
        """
        Define model architecture for eyes_fcscratch
        :param n_output: number of network outputs
        :param dropout: dropout value
        :param base_model: Base model whose architecture and weights are used for convolutional blocks.
        """

        hidden_dim = 1536
        image_input = Input(shape=base_model.input_size[input_type.EYES], name='input')

        # Load base model without FC layers
        base = base_model.load_model(input_tensor=image_input, include_top=False)

        weight_init = glorot_uniform(seed=3)

        # Define architecture on top of base model
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc6')(x)
        if dropout < 1.:
            x = Dropout(dropout, seed=0, name='dp6')(x)
        out = Dense(n_output, kernel_initializer=weight_init, name='fc8')(x)

        # First for layers are not trained
        for layer in base.layers[:4]:
            layer.trainable = False

        self.model = Model([image_input], out)

        print(len(self.model.layers))
        print([n.name for n in self.model.layers])

        # Print model summary
        self.model.summary()
