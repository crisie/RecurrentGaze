# !/usr/bin/env python
# title           :face_finetune.py
# description     :Model architecture for face only as input. Conv. blocks and FCs (all but last) are pre-trained
#                  from base model.
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
from keras.layers import Dense, Input, Dropout
from data_utils import input_type


class Facefinetune(MyModel):
    """
    Model architecture for face only as input, using specific base model for convolutional blocks and FC layers.
    Weights are already pre-trained using the base model, so finetuning is carried out in all layers except for last
    layer, which is trained from scratch.
    """
    def __init__(self):
        """
        Initialize class
        """
        super().__init__()

    def define(self, n_output: int=2, dropout: float=1., base_model=None):
        """
        Define model architecture for face_finetune
        :param n_output: number of network outputs
        :param dropout: dropout value
        :param base_model: Base model whose architecture and weights are used for all network except last FC layer.
        """

        image_input = Input(shape=base_model.input_size[input_type.FACE], name='input')
        weight_init = glorot_uniform(seed=3)

        # Load model with FC layers
        base = base_model.load_model(input_tensor=image_input, include_top=True)

        last_layer = base.get_layer('fc6/relu').output
        fc7 = base.get_layer('fc7')
        fc7r = base.get_layer('fc7/relu')
        x = last_layer

        if dropout < 1.:
            x = Dropout(dropout, seed=0, name='dp6')(x)
        x = fc7(x)
        x = fc7r(x)
        if dropout < 1.:
            x = Dropout(dropout, seed=1, name='dp7')(x)
        out = Dense(n_output, kernel_initializer=weight_init, name='fc8')(x)

        # Freeze first conv layers
        for layer in base.layers[:4]:
            layer.trainable = False

        self.model = Model(image_input, out)

        # Print model summary
        self.model.summary()


