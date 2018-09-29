# !/usr/bin/env python
# title           :two_stream.py
# description     :Model architecture for face and eyes as input. Metadata is optional, which can be concatenated to
#                  flattened face and eyes features. Individual block finetuned from base model, FCs from fusion trained
#                  from scratch
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from models import Model as MyModel
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Dropout, concatenate
from keras.initializers import glorot_uniform
from data_utils import input_type


class Twostream(MyModel):
    """
    Model architecture for face and eyes (and landmarks optionally) as input. Specific base model is used to fine tune
    individual blocks of face and eyes streams. All features are concatenated in fusion block, which is trained from
    scratch along with linear regression layer.
    """
    def __init__(self):
        """
        Initialize class
        """
        super().__init__()

    def define(self, n_output: int=2, dropout: float=1., hidden_dim_last: int=4096,
               base_model=None, use_metadata: bool=False):
        """

        :param n_output: number of network outputs
        :param dropout: dropout value
        :param hidden_dim_last: dimensions of last FC layer of fusion (not linear regression layer!)
        :param base_model: Base model whose architecture and weights are used for individual weights
        :param use_metadata: add metadata (landmarks) to model
        """

        hidden_dim_1 = 4096
        hidden_dim_2 = 1536
        hidden_dim_3 = hidden_dim_1 + hidden_dim_2
        hidden_dim_4 = hidden_dim_last
        if use_metadata:
            hidden_dim_3 = hidden_dim_3 + base_model.input_size[input_type.LANDMARKS][0]
            metadata_input = Input(shape=base_model.input_size[input_type.LANDMARKS],
                                   name='input-'+input_type.LANDMARKS.value)
            hidden_dim_4 = hidden_dim_last

        weight_init = glorot_uniform(seed=3)

        # --- INDIVIDUAL ---

        # --- Face ---
        face_input = Input(shape=base_model.input_size[input_type.FACE], name=input_type.FACE.value)
        face_model = base_model.load_model(input_tensor=face_input, include_top=True)
        for layer in face_model.layers:
            layer.name = layer.name + "-" + input_type.FACE.value
        face = face_model.get_layer('fc6/relu-'+input_type.FACE.value).output

        # --- Eyes ---
        eyes_input = Input(shape=base_model.input_size[input_type.EYES], name=input_type.EYES.value)
        eyes_model = base_model.load_model(input_tensor=eyes_input, include_top=False)
        for layer in eyes_model.layers:
            layer.name = layer.name + "-" + input_type.EYES.value
        eyes_last_layer = eyes_model.get_layer('pool5-'+input_type.EYES.value).output
        eyes = Flatten(name='flatten-'+ input_type.EYES.value)(eyes_last_layer)
        eyes = Dense(hidden_dim_2, activation='relu', kernel_initializer=weight_init,
                     name='fc6-'+ input_type.EYES.value)(eyes)
        # if (dropout < 1.):
        #    eyes = Dropout(dropout, seed=0, name='dp6-'+ input_type.EYES.value)(eyes)

        # Freeze first conv layers from face_and eyes model
        for layer in face_model.layers[:4]:
            layer.trainable = False
        for layer in eyes_model.layers[:4]:
            layer.trainable = False

        # --- FUSION ----
        if use_metadata:
            joint_fusion = concatenate([face, eyes, metadata_input])
        else:
            joint_fusion = concatenate([face, eyes])

        joint_fusion = Dense(hidden_dim_3, activation='relu', kernel_initializer=weight_init, name='fc7')(joint_fusion)
        if dropout < 1.:
            joint_fusion = Dropout(dropout, seed=0, name='dp7')(joint_fusion)
        joint_fusion = Dense(hidden_dim_4, activation='relu', kernel_initializer=weight_init, name='fc8')(joint_fusion)
        if dropout < 1.:
            joint_fusion = Dropout(dropout, seed=0, name='dp8')(joint_fusion)

        # --- LINEAR REGRESSION ---
        out = Dense(n_output, kernel_initializer=weight_init, name='out')(joint_fusion)

        if use_metadata:
            self.model = Model([face_input, eyes_input, metadata_input], out)
        else:
            self.model = Model([face_input, eyes_input], out)

        print(len(self.model.layers))
        print([n.name for n in self.model.layers])
        # Print model summary
        self.model.summary()

