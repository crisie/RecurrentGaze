# !/usr/bin/env python
# title           :two_stream_rnn.py
# description     :Model architecture for temporal model: individual streams are frozen, fusion block is fine-tuned and
#                  temporal module is trained from scratch along with last linear regression layer.
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import gc
from models import Model as MyModel
from keras.engine import Model
from keras.layers import Dense, Input, Dropout, TimeDistributed, concatenate, CuDNNLSTM, CuDNNGRU, Lambda, LSTM, GRU
from keras.initializers import glorot_uniform
from keras.models import load_model
from keras import backend as K
from data_utils import input_type


class Twostreamrnn(MyModel):
    """"
    Model architecture for temporal model: individual streams are frozen, fusion block is fine-tuned and
    temporal module is trained from scratch along with last linear regression layer.
    """
    def __init__(self):
        """
        Initializes class
        """
        super().__init__()

    def define(self, features_model: str, n_output: int=2, dropout: float=1., n_units: int=256,
               lstm_layers: int=1, rec_type: str="gru", base_model=None):
        """
        Define two_stream_rnn model architecture for trained two_stream network + rnn module from scratch
        :param features_model: already trained static model
        :param n_output: number of network outputs
        :param dropout: dropout value
        :param n_units: number of RNN units
        :param lstm_layers: number of RNN layers
        :param rec_type: type of recurrent layer (GRU or LSTM)
        :param base_model: Base model whose architecture and weights are used for individual weights. In this case
                base model is just used to know the data input sizes, as base_model is already used within
                "features_model"
        """

        # Initialize weights
        weight_init = glorot_uniform(seed=3)

        # Load pre-trained model
        print("Loading model... ", features_model)
        cnn_model = features_model#load_model(features_model, custom_objects=self.custom_objects)

        face_input = base_model.input_size[input_type.FACE]
        eyes_input = base_model.input_size[input_type.EYES]
        landmarks_input = base_model.input_size[input_type.LANDMARKS]
        face_input_seq = Input(shape=(None, face_input[0], face_input[1], face_input[2]),
                               name='seq_input_'+input_type.FACE.value)
        eyes_input_seq = Input(shape=(None, eyes_input[0], eyes_input[1], eyes_input[2]),
                               name='seq_input_'+input_type.EYES.value)
        landmarks_input_seq = Input(shape=(None, landmarks_input[0]),
                                    name='seq_'+input_type.LANDMARKS.value)

        model_input = [face_input_seq, eyes_input_seq, landmarks_input_seq]

        # --- INDIVIDUAL and FUSION ---
        face_counter = 0
        eyes_counter = 0
        for layer in cnn_model.layers:
            layer.trainable = False

            if input_type.FACE.value in layer.name and "dp" not in layer.name:
                if face_counter == 0:
                    fx = face_input_seq
                    face_counter += 1
                else:
                    # The layer is not trained when using lambda.
                    # fx = TimeDistributed(layer)(fx)
                    fx = TimeDistributed(Lambda(lambda x: layer(x)))(fx)

            elif input_type.EYES.value in layer.name and "dp" not in layer.name:
                if eyes_counter == 0:
                    ex = eyes_input_seq
                    eyes_counter += 1
                else:
                    # ex = TimeDistributed(layer)(ex)
                    ex = TimeDistributed(Lambda(lambda x: layer(x)))(ex)

            elif "concatenate" in layer.name:
                x = concatenate([fx, ex, landmarks_input_seq])

            else:
                if "out" not in layer.name and "input" not in layer.name:
                    layer.trainable = True
                    x = TimeDistributed(layer)(x)
                    # x = TimeDistributed(Lambda(lambda x: layer(x)))(x)

        # --- TEMPORAL ---
        for i in range(lstm_layers):
            namei = "rec_l" + str(i)
            named = "rec_dp" + str(i)

            if lstm_layers > 1 and i == 0:
                return_sequences = True
            else:
                return_sequences = False

            # if dropout < 1.:
            #    x = Dropout(dropout, seed=0, name=named)(x)

            num_units = int(n_units / (int(i) + 1))
            print("Num units lstm/gru: ", num_units)
            if rec_type == "lstm":
                if len(K.tensorflow_backend._get_available_gpus()) > 0:
                    x = CuDNNLSTM(num_units, name=namei, return_sequences=return_sequences)(x)
                else:
                    x = LSTM(num_units, name=namei, return_sequences=return_sequences)(x)
            else:
                if len(K.tensorflow_backend._get_available_gpus()) > 0:
                    x = CuDNNGRU(num_units, name=namei, return_sequences=return_sequences)(x)
                else:
                    x = GRU(num_units, name=namei, return_sequences=return_sequences)(x)

            if i < (lstm_layers - 1) and dropout < 1.:
                x = Dropout(dropout, seed=0, name=named)(x)

        # --- LINEAR REGRESSION ---
        out = Dense(n_output, kernel_initializer=weight_init, name='out')(x)

        self.model = Model(inputs=model_input, outputs=out)

        self.model.summary()

        print(len(self.model.layers))
        print([n.name for n in self.model.layers])

        del cnn_model
        gc.collect()
