# !/usr/bin/env python
# title           :experiment_utils.py
# description     : Utility functions for experiments
# author          :Cristina Palmero
# date            :30092018
# version         :1.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================
from keras.utils.data_utils import get_file as gf


class ModelWeights:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


NF4096_VGG16_PATH = ''
NF4096_VGG16_NAME = ''
NF4096_VGG16 = ModelWeights(NF4096_VGG16_NAME, NF4096_VGG16_PATH)

NF5632_VGG16_PATH = ''
NF5632_VGG16_NAME = ''
NF5632_VGG16 = ModelWeights(NF5632_VGG16_NAME, NF5632_VGG16_PATH)

NE1536_VGG16_PATH = ''
NE1536_VGG16_NAME = ''
NE1536_VGG16 = ModelWeights(NE1536_VGG16_NAME, NE1536_VGG16_PATH)

NFE5632_VGG16_PATH = ''
NFE5632_VGG16_NAME = ''
NFE5632_VGG16 = ModelWeights(NFE5632_VGG16_NAME, NFE5632_VGG16_PATH)

NFL4300_VGG16_PATH = ''
NFL4300_VGG16_NAME = ''
NFL4300_VGG16 = ModelWeights(NFL4300_VGG16_NAME, NFL4300_VGG16_PATH)
NFL4300_MIN_LNMDK = []
NFL4300_MAX_LNMDK = []

NFEL5836_VGG16_NAME = 'NFEL5836_FT_SM_NFEL5836_model.hdf5'
NFEL5836_VGG16_PATH = 'https://dl.dropbox.com/s/hws6zpfpueydz1s/NFEL5836_FT_SM_NFEL5836_model.hdf5?dl=1'
NFEL5836_VGG16 = ModelWeights(NFEL5836_VGG16_NAME, NFEL5836_VGG16_PATH)
NFEL5836_MIN_LNMDK = [-79.15255198, -63.78408521, -30.48750495]
NFEL5836_MAX_LNMDK = [79.15255198, 72.08113559, 56.73828705]

NFEL5836_2918_VGG16_NAME = 'NFEL5836_2918_FT_SM_NFEL5836_2918_model.hdf5'
NFEL5836_2918_VGG16_PATH = 'https://dl.dropbox.com/s/6vtm5ykr9fa46rh/NFEL5836_2918_FT_SM_NFEL5836_2918_model.hdf5?dl=1'
NFEL5836_2918_VGG16 = ModelWeights(NFEL5836_2918_VGG16_NAME, NFEL5836_2918_VGG16_PATH)
NFEL5836_2918_MIN_LNMDK = [-79.15255198, -63.78408521, -30.48750495]
NFEL5836_2918_MAX_LNMDK = [79.15255198, 72.08113559, 56.73828705]

NFEL5836GRU_VGG16_NAME = 'NFEL5836GRU_FT_SM_NFEL5836GRU_model.hdf5'
NFEL5836GRU_VGG16_PATH = 'https://dl.dropbox.com/s/qw3f8c7qj3y8g7h/NFEL5836GRU_FT_SM_NFEL5836GRU_model.hdf5?dl=1'
NFEL5836GRU_VGG16 = ModelWeights(NFEL5836GRU_VGG16_NAME, NFEL5836GRU_VGG16_PATH)
NFEL5836GRU_MIN_LNMDK = [-79.15255198, -63.78408521, -30.48750495]
NFEL5836GRU_MAX_LNMDK = [79.15255198, 72.08113559, 56.73828705]

RECURRENT_GAZE_DIR = 'models/recurrent_gaze'


def get_file(model_weights: ModelWeights):
    return gf(model_weights.name, model_weights.path, cache_subdir=RECURRENT_GAZE_DIR)


