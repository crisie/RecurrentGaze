# !/usr/bin/env python
# title           :experiment_helper.py
# description     : Experiment helper class along with subclasses for each experiment, with their respective
#                   characteristics and methods.
# author          :Cristina Palmero
# date            :30092018
# version         :1.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

import os
os.environ['KERAS_BACKEND']='tensorflow'
from data_utils import *
from base_model import BaseModel
from models import get_model
from data_generator import DataGenerator
from images_data_augmenter_seqaware import ImageDataAugmenter
import experiment_utils as exp_utils


class ExperimentHelper(object):
    """
    Class that contains all experiments and their specific characteristics. It contains methods to register new 
    experiments (as subclasses) and to gather them based on their name. Characteristics and methods that are shared by
    all experiments are defined in the parent class (this class) as DEFAULT. The Default mode is Face only networks,
    without converting data to normalized space. Each experiment can have their own specific methods and characteristics,
     which are overridden in their respective subclasses.

    This class and subclasses also contain methods to be passed to DataGenerator. See DataGenerator for more info.
    
    """
    experiments = {}

    def __init__(self,
                 name: str=None,
                 description: str=None,
                 weights: str=None,
                 train: bool=False,
                 base_model: BaseModel=None,
                 model=None,
                 fc_dimensions: int=4096,
                 label_pos: int=-1,
                 look_back: int=1,
                 n_output: int=2,
                 recurrent_type: str="lstm",
                 num_recurrent_layers: int=1,
                 num_recurrent_units: int=128,
                 train_data_generator: DataGenerator=None,
                 val_data_generator: DataGenerator=None):
        """
        Initialize ExperimentHelper class.
        :param name: name of experiment
        :param description: description of experiment
        :param weights: weights of model (in case it has been already trained)
        :param train: True if training is activated
        :param base_model: base model used (for instance, VGGFace)
        :param model: model architecture type
        :param fc_dimensions: dimensions of FC layers
        :param label_pos: label position
        :param look_back: sequence length
        :param n_output: number of outputs of model
        :param recurrent_type: type of recurrent network (gru or lstm)
        :param num_recurrent_layers: number of recurrent layers
        :param num_recurrent_units: number of recurrent units
        :param train_data_generator: DataGenerator for training
        :param val_data_generator: DataGenerator for validation/test (in case there is any)
        """
        self.name = name
        self.description = description
        self.weights = weights
        self.train = train
        self.base_model = base_model
        self.model = model
        self.fc_dimensions = fc_dimensions
        self.label_pos = label_pos
        self.n_output = n_output
        # --- temporal options ---
        self.look_back = look_back
        self.recurrent_type = recurrent_type
        self.num_recurrent_layers = num_recurrent_layers
        self.num_recurrent_units = num_recurrent_units
        # --- other ---
        self.train_data_generator = train_data_generator
        self.val_data_generator = val_data_generator


    @classmethod
    def register_subclass(cls, name: str):
        """
        Register experiment to list of available experiments
        (there can be subclasses not registered that cannot be used)
        :param name: experiment name
        :return:
        """
        def decorator(subclass):
            cls.experiments[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_experiment(cls, name: str, *args, **kwargs):
        """
        Gather experiment subclass by name
        :param name: experiment name
        :param args: arguments
        :param kwargs:
        :return: Retrieved experiment class (if it exists)
        """
        name = name.upper()

        if name not in cls.experiments:
            raise ValueError('{} is not a valid experiment'.format(name))

        return cls.experiments[name](*args, **kwargs)

    def get_name(self):
        """
        Get class name
        :return: class name
        """
        return self.__class__.__name__

    def init_data_gen_train(self,
                            data: DataTuple,
                            batch_size: int=64,
                            augmenter: ImageDataAugmenter=None,
                            shuffle: bool=True,
                            debug: bool=False):
        """
        Initialize data generator for training stage
        :param data: DataTuple including x, y and feats
        :param batch_size: batch size
        :param augmenter: augmenter object (ImageDataAugmenter)
        :param shuffle: True to shuffle input data
        :param debug: True if debug mode is activated to show augmentation and normalization image results
        """
        self.train_data_generator = self.init_data_gen(data, batch_size, augmenter, shuffle, debug)

    def init_data_gen_val(self,
                          data: DataTuple,
                          batch_size: int=64,
                          augmenter: ImageDataAugmenter=None,
                          shuffle: bool=False,
                          debug: bool = False):
        """
        Initialize data generator for validation/test stage
        :param data: DataTuple including x, y and feats
        :param batch_size: batch size
        :param augmenter: augmenter object (ImageDataAugmenter)
        :param shuffle: True to shuffle input data
        :param debug: True if debug mode is activated to show augmentation and normalization image results
        """
        self.val_data_generator = self.init_data_gen(data, batch_size, augmenter, shuffle, debug)

    def init_data_gen(self,
                      data: DataTuple,
                      batch_size: int=64,
                      augmenter: ImageDataAugmenter=None,
                      shuffle: bool=False,
                      debug: bool=False):
        """
        Initialize new data generator object with custom methods that depend on the experiment used. The code assumes
        that the "default" mode is to convert to normalized space the input data, so "norm" methods are used as input
        for the data generator here. If that's not the case, this method is overridden in respective experiments.
        :param data: DataTuple including x, y and feats
        :param batch_size: batch size
        :param augmenter: augmenter object (ImageDataAugmenter)
        :param shuffle: True to shuffle input data
        :param debug: True if debug mode is activated to show augmentation and normalization image results
        """
        datagen = DataGenerator(data.x, data.y, data.feats, batch_size, augmenter, shuffle, debug)
        datagen.set_methods(self.arrange_arrays, self.arrange_label_array, self.look_back_range,
                            self.get_preprocess_info, self.load_image, self.preprocess_input_data_norm,
                            self.preprocess_input_label_norm, self.resize_input_data, self.prepare_tensor_dims,
                            self.normalize_input_data, self.arrange_final_data, self.decide_input_label)
        return datagen

    def arrange_arrays(self, batch_size: int):
        """
        Initialize data arrays for generator according to batch size and type of data.
        In this case FACE only (default).
        :param batch_size: batch size
        :return: empty data arrays
        """
        return arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.FACE]))

    def arrange_label_array(self, batch_size: int):
        """
        Initialize data arrays for generator according to batch size and number of output labels
        :param batch_size: batch size
        :return: empty label arrays
        """
        return arrange_array(arrange_array_size(batch_size, [self.n_output]))

    def look_back_range(self):
        """
        Returns the range of frames between 0 and the sequence length (i.e. for a 4-frame sequence, the method
        would return range[0,1,2,3]
        :return: look back range
        """
        return range(0, self.look_back)

    def load_image(self, img: str):
        """
        Reads image from directory
        :param img: Image directory
        :return: Image object (ndarray)
        """
        return load_image(img)

    def prepare_data(self,
                     train_data: DataTuple,
                     validation_data: DataTuple,
                     args: list,
                     train: bool=True): #Used
        """
        Perform all necessary actions to data before fitting model.
        In this case, add extra dimension to data after first dimension.
        :param train_data: training data DataTuple
        :param validation_data: validation data DataTuple
        :param train: True if training, False if validating/testing
        :param args: list of possible arguments
        :return: modified (or not) training and validation data
        """
        if train:
            train_data = add_dimension(train_data)
        if validation_data is not None:
            validation_data = add_dimension(validation_data)
        return train_data, validation_data, args

    def prepare_metadata(self, train_data: DataTuple, validation_data: DataTuple, args, train=True):
        """
        Perform necessary actions to metadata before fitting model.
        In case of training, minimum and maximum landmarks of training set are computed so as to normalize all landmarks
        later. Otherwise, previously computed min and max landmarks are added to set.
        :param train_data: DataTuple containing training data
        :param validation_data: DataTuple containing validation data (if there is none, it's None).
        :param args: possible variables needed to prepare metadata
        :param train: True if training is activated
        :return: training (and validation) DataTuples now containing min and max training landmarks, input arguments
                plus computed min and max landmarks
        """
        if train:
            min_lndmk, max_lndmk = compute_min_max_landmarks_fold(train_data.feats, True)
            train_data = train_data._replace(
                feats=add_minxmax_landmarks_values(train_data.feats, min_lndmk, max_lndmk))
            args['min_landmark'] = min_lndmk
            args['max_landmark'] = max_lndmk
        else:
            args['min_landmark'] = self.min_lndmk
            args['max_landmark'] = self.max_lndmk

        if validation_data is not None:
            validation_data = validation_data._replace(
                feats=add_minxmax_landmarks_values(validation_data.feats, args['min_landmark'], args['max_landmark']))
        return train_data, validation_data, args

    def get_preprocess_info_ind(self, index: dict, feature: np.ndarray):
        """
        Get necessary information to preprocess the frame (from face_features, see data_utils.read_face_features_file)
        and store in dict.
        :param index: dict to contain preprocessing info for frame
        :param feature: original preprocessing info saved in maps
        :return: index
        """
        index["face_conv"] = get_face_conv(feature)
        index["gaze_conv"] = get_gaze_conv(feature)
        index["face_roi_size"] = get_face_roi_size(feature)
        index["eyes_roi_size"] = get_eyes_roi_size(feature)
        index["face_warp"] = get_face_warp(feature)
        index["leye_warp"] = get_leye_warp(feature)
        index["reye_warp"] = get_reye_warp(feature)
        index["bb"] = get_bb(feature)
        index["landmarks"] = get_landmarks(feature)
        index["min_landmark"], index["max_landmark"] = get_min_max_landmarks(feature)
        return index

    def get_preprocess_info(self, features: np.ndarray):
        """
        Get necessary information from face_features to preprocess (a series of) frames and stores it in list of dicts.
        :param features: original preprocessing info saved in maps
        :return: list of dicts containing preprocessing information.
        """
        features_array = copy_face_features(features)

        info = [dict() for x in range(len(features_array))]

        for i, f in enumerate(features_array):
            info[i] = self.get_preprocess_info_ind(info[i], f)
        return info

    def preprocess_input_imgs(self, img: np.ndarray, info: dict):
        """
        Preprocess input images: crops "original image" and applies warping to normalize rest of images to the
        "normalized space". Note that even though the chosen experiment may not contain all images, they are all
        processed anyway (to ensure reproducibility due to random number generation).
        :param img: array of images
        :param info: preprocessing information for this specific frame.
        :return: preprocessed and normalized images
        """
        return [preprocess_oface(img, info["bb"]),
                warp_image(img, info["face_warp"], info["face_roi_size"]),
                warp_image(img, info["leye_warp"], info["eyes_roi_size"]),
                warp_image(img, info["reye_warp"], info["eyes_roi_size"])]

    def preprocess_input_metadata(self, info: dict):
        """
        Preprocess input metadata (landmarks) by substracting the mean face coordinate.
        :param info: preprocessing information for this specific frame.
        :return: mean centered landmarks
        """
        mean_face = np.mean(info["landmarks"], axis=0)
        return info["landmarks"] - mean_face

    def preprocess_input_metadata_norm(self, info: dict):
        """
        If landmarks have to be converted to normalized space, normalize them first and then mean center them
        :param info: preprocessing information for this specific frame.
        :return: normalized, mean centered landmarks
        """
        mean_face = np.mean(info["landmarks"], axis=0)
        landmarks, mean_face = transform_landmarks(info["landmarks"], info["face_conv"], mean_face)
        return landmarks - mean_face

    def preprocess_input_data(self, imgs: list, info: dict):
        """
        Preprocess all input data (images and metadata)
        :param imgs: list of input images
        :param info: preprocessing information for this specific frame.
        :return: preprocessed/normalized data
        """
        data = self.preprocess_input_imgs(imgs, info)
        data.append(self.preprocess_input_metadata(info))
        return data

    def preprocess_input_data_norm(self, imgs: list, info: dict):
        """
        Preprocess all input data (images and metadata), taking into account that metadata has to be converted to
        normalized space
        :param imgs: list of input images
        :param info: preprocessing information for this specific frame.
        :return: preprocessed/normalized data
        """
        data = self.preprocess_input_imgs(imgs, info)
        data.append(self.preprocess_input_metadata_norm(info))
        return data

    def preprocess_input_label(self, label: np.ndarray, info: dict=None):
        """
        Preprocess input label: convert 3D unit gaze vector to 2D angles (yaw and pitch).
        :param label: 3D unit gaze vector as input label (y)
        :param info: preprocessing information for this specific frame (not used here, included for compatibility).
        :return: preprocessed input label
        """
        preproc_label = vector2angles(label)
        return preproc_label

    def preprocess_input_label_norm(self, label: np.ndarray, info: dict):
        """
        Preprocess input label: convert it to normalized space and then convert it to 2D angles.
        :param label: 3D unit gaze vector as input label (y)
        :param info: preprocessing information for this specific frame.
        :return: normalized and preprocessed input label
        """
        norm_label = normalize_gaze(label, info["gaze_conv"])
        return self.preprocess_input_label(norm_label, info)

    def resize_input_data(self, input_data: list, info: dict):
        """
        Resize input images to size that is compatible with model architecture. In this case metadata is not resized.
        :param input_data: list of input data (images and metadata).
        :param info: preprocessing information for this specific frame.
        :return: resized images
        """
        return [resize_oface(input_data[0], info["bb"], self.base_model.input_size[input_type.FACE]),
                resize_nface(input_data[1], self.base_model.input_size[input_type.FACE]),
                resize_eyes(input_data[2], input_data[3], self.base_model.input_size[input_type.EYES]),
                input_data[4]]

    def prepare_tensor_dims(self, input_data: list):
        """
        Modify tensor dimensions so that they are compatible with model architecture and Keras. All but the last
        input_data elements are modified, since the code assumes that the last element is the metadata.
        :param input_data: list of input data (images and metadata).
        :return: input data with ready-to-be-used dimensions.
        """
        prepared_input_data = []
        for i in range(len(input_data)-1):
            prepared_input_data.append(prepare_tensor_dims(image.img_to_array(input_data[i])))
        prepared_input_data.append(input_data[-1])
        return prepared_input_data

    def normalize_input_data(self, input_data: list, info: dict):
        """
        Performs mean centering in each of the input images (not metadata - the code assumes that the last element of
        input_data is the metadata). Mean centering values are given by the base model.
        :param input_data: list of input data (images and metadata).
        :param info: preprocessing information for this specific frame.
        :return: mean centered data
        """
        normalized_input_data = []
        for i in range(len(input_data) - 1):
            normalized_input_data.append(self.base_model.mean_center(input_data[i]))
        return normalized_input_data

    def normalize_metadata(self, landmarks: np.ndarray, info: dict):
        """
        Normalize/standardize metadata. In this case the code performs a min/max normalization.
        :param landmarks: list of 3D landmarks
        :param info: preprocessing information for this specific frame.
        :return: normalized/standardized metadata
        """
        lndmks = min_max_normalization(landmarks, info["min_landmark"], info["max_landmark"], 20)
        lndmks = lndmks.reshape((3 * 68,))
        return lndmks

    def arrange_final_data(self, input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Select data from input_data that will be fed to the network (note that all data is processed but only some of it
        is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[batch_pos] = input_data[0]
        return batch_data

    def decide_input_label(self, input_labels: list, info: dict=None):
        """
        Decide final label for this frame/sequence.
        :param input_labels: list of input labels (there are as many labels as frames in sequence).
        :param info: preprocessing information for this specific frame.
        :return: value to be used as label in training.
        """
        return input_labels[-1]

    def compile_model(self, learning_rate: float):
        """
        Compiles experiment model.
        :param learning_rate: learning rate
        """
        self.model.compile(learning_rate)

    def define_model(self, dropout: float):
        """
        Defines model architecture.
        :param dropout: dropout value
        """
        pass

    def load_model(self):
        """
        Load trained model and weights
        """
        self.define_model(1)
        self.model.load_weights(exp_utils.get_file(self.weights))


@ExperimentHelper.register_subclass('OF4096')
class OF4096(ExperimentHelper):
    """
    OF4096 experiment
    """
    def __init__(self):
        """
        Initialize exp.
        """
        super().__init__()
        self.name = "OF4096"
        self.description = "Original face, fc 4096D, finetuned VGGFace except last fc"
        self.weights = exp_utils.OF4096_VGG16
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("face_finetune")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model)

    def get_preprocess_info(self, features: np.ndarray):
        """
        Overridden. Get necessary information from face_features to preprocess (a series of) frames and stores it in
        list of dicts.
        :param features: original preprocessing info saved in maps
        :return: list of dicts containing preprocessing information.
        """
        info = super().get_preprocess_info(features)
        for i in range(len(info)):
            info[i]["gaze_conv"] = None
            info[i]["face_conv"] = None
        return info

    def init_data_gen(self, data: DataTuple, batch_size: int=64, augmenter: ImageDataAugmenter=None,
                      shuffle: bool=False, debug: bool=False):
        """
        Overridden. Initialize new data generator. Overrides (and calls) super() method so that "norm" methods are not
        used.
        :param data: DataTuple including x, y and feats
        :param batch_size: batch size
        :param augmenter: augmenter object (ImageDataAugmenter)
        :param shuffle: True to shuffle input data
        :param debug: True if debug mode is activated to show augmentation and normalization image results
        """
        datagen = super().init_data_gen(data, batch_size, augmenter, shuffle, debug)
        datagen.set_methods(preprocess_input_data=self.preprocess_input_data,
                            preprocess_input_label=self.preprocess_input_label)
        return datagen


@ExperimentHelper.register_subclass('NF4096')
class NF4096(ExperimentHelper):
    """
    NF4096 experiment
    """
    def __init__(self):
        """
        Initializes exp.
        """
        super().__init__()
        self.name = "NF4096"
        self.description = "Normalized face, fc 4096D, fcs trained from scratch"
        self.weights = exp_utils.NF4096_VGG16
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("face_fcscratch")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model)

    def arrange_final_data(self, input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but
        only some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[batch_pos] = input_data[1]
        return batch_data


@ExperimentHelper.register_subclass('NF5632')
class NF5632(ExperimentHelper):
    """
    NF5632 experiment
    """
    def __init__(self):
        """
        Initializes exp.
        """
        super().__init__()
        self.name = "NF5632"
        self.description = "Normalized face, fc 5632D, fcs trained from scratch"
        self.fc_dimensions = 5632
        self.weights = exp_utils.NF5632_VGG16
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("face_fcscratch")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model, hidden_dim=self.fc_dimensions)

    def arrange_final_data(self, input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but
        only some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[batch_pos] = input_data[1]
        return batch_data


@ExperimentHelper.register_subclass('NE1536')
class NE1536(ExperimentHelper):
    """
    NE1536 experiment
    """
    def __init__(self):
        """
        Initializes exp.
        """
        super().__init__()
        self.name = "NE1536"
        self.description = "Normalized eyes, fc 1536D, fcs trained from scratch"
        self.weights = exp_utils.NE1536_VGG16
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("eyes_fcscratch")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model)

    def arrange_arrays(self, batch_size: int):
        """
        Overridden. Initialize data arrays for generator according to batch size and type of data.
        In this case EYES only (default).
        :param batch_size: batch size
        :return: empty data arrays with correct input size.
        """
        return [arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.EYES]))]

    def arrange_final_data(self,  input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but
         only some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[batch_pos] = input_data[2]
        return batch_data


@ExperimentHelper.register_subclass('NFL4300')
class NFL4300(ExperimentHelper):
    """
    NFL4300 experiment
    """
    def __init__(self):
        """Initializes experiment"""
        super().__init__()
        self.name = "NFL4300"
        self.description = "Normalized face and landmarks, fc 4300D"
        self.fc_dimensions = 4300
        self.weights = exp_utils.NFL4300_VGG16
        self.min_lndmk = exp_utils.NFL4300_MIN_LNMDK
        self.max_lndmk = exp_utils.NFL4300_MAX_LNMDK
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("face_fcscratch")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model, hidden_dim=self.fc_dimensions,
                          use_metadata=True)

    def arrange_arrays(self, batch_size: int):
        """
        Overridden. Initialize data arrays for generator according to batch size and type of data.
        In this case FACE and LANDMARKS.
        :param batch_size: batch size
        :return: empty data arrays with correct input size.
        """
        return [arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.FACE])),
                arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.LANDMARKS]))]

    def arrange_final_data(self,  input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but only
        some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[0][batch_pos] = input_data[1]
        batch_data[1][batch_pos] = input_data[3]
        return batch_data

    def prepare_data(self, train_data: DataTuple, validation_data: DataTuple, args: list, train: bool=True): #Used
        """
        Perform all necessary actions to data before fitting model.
        In this case, metadata and regular data are processed.
        :param train_data: training data DataTuple
        :param validation_data: validation data DataTuple
        :param train: True if training, False if validating/testing
        :param args: list of possible arguments
        :return: modified (or not) training and validation data,
                and list of arguments containing max look back and max/min training landmarks
        """
        train_data, validation_data, args = self.prepare_metadata(train_data, validation_data, args, train)
        return super().prepare_data(train_data, validation_data, args, train)

    def normalize_input_data(self, input_data, info):
        """
        Overridden. Normalizes/standardizes input data (images and metadata) according to used functions.
        :param input_data: list of input data (images and metadata).
        :param info: preprocessing information for this specific frame.
        :return: normalized/standardized data
        """
        return super().normalize_input_data(input_data, info) + [self.normalize_metadata(input_data[3], info)]


@ExperimentHelper.register_subclass('NFE5632')
class NFE5632(ExperimentHelper):
    """
    NFE5632 experiment
    """
    def __init__(self):
        """Initializes exp."""
        super().__init__()
        self.name = "NFE5632"
        self.description = "Normalized face and eyes, two-stream network, fc 5632D"
        self.fc_dimensions = 5632
        self.weights = exp_utils.NFE5632_VGG16
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("two_stream")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model, hidden_dim_last=self.fc_dimensions)

    def arrange_arrays(self, batch_size: int):
        """
        Overridden. Initialize data arrays for generator according to batch size and type of data.
        In this case FACE and EYES.
        :param batch_size: batch size
        :return: empty data arrays with correct input size.
        """
        return [arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.FACE])),
                arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.EYES]))]

    def arrange_final_data(self,  input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but
        only some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[0][batch_pos] = input_data[1]
        batch_data[1][batch_pos] = input_data[2]
        return batch_data


@ExperimentHelper.register_subclass('NFEL5836')
class NFEL5836(ExperimentHelper):
    """
    NFEL5836 experiment
    """
    def __init__(self):
        """Initializes exp."""
        super().__init__()
        self.name = "NFEL5836"
        self.description = "Normalized face, eyes and landmarks, two-stream + metadata network, fc 5836D"
        self.fc_dimensions = 2918
        self.weights = exp_utils.NFEL5836_VGG16
        self.min_lndmk = exp_utils.NFEL5836_MIN_LNMDK
        self.max_lndmk = exp_utils.NFEL5836_MAX_LNMDK
        self.base_model = BaseModel.get_base_model("VGGFace")
        self.model = get_model("two_stream")
        print(self.name)
        print(self.description)

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """
        self.model.define(dropout=dropout, base_model=self.base_model, hidden_dim_last=self.fc_dimensions,
                          use_metadata=True)

    def arrange_arrays(self, batch_size: int):
        """
        Overridden. Initialize data arrays for generator according to batch size and type of data.
        In this case FACE, EYES and LANDMARKS.
        :param batch_size: batch size
        :return: empty data arrays with correct input size.
        """
        return [arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.FACE])),
                arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.EYES])),
                arrange_array(arrange_array_size(batch_size, self.base_model.input_size[input_type.LANDMARKS]))]

    def arrange_final_data(self,  input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but only
        some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[0][batch_pos] = input_data[1]
        batch_data[1][batch_pos] = input_data[2]
        batch_data[2][batch_pos] = input_data[3]
        return batch_data

    def prepare_data(self, train_data: DataTuple, validation_data: DataTuple, args: list, train: bool=True):
        """
        Overridden. Perform all necessary actions to data before fitting model.
        In this case, metadata and images are processed.
        :param train_data: training data DataTuple
        :param validation_data: validation data DataTuple
        :param train: True if training, False if validating/testing
        :param args: list of possible arguments
        :return: modified (or not) training and validation data,
                and list of arguments containing max look back and max/min training landmarks
        """
        train_data, validation_data, args = self.prepare_metadata(train_data, validation_data, args, train)
        return super().prepare_data(train_data, validation_data, args, train)

    def normalize_input_data(self, input_data: list, info: dict):
        """
        Overridden. Normalizes/standardizes input data (images and metadata) according to used functions.
        :param input_data: list of input data (images and metadata).
        :param info: preprocessing information for this specific frame.
        :return: normalized/standardized data
        """
        return super().normalize_input_data(input_data, info) + [self.normalize_metadata(input_data[3], info)]


@ExperimentHelper.register_subclass('NFEL5836_2918')
class NFEL5836_2918(NFEL5836):
    """
    NFEL5836_2918 experiment. Same as NFEL5836 but last FC has dimension of 2918D.
    Used only as pre-model for RNN models.
    """
    def __init__(self):
        """ Initializes exp. """
        super().__init__()
        self.name = "NFEL5836_2918"
        self.description = "Normalized face, eyes and landmarks, two-stream + metadata network, fc 5836D-2918D"
        self.fc_dimensions = 2918
        self.weights = exp_utils.NFEL5836_2918_VGG16
        self.min_lndmk = exp_utils.NFEL5836_2918_MIN_LNMDK
        self.max_lndmk = exp_utils.NFEL5836_2918_MAX_LNMDK
        print(self.name)
        print(self.description)


@ExperimentHelper.register_subclass('NFEL5836GRU')
class NFEL5836GRU(ExperimentHelper):
    """
    NFEL5836GRU experiment.
    """
    def __init__(self):
        """Initializes exp."""
        super().__init__()
        self.name = "NFEL5836GRU"
        self.description = "Sequence of normalized face, eyes and landmarks. Frozen static model, fine-tune fusion " \
                           "layers and train RNN-GRU module from scratch"
        self.recurrent_type = "gru"
        self.num_recurrent_layers = 1
        self.num_recurrent_units = 128
        self.look_back = 4
        self.weights = exp_utils.NFEL5836GRU_VGG16
        self.min_lndmk = exp_utils.NFEL5836GRU_MIN_LNMDK
        self.max_lndmk = exp_utils.NFEL5836GRU_MAX_LNMDK
        self.label_pos = -1
        self.model = get_model("two_stream_rnn")
        print(self.name)
        print(self.description)

        self.feature_arch = NFEL5836_2918()
        self.base_model = self.feature_arch.base_model

    def define_model(self, dropout: float):
        """
        Overridden. Defines model architecture based on experiment characteristics.
        :param dropout: dropout value
        """

        self.feature_arch.define_model(dropout)
        self.feature_arch.model.load_weights(exp_utils.get_file(self.feature_arch.weights))

        self.model.define(dropout=dropout, features_model=self.feature_arch.model.model,
                          base_model=self.base_model, n_units=self.num_recurrent_units,
                          lstm_layers=self.num_recurrent_layers, rec_type=self.recurrent_type)

    def compile_model(self, learning_rate: float):
        """
        Overridden. Compiles experiment model. Using ADAM optimizer that accumulates mini-batch updates.
        :param learning_rate: learning rate
        """
        self.model.compile(learning_rate, accum=True)

    def arrange_arrays(self, batch_size: int):
        """
        Overridden. Initialize data arrays for generator according to batch size and type of data.
        In this case FACE, EYES and LANDMARKS SEQUENCES of sequence length = self.look_back.
        :param batch_size: batch size
        :return: empty data arrays with correct input size.
        """
        return [arrange_array(
            arrange_sequence_array_size(batch_size, self.base_model.input_size[input_type.FACE], self.look_back)),
               arrange_array(
            arrange_sequence_array_size(batch_size, self.base_model.input_size[input_type.EYES], self.look_back)),
               arrange_array(
            arrange_sequence_array_size(batch_size, self.base_model.input_size[input_type.LANDMARKS], self.look_back))]

    def arrange_final_data(self,  input_data: list, batch_data: list, batch_pos: int, frame: int=0):
        """
        Overridden. Select data from input_data that will be fed to the network (note that all data is processed but only
        some of it is fed to the network according to experiment and model architecture).
        :param input_data: list of input data (images and metadata).
        :param batch_data: list of data to be fed to network.
        :param batch_pos: index within the batch (batch position).
        :param frame: frame number in sequence (only used with sequences).
        :return: batch_data
        """
        batch_data[0][batch_pos][frame] = input_data[1]
        batch_data[1][batch_pos][frame] = input_data[2]
        batch_data[2][batch_pos][frame] = input_data[3]
        return batch_data

    def prepare_data(self, train_data: DataTuple, validation_data: DataTuple, args: list, train: bool=True):
        """
        Overridden. Perform all necessary actions to data before fitting model.
        In this case, metadata and images are processed.
        See arrange_sequences for more info.
        :param train_data: training data DataTuple
        :param validation_data: validation data DataTuple
        :param train: True if training, False if validating/testing
        :param args: list of possible arguments
        :return: modified (or not) training and validation data,
                and list of arguments containing max look back and max/min training landmarks
        """
        train_data, validation_data, args = self.prepare_metadata(train_data, validation_data, args, train)
        if train:
            train_data = arrange_sequences(train_data, self.look_back, self.look_back)
        if validation_data is not None:
            validation_data = arrange_sequences(validation_data, self.look_back, args['max_look_back'])

        return train_data, validation_data, args

    def normalize_input_data(self, input_data: list, info: dict):
        """
        Overridden. Normalizes/standardizes input data (images and metadata) according to used functions.
        :param input_data: list of input data (images and metadata).
        :param info: preprocessing information for this specific frame.
        :return: normalized/standardized data
        """
        return super().normalize_input_data(input_data, info) + [self.normalize_metadata(input_data[3], info)]

    def decide_input_label(self, input_labels: list, info: dict=None):
        """
        Overridden. Decide final label for this frame/sequence.
        :param input_labels: list of input labels (there are as many labels as frames in sequence).
        :param info: preprocessing information for this specific frame.
        :return: value to be used as label in training.
        """
        return input_labels[self.label_pos]

