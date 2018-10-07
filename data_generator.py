# !/usr/bin/env python
# title           :data_generator.py
# description     :Class that generates input data for the network in an online fashion.
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================

from keras.utils import Sequence as kerasSequence
import matplotlib.pyplot as plt
from data_utils import *
from images_data_augmenter_seqaware import ImageDataAugmenter


class DataGenerator(kerasSequence):
    """
    Generates input data for network, optionally augmenting it online. The generation function is created in a generic
    fashion. Specific methods have to be implemented outside this class and passed using set_methods.
    """

    def __init__(self,
                 data: list,
                 labels: list,
                 feats: list,
                 batch_size: int = 1,
                 augmenter=None,
                 shuffle: bool = False,
                 debug: bool = False):
        """
        Initializes class. Data/labels/feats are indexed equally andtreated as X-image sequences.
        :param data: list of image directories.
        :param labels: list of labels.
        :param feats: list of features (information needed to preprocess data)
        :param batch_size: size of batch
        :param augmenter: Augmenter class
        :param shuffle: True to shuffle data before starting an epoch
        :param debug: True for debugging (not used, kept for legacy purposes).
        """
        self.data = data
        self.labels = labels
        self.feats = feats
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.augment = augmenter is not None
        self.debug = debug

        # --- METHODS ---
        self.look_back_range = None
        self.arrange_arrays = None
        self.arrange_label_array = None
        self.get_preprocess_info = None
        self.load_image = None
        self.preprocess_input_data = None
        self.preprocess_input_label = None
        self.resize_input_data = None
        self.prepare_tensor_dims = None
        # Here normalize means mean centering, standardizing, etc. not converting to normalized space as in paper.
        # That is done within preprocess_input_data and preprocess_input_label
        self.normalize_input_data = None
        self.arrange_final_data = None
        self.decide_input_label = None

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, batch_index: int):
        """
        Generates one batch of data
        :param batch_index: index of current batch
        """
        batch_idxs = self.idxs[batch_index:(batch_index + self.batch_size)]

        x, y = self.__data_generation(batch_idxs)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            self.idxs = np.random.permutation(len(self.data))
        else:
            self.idxs = list(range(len(self.data)))

    def set_methods(self,
                    arrange_arrays=None,
                    arrange_label_array=None,
                    look_back_range=None,
                    get_preprocess_info=None,
                    load_image=None,
                    preprocess_input_data=None,
                    preprocess_input_label=None,
                    resize_input_data=None,
                    prepare_tensor_dims=None,
                    normalize_input_data=None,
                    arrange_final_data=None,
                    decide_input_label=None
                    ):
        """
        Sets methods implemented outside DataGenerator to use them in __data_generation.
        :param arrange_arrays: Func to create empty input arrays depending on network characteristics (i.e. input size)
        :param arrange_label_array: Func to create empty array for labels, depending on network charact. (num outputs)
        :param look_back_range: Func to obtain sequence length range (0,look_back) (0 in case of 1 frame)
        :param get_preprocess_info: Func to obtain all info needed to preprocess input data
        :param load_image: Func to load images given data content
        :param preprocess_input_data: Func to preprocess input data given preprocess info
        :param preprocess_input_label: Func to preprocess labels given preprocess info
        :param resize_input_data: Func to resize input images to be compatible with network
        :param prepare_tensor_dims: Func to modify tensor dimensions to be compatible with network
        :param normalize_input_data: Func to standardize/normalize input data (not in the sense of converting to norm. space!)
        :param arrange_final_data: Func to fill input data arrays with preprocessed data
        :param decide_input_label: Func to fill input label array with preprocessed data
        """
        if arrange_arrays is not None:
            self.arrange_arrays = arrange_arrays
        if arrange_label_array is not None:
            self.arrange_label_array = arrange_label_array
        if look_back_range is not None:
            self.look_back_range = look_back_range
        if get_preprocess_info is not None:
            self.get_preprocess_info = get_preprocess_info
        if load_image is not None:
            self.load_image = load_image
        if preprocess_input_data is not None:
            self.preprocess_input_data = preprocess_input_data
        if preprocess_input_label is not None:
            self.preprocess_input_label = preprocess_input_label
        if resize_input_data is not None:
            self.resize_input_data = resize_input_data
        if prepare_tensor_dims is not None:
            self.prepare_tensor_dims = prepare_tensor_dims
        if normalize_input_data is not None:
            self.normalize_input_data = normalize_input_data
        if arrange_final_data is not None:
            self.arrange_final_data = arrange_final_data
        if decide_input_label is not None:
            self.decide_input_label = decide_input_label

    def __data_generation(self, batch_idxs: list):
        """
        Generates data containing batch_idxs samples from current batch. Each data sample is processed accordingly
        to feats info and network characteristics. Online augmentation is possible if activated.
        :param batch_idxs: indexes of current batch samples
        :return: pairs of input data and labels, formatted according to network characteristics
        """

        # Initialize input arrays, which will be fed to network
        batch_data = self.arrange_arrays(self.batch_size)
        batch_labels = self.arrange_label_array(self.batch_size)

        # Read in and preprocess a batch of images/sequences (called sample onwards). For each sample:
        for idi, i in enumerate(batch_idxs):

            # bb = self.compute_max_bb() # THIS IS NOT INCLUDED IN BMVC VERSION

            # Create array for processed labels
            input_labels = np.array(self.labels[i], copy=True)
            input_labels_processed = np.empty((input_labels.shape[0], batch_labels.shape[1]), dtype=batch_labels.dtype)

            # Augmenter class can keep the last augmentation state (i.e. random values and augmentation procedures).
            # If augmentation is activated, reset state before starting sample content.
            keep = False
            if self.augment:
                last_state = self.augmenter.reset_state()

            # Create range of sample length (for 1-frame sequence sample length is 1).
            look_back_range = self.look_back_range()

            # Get preprocess info for current sample
            preprocess_info = self.get_preprocess_info(self.feats[i])

            # Per each element of sample:
            for f in look_back_range:
                # Load image
                x = self.load_image(self.data[i][f])

                # Preprocess input data and labels according to preprocess_info
                input_data = self.preprocess_input_data(x, preprocess_info[f])
                input_labels_processed[f] = self.preprocess_input_label(input_labels[f], preprocess_info[f])

                # Augment data
                if self.augment:
                    assert (type(self.augmenter) is ImageDataAugmenter)
                    if f > 0:
                        keep = True

                    # Perform augmentation
                    input_data, input_labels_processed[f], last_state = \
                        self.augmenter.augment(*input_data, input_labels_processed[f], keep, last_state)

                # Resize input data according to network characteristics
                input_data = self.resize_input_data(input_data, preprocess_info[f])

                # debug
                if self.debug:
                    fig, (ax2, ax3, ax4) = plt.subplots(1, 3)
                    fig.suptitle(self.data[i][f] + "\ny:" + str(input_labels_processed[f]))
                    ax2.set_title('original_face')
                    ax3.set_title('norm face')
                    ax4.set_title('eyes')
                    ax2.imshow(input_data[0] / 255)
                    ax3.imshow(input_data[1] / 255)
                    ax4.imshow(input_data[2] / 255)
                    plt.savefig('images/test/' + str(i) + '_' + str(f) + '_final.jpg')
                    plt.show()
                    plt.close("all")

                # Modify tensor dimensions according to network characteristics
                input_data = self.prepare_tensor_dims(input_data)

                # Normalize/standardize/mean center input data
                input_data = self.normalize_input_data(input_data, preprocess_info[f])

                # Store in batch data only the data needed for specific architecture
                batch_data = self.arrange_final_data(input_data, batch_data, idi, f)

            # Decide which label to feed into network
            y = self.decide_input_label(input_labels_processed, preprocess_info)
            batch_labels[idi, :] = y

        return batch_data, batch_labels
