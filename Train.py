# !/usr/bin/env python
# title           :Train.py
# description     :Main script to TRAIN the gaze estimation network
# author          :Cristina Palmero
# date            :30092018
# version         :2.0
# usage           :Example (see more @init_main):
#                   python3 Train.py -t FT_SM_NFEL5836GRU_fold3 -exp NFEL5836GRU -dp 0.3 -bs 8 -aug 1
#                   -lr 0.0001 -epochs 21 -gt "/Work/EYEDIAP/Annotations_final_exps/gt_cam_FT_S.txt"
#                   "/Work/EYEDIAP/Annotations_final_exps/gt_cam_FT_M.txt"
#                   -vgt "/Work/EYEDIAP/Annotations_final_exps/gtv_cam_FT_S.txt"
#                   "/Work/EYEDIAP/Annotations_final_exps/gtv_cam_FT_M.txt"
#                   -data "/Work/EYEDIAP/Annotations_final_exps/data_FT_S.txt"
#                   "/Work/EYEDIAP/Annotations_final_exps/data_FT_M.txt"
#                   -feats "/Work/EYEDIAP/Annotations_final_exps/face_features_FT_S.txt"
#                   "/Work/EYEDIAP/Annotations_final_exps/face_features_FT_M.txt"
#                   -test 2_A_FT_S 2_A_FT_M 3_A_FT_S 3_A_FT_M 8_A_FT_S 8_A_FT_M 16_A_FT_S 16_A_FT_M 16_B_FT_S 16_B_FT_M
#                   -p "/Work"

# notes           : -
# python_version  :3.5.5
# ==============================================================================

N_SEED = 32

import random as rn
import argparse
import pickle

from experiment_helper import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K
from images_data_augmenter_seqaware import ImageDataAugmenter


def str2bool(v):
    """
    Convert string to boolean
    :param v: string
    :return: boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_main():
    """
    Defines the type of input arguments expected.
    Definition of each of them is included in "help" variable of each argument
    :return: parsed input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-exp", "--experiment", dest="experiment", default="NFEL5836", help="Experiment name")
    parser.add_argument("-dp", "--dropout", dest="dropout", type=float, default=0.3, help="Dropout value")
    parser.add_argument("-aug", "--augmentation", dest="augmentation", type=str2bool, default=True,
                        help="True if Data augmentation is activated")
    parser.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate",type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("-data", "--data_file", dest="data_files", default=[], help="Data file", action="append",
                        nargs=2)
    parser.add_argument("-gt", "--gt_files", dest="gt_files", default=[], help="Ground truth files", action="append",
                        nargs=2)
    parser.add_argument("-vgt", "--vector_gt_files", dest="vector_gt_files", default=[],
                        help="Vector ground truth files", action="append", nargs=2)
    parser.add_argument("-feats", "--face_features", dest="face_features_file", default=[], help="Face features file",
                        nargs=2)
    parser.add_argument("-test", "--test_folders", dest="test_folders", default=[], help="Test folders",
                        action="append", nargs=10)
    parser.add_argument("-vp", "--validation_participants", dest="val_parts", type=int, default=0,
                        help="Number of participants to perform validation on")
    parser.add_argument("-mlb", "--max_look_back", dest="max_look_back", type=int, default=4,
                        help="Maximum number of frames to take into account before current frame, in sequence mode")
    parser.add_argument("-t", "--title", dest="title", default="", help="Experiment description")
    parser.add_argument("-p", "--path", dest="path", type=str, default="", help="Path")
    parser.add_argument("-mp", "--multi_processing", dest="multi_processing", type=str2bool, default=False,
                        help="True if GPU multi processing is activated")
    return parser.parse_args()


if __name__ == '__main__':
    # Parse input arguments
    print("Parsing arguments...")
    args = init_main()

    # Read data and ground truth (both 2D and 3D)
    print("Reading input files...")
    data, gt, vgt, _ = read_input(args.data_files, args.path, args.gt_files, args.vector_gt_files)

    # Read face features
    print("Reading face features...")
    face_features = read_face_features_file(args.face_features_file)

    # Get train-validation split
    print("Splitting data in train and validation sets...")
    train, validation = train_valtest_split(data, vgt, face_features, args.test_folders, args.val_parts)

    # Get experiment details and methods
    print("Get experiment and define associated model...")
    experiment = ExperimentHelper.get_experiment(args.experiment)

    print("Preparing data...")
    variables = {'max_look_back': args.max_look_back}
    train, validation, variables = experiment.prepare_data(train, validation, variables)

    # Make sure that from this point on experiments are reproducible (not valid with multi_processing)
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed=N_SEED)
    rn.seed(N_SEED)

    augmenter = None
    if args.augmentation:
        print("Loading augmentation...")
        # Define data augmenter
        augmenter = ImageDataAugmenter(rotation_range=0,
                                       width_shift_range=5,  # pixels
                                       height_shift_range=5,  # pixels
                                       zoom_range=[0.98, 1.02],  # %
                                       horizontal_flip=True,
                                       illumination_range=[0.4, 1.75],
                                       gaussian_noise_range=0.03)
        print("Augmentation is on.")
    print(augmenter)  # Just checking

    # Shuffle
    print("Initiate data generators...")
    train = unison_shuffled_copy(train)
    print("Training: ", len(train.x))
    experiment.init_data_gen_train(train, args.batch_size, augmenter, True, True)
    if validation is not None:
        validation = unison_shuffled_copy(validation)
        print("Test: ", len(validation.x))
        experiment.init_data_gen_val(validation, args.batch_size, None, False)

    print("Define and compile model...")
    experiment.define_model(args.dropout)
    experiment.compile_model(args.learning_rate)

    # Keras Callbacks
    # Checkpoint model
    # NOTE: With Lambda, used in LSTM model definition, model.save and model.to_json() do not work, so only
    # weights checkpoints can be saved
    if validation is not None:
        filepath = os.path.join(args.path, "Results", str(experiment.base_model.__class__.__name__) +
                                "-{epoch:03d}-{val_loss:.5f}-" + str(args.experiment) + ".hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=7)  # 21
    else:
        filepath = os.path.join(args.path, "Results", str(experiment.base_model.__class__.__name__) +
                                "-{epoch:03d}-{loss:.5f}-" + str(args.experiment) + ".hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, period=1)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=55, verbose=1, mode='min')

    # CSV logger
    csvlogger = CSVLogger("log_" + str(args.experiment) + ".csv", append=True, separator=";")

    # Earlystopping is only saved when evaluation on a validation set.
    if validation is not None:
        callbacks_list = [earlystopping, csvlogger, checkpoint]
    else:
        callbacks_list = [csvlogger, checkpoint]

     # Fit model using Keras data generator
    print("Start training...")
    fit_args = dict(steps_per_epoch=np.ceil(len(train.x) / args.batch_size),
                    epochs=args.n_epochs, callbacks=callbacks_list)
    if len(K.tensorflow_backend._get_available_gpus()) > 0 and args.multi_processing:
        fit_args.update(dict(max_queue_size=10, workers=8, use_multiprocessing=True))
    if validation is not None:
        fit_args.update(dict(validation_data=experiment.val_data_generator,
                        validation_steps=np.ceil(len(validation.x) / args.batch_size)))

    hist = experiment.model.fit_generator(experiment.train_data_generator, fit_args)


    print("Saving model weights...")
    # Save model weights
    experiment.model.save(
        os.path.join(args.path, "Results", str(args.experiment) + "_" + args.title + "_model.hdf5"), True)

    # Save history
    print("Saving history...")
    with open(os.path.join(args.path, "Results", str(args.experiment) + '_' + args.title + "_hist.pickle"), 'wb') as f:
        more = {'test_folders': args.test_folders, 'training_length': len(train.x), 'batch_size': args.batch_size}
        if validation is not None:
            more.update({'validation_participants': validation.parts, 'validation_length': len(validation.x)})
        else:
            more.update({'validation_participants': 0, 'validation_length': 0})
        more.update(variables)
        stats = [hist.history, hist.params, more]
        pickle.dump(stats, f)




