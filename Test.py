# !/usr/bin/env python
# title           :Test.py
# description     :Main script to test gaze estimation models as a batch process
# author          :Cristina Palmero
# date            :30092018
# version         :1.0
# usage           :Example: -exp NFEL5836_2918 -data C:\path\Test\data.txt -info C:\path\Test\CCS_3D_info.txt
#                           -lndmk C:\path\Test\landmarks.txt -cal C:\path\Test\calibration.txt
#                           -p C:\path\
#                  Example 2: -exp NFEL5836_2918 -data C:\path\Test\data.txt -cal C:\path\Test\calibration.txt
# #                           -feats C:\path\Test\face_features.txt -p C:\path\
# notes           :Inputs are read from files to keep compatibility with Train code and associated methods.
# python_version  :3.5.5
# ==============================================================================

import argparse

from experiment_helper import *
from data_utils import *
from matplotlib import pyplot as plt

def init_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-exp", "--experiment", dest="experiment", type=str, default="NFEL5836", help="Experiment name")
    parser.add_argument("-data", "--data_file", dest="data_file", type=str, default="", help="Image paths file")
    parser.add_argument("-feats", "--face_features", dest="face_features_file", type=str, default="",
                        help="Face features file")
    parser.add_argument("-ec", "--eyes_center", dest="eyes_center_file", type=str, default="",
                        help="File with middle point between both eyes, for plotting purposes")
    parser.add_argument("-p", "--path", dest="path", type=str, default="", help="Path")
    parser.add_argument("-mlb", "--max_look_back", dest="max_look_back", type=int, default=4,
                        help="Maximum number of frames to take into account before current frame, in sequence mode")

    # If face_features file has not been computed yet:
    parser.add_argument("-info", "--info_3D_file", dest="info_3D_file", type=str, default="",
                        help="File with 3D face landmarks and head pose wrt camera coordinate system per frame")
    parser.add_argument("-lndmk", "--landmarks_file", dest="landmarks_file", type=str, default="",
                        help="File with Bulat et al landmarks per frame")
    parser.add_argument("-cal", "--cam_calibration_file", dest="cam_calibration_file", type=str, default="",
                        help="Camera calibration matrix file, .yml format")

    return parser.parse_args()


def project_points(points_3D, intrinsic_mat):
    points_2D = np.dot(points_3D, intrinsic_mat.transpose())
    points_2D = points_2D[:, :2] / (points_2D[:, 2].reshape(-1, 1))
    return points_2D


def project_gaze(init_vector, vector, intrinsic_mat):
    offset = 500.0
    points_3D = np.empty((2,3))
    points_3D[0,:] = init_vector
    points_3D[1,:] = init_vector + vector*offset
    return project_points(points_3D, intrinsic_mat)


def euler2rot_mat(euler_angles):
    """
    Convert euler to rotation matrix, using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
    (from Openface)
    :param euler_angles: euler angles
    :return: rotation matrix
    """
    s1 = np.sin(euler_angles[0])
    s2 = np.sin(euler_angles[1])
    s3 = np.sin(euler_angles[2])
    c1 = np.cos(euler_angles[0])
    c2 = np.cos(euler_angles[1])
    c3 = np.cos(euler_angles[2])

    rot_mat = np.empty((3,3), dtype=np.float32)
    rot_mat[0, 0] = c2 * c3
    rot_mat[0, 1] = -c2 * s3
    rot_mat[0, 2] = s2
    rot_mat[1, 0] = c1 * s3 + c3 * s1 * s2
    rot_mat[1, 1] = c1 * c3 - s1 * s2 * s3
    rot_mat[1, 2] = -c2 * s1
    rot_mat[2, 0] = s1 * s3 - c1 * c3 * s2
    rot_mat[2, 1] = c3 * s1 + c1 * s2 * s3
    rot_mat[2, 2] = c1 * c2
    return np.linalg.inv(rot_mat)


def read_calibration_file(file):
    """
    Reads the calibration parameters
    """
    calib = {}
    f = open(file, 'r')

    # Read the [resolution] section
    f.readline().strip()
    calib['size'] = [int(value) for value in f.readline().strip().split(';')]
    calib['size'] = calib['size'][0], calib['size'][1]

    # Read the [intrinsics] section
    f.readline().strip()
    values = []
    for i in range(3):
        values.append([float(value) for value in f.readline().strip().split(';')])
    calib['intrinsics'] = np.array(values).reshape(3, 3)

    # Read the [distortion] section
    f.readline().strip()
    calib['distortion'] = [float(value) for value in f.readline().strip().split(';')]

    return calib


def generate_face_feats_file(landmarks_file, info_3D_file, calibration_file, path):
    facefeats_file = open(os.path.join(path, 'Test','face_features.txt'), 'w')
    eyescenter_file = open(os.path.join(path, 'Test','eyes_center.txt'), 'w')
    facefeats_file_name = facefeats_file.name
    eyescenter_file_name = eyescenter_file.name

    calib = read_calibration_file(calibration_file)

    # Read Bulat et al ICCV2017 landmarks
    # Format: one row per frame, all landmark coordinates separated by ' '
    # e.g. "x1 y1 z1 x2 y2 z2 ...."
    with open(landmarks_file, 'rb') as infile:
        siz = sum(1 for _ in infile)
        landmarks = np.empty((siz, 68, 3))
    with open(landmarks_file, 'rb') as infile:
        for i, line in enumerate(infile):
            myline = line.split(b' ')
            for j in list(range(0, 68)):
                landmarks[i, j, :] = myline[3 * j:3 * j + 3]

    # 3D landmarks wrt Camera coordinate system, to compute coordinates of mean face and eyes centers.
    landmarks_CCS = np.empty((siz, 68, 3))
    R = np.empty((siz, 3))
    T = np.empty((siz, 3))
    with open(info_3D_file, 'rb') as infile:
        for i, line in enumerate(infile):
            if i > 0:
                myline = line.split(b';')
                T[i-1, :] = myline[0:3]
                R[i-1, :] = myline[3:6]
                for j in list(range(0, 68)):
                    landmarks_CCS[i-1, j, :] = myline[6 + 3 * j:6 + 3 * j + 3]

    face_roi_size = [250, 250]
    eyes_roi_size = [70, 58]
    for frameIndex in range(siz):

        # Face Bounding box
        # Get max distance between landmarks
        max_dist = -1
        for l1 in landmarks[frameIndex, :, :2]:
            for l2 in landmarks[frameIndex, :, :2]:
                if l1 is not l2:
                    dist = np.linalg.norm(l1 - l2)
                    if dist > max_dist:
                        max_dist = dist

        mean_landmarks = np.mean(landmarks[frameIndex, :, :2], axis=0)
        bb_height = max_dist
        bb_dims = np.empty([4, 1])
        bb_dims[0] = mean_landmarks[0] - bb_height / 2  # x
        bb_dims[1] = mean_landmarks[1] - bb_height / 2  # y
        bb_dims[2] = bb_dims[3] = bb_height

        mean_face = np.mean(landmarks_CCS[frameIndex], axis=0)

        eyes_center = np.mean(landmarks_CCS[frameIndex, 36:48, :], axis=0)

        leye_center = np.mean(landmarks_CCS[frameIndex, 36:42, :], axis=0)
        reye_center = np.mean(landmarks_CCS[frameIndex, 42:48, :], axis=0)

        # Openface rotation is given as euler angles, the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
        R_mat = euler2rot_mat(R[frameIndex,:])

        face_patch_conv, face_patch_warp, face_patch_gaze = get_normalized_data(mean_face, R_mat, face_roi_size,
                                                                                calib)

        leye_patch_conv, leye_patch_warp, leye_patch_gaze = get_normalized_data(leye_center, R_mat,
                                                                                eyes_roi_size, calib)
        reye_patch_conv, reye_patch_warp, reye_patch_gaze = get_normalized_data(reye_center, R_mat,
                                                                                eyes_roi_size, calib)

        # 0 seq_num; 1 bb;
        # 2 face patch warp; 3 face patch conv; 4 face patch gaze;
        # 5 leye patch warp; 6 leye patch conv; 7 leye patch gaze;
        # 8 reye patch warp; 9 reye patch conv; 10 reye patch gaze;
        # 11 face roi size; 12 eye roi size; 13 - 80 3d landmarks
        facefeats_file.write(str(0) + ';')
        facefeats_file.write(str(bb_dims[0][0]) + ',' + str(bb_dims[1][0]) + ',' + str(bb_dims[2][0]) + ','
                             + str(bb_dims[3][0]) + ';')

        write_vector_to_file(facefeats_file, face_patch_warp.reshape(9, 1))
        write_vector_to_file(facefeats_file, face_patch_conv.reshape(9, 1))
        write_vector_to_file(facefeats_file, face_patch_gaze.reshape(9, 1))

        write_vector_to_file(facefeats_file, leye_patch_warp.reshape(9, 1))
        write_vector_to_file(facefeats_file, leye_patch_conv.reshape(9, 1))
        write_vector_to_file(facefeats_file, leye_patch_gaze.reshape(9, 1))

        write_vector_to_file(facefeats_file, reye_patch_warp.reshape(9, 1))
        write_vector_to_file(facefeats_file, reye_patch_conv.reshape(9, 1))
        write_vector_to_file(facefeats_file, reye_patch_gaze.reshape(9, 1))

        facefeats_file.write(str(face_roi_size[0]) + ',' + str(face_roi_size[1]) + ';')
        facefeats_file.write(str(eyes_roi_size[0]) + ',' + str(eyes_roi_size[1]) + ';')
        for j in list(range(0, 68)):
            if j < 67:
                facefeats_file.write(str(landmarks[frameIndex, j, 0]) + ',' + str(landmarks[frameIndex, j, 1])
                                     + ',' + str(landmarks[frameIndex, j, 2]) + ';')
            else:
                facefeats_file.write(str(landmarks[frameIndex, j, 0]) + ',' + str(landmarks[frameIndex, j, 1])
                                     + ',' + str(landmarks[frameIndex, j, 2]))
        facefeats_file.write('\n')

        eyescenter_file.write(str(eyes_center[0]) + ',' + str(eyes_center[1]) + ',' + str(eyes_center[2]) + '\n')

    facefeats_file.close()
    eyescenter_file.close()

    return facefeats_file_name, eyescenter_file_name


if __name__ == '__main__':

    args = init_main()
    batch_size = 1

    if args.face_features_file == "":
        args.face_features_file, args.eyes_center_file = generate_face_feats_file(
            args.landmarks_file, args.info_3D_file, args.cam_calibration_file, args.path)

    print("Reading input files...")

    data, _, _,  _ = read_input([[args.data_file]], args.path)
    face_features = read_face_features_file([args.face_features_file])
    eyes_center = np.loadtxt(args.eyes_center_file, delimiter=',')
    calib = read_calibration_file(args.cam_calibration_file)

    # Treat train split as validation, as we are not going to perform training.
    gt = [[0.0, 0.0, 0.0] for i in range(len(face_features))]  # dummy GT for compatibility
    validation, _ = train_valtest_split(data, gt, face_features, [])

    # Get experiment details and methods
    print("Get experiment and define associated model...")
    experiment = ExperimentHelper.get_experiment(args.experiment)

    print("Preparing data...")
    variables = {'max_look_back': args.max_look_back}
    train, validation, variables = experiment.prepare_data(None, validation, variables, False)

    print("Initialize data generator...")
    experiment.init_data_gen_val(validation, batch_size, None, False, False) #, True)

    print("Loading model and weights...")
    experiment.load_model()

    # Predict gaze vector for each frame and show results
    for idx in list(range(int(np.ceil(len(validation.x) / batch_size)))):
        input_x, _ = experiment.val_data_generator.__getitem__(idx)
        normalized_predictions = experiment.model.predict(input_x)
        frame_feats = copy_face_features(validation.feats[idx][experiment.label_pos])
        gaze_conv = get_gaze_conv(frame_feats)
        predictions = denormalize_gaze(gaze_conv, normalized_predictions)
        print("Predicted 3D gaze vector: ", predictions[:,0])

        img = load_image(validation.x[idx][experiment.label_pos])
        projected_gaze = project_gaze(np.array(eyes_center[validation.idxs[idx][experiment.label_pos]]),
                                      predictions[:,0], calib['intrinsics'])

        fig, ax = plt.subplots()
        ax.imshow(img/255)
        projected_gaze[:,0]
        ax.plot(projected_gaze[:,0], projected_gaze[:,1], '-', linewidth=3, color='firebrick')
        plt.show()
