# !/usr/bin/env python
# title           :generate_training_FT.py
# description     :Script that processes FT videos and input data from EYEDIAP dataset and generates training and
#                  testing data files compatible with our gaze estimation network. Data is filtered according to
#                  specific criteria. Transformation matrices for normalized space are also computed here.
# author          :Cristina Palmero
# date            :01062017
# version         :2.0
# usage           : -
# notes           : FT: floating target. This script needs EYEDIAP dataset and folder structure to work.
#                   Change directories accordingly!
# python_version  :3.5.5
# ==============================================================================
my_path = "F:\\EYEDIAP"
EYEDIAP_path = "EYEDIAP"
import os
import sys
sys.path.append(os.path.join(my_path, EYEDIAP_path, 'Scripts'))
from EYEDIAP_misc import * # script provided by EYEDIAP

from EYEDIAP_utils import *
from data_utils import *
import numpy as np
from data_utils import get_normalized_data, write_vector_to_file

mod = 'FT'   # target type
type = ['S', 'M']  # type of head movement


# world coordinates to camera coordinates (see EYEDIAP for details)
Rw = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
tw = [0, 0, 1]
Rw = np.array(Rw)
tw = np.array(tw).reshape((3, 1))

for t in type:
    # Session selection (from EYEDIAP scripts)
    sessions = []
    for P in range(1, 17):
        if P < 12:
            Cs = ['A']
        elif P < 14:
            Cs = ['B']
        else:
            Cs = ['A', 'B']
        for C in Cs:
            sessions.append(getSessionIndex(P, C, mod, t))

    # Output files (all files are indexed:

    # Validity: stores whether an specific frame is valid in terms of head pose, screen coords/point of gaze,
    # correct gaze, face detection and face angles
    validity_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('validity_' + mod + '_' + t + '.txt')), 'w')

    # Data: stores directories of valid frames
    data_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('data_' + mod + '_' + t + '.txt')), 'w')

    # GTV: ground truth 3D gaze unit vectors of valid frames, in head coordinate system (HCS)
    gtv_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gtv_' + mod + '_' + t + '.txt')), 'wb')

    # GT: ground truth 2D gaze direction angles (yaw and pitch) of valid frames wrt HCS
    gt_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gt_' + mod + '_' + t + '.txt')), 'wb')

    # GTV cam: ground truth 3D gaze unit vectors of valid frames, in camera coordinate system (CCS)
    gtv_cam_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gtv_cam_' + mod + '_' + t + '.txt')), 'wb')

    # GT cam: ground truth 2D gaze direction angles (yaw and pitch) of valid frames wrt CCS
    gt_cam_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gt_cam_' + mod + '_' + t + '.txt')), 'wb')

    # GTH cam: ground truth 2D head direction angles of valid frames wrt CCS
    gth_cam_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gth_cam_' + mod + '_' + t + '.txt')), 'wb')

    # GTV cam: ground truth 3D head direction unit vectors of valid frames wrt CCS
    gthv_cam_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('gthv_cam_' + mod + '_' + t + '.txt')), 'wb')

    # Face features: contains all needed characteristics of face and transformation matrices needed to (pre)process
    # valid frames. This file contains the following 81 elements:
    # (followed by their respective notation in BMVC paper):
    #  0 seq_num (sequence number); 1 bb (bounding box);
    #  2 face patch warp matrix (W); 3 face patch normalization mat. (M); 4 face patch gaze normalization matrix (R);
    # 5 left eye patch warp matrix (W); 6 left eye patch norm. mat. (M); 7 left eye patch gaze norm. mat. (R);
    # 8 right eye patch warp matrix (W); 9 right eye patch norm. mat. (M); 10 right eye patch gaze norm. mat. (R);
    # 11 face ROI size; 12 eye ROI size; 13 - 80 3d landmarks
    facefeats_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('face_features_' + mod + '_' + t + '.txt')), 'w')

    # Sequence: information about sequences. A sequence is defined as a series of consecutive valid frames.
    # Format: sequence number; video code;init frame; end frame;
    sequence_file = open(os.path.join(my_path, EYEDIAP_path, 'Annotations', ('sequence_' + mod + '_' + t + '.txt')), 'w')

    frames_done = 0
    seq_num_t = 0
    for session_num in sessions:
        print("frames_done", frames_done)
        print("Session num: ", session_num)
        session_str = get_session_string(session_num)
        print("Session string: ", session_str)
        # session_str = '1_A_FT_M'

        mpath = os.path.join(my_path, EYEDIAP_path, 'Eyes2', session_str)
        if not os.path.exists(mpath):
            os.makedirs(mpath)

        # EYEDIAP data path
        head_track_file = os.path.join(my_path, EYEDIAP_path, 'Data', session_str, 'head_pose.txt')
        ball_track_file = os.path.join(my_path, EYEDIAP_path, 'Data', session_str, 'ball_tracking.txt')
        valid_file = os.path.join(my_path, EYEDIAP_path, 'Annotations', 'GazeState', 'GazeStateExport', 'Data',
                                  session_str,
                                  'gaze_state.txt')
        eyeball_centers_file = os.path.join(my_path, EYEDIAP_path, 'Metadata', 'Participants',
                                            'eyes_position_%d' % int(session_str.split('_')[0]) + '.txt')
        vga_calibration = os.path.join(my_path, EYEDIAP_path, 'Data', session_str, 'rgb_vga_calibration.txt')

        # Video frames data path
        frames_path = os.path.join(EYEDIAP_path, 'Data', session_str, 'frames')

        # Landmarks file path
        landmarks_file = os.path.join('F:\\landmarks', ('result3D_' + session_str + '.txt'))

        # If annotations on valid frames are available, then these are loaded and taken into account
        valid_frames = None
        if os.path.exists(valid_file):
            valid_frames = np.genfromtxt(valid_file, delimiter='\t', dtype=None)

        # Frames path
        frames_path = os.path.join(my_path, EYEDIAP_path, 'Data', session_str, 'frames')
        frames = [os.path.join(frames_path, file) for file in os.listdir(frames_path) if file.endswith(".bmp")]

        # Read the files with the frame-by-frame tracking parameters
        ball_track = read_ball_track_file(ball_track_file)
        head_track = read_head_track_file(head_track_file)

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

        # Read calibration file
        calib = readCalibrationFile(vga_calibration)

        # The estimated eyeball centers with respect to the head coordinate system
        eyeball_centers = np.loadtxt(eyeball_centers_file)

        # Containers for valid frames
        valid_head = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_ball = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_gaze = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_face = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_angles = np.zeros(ball_track.shape[0], dtype=np.bool)

        valid = False
        start_sequence = 0
        end_sequence = 0
        seq_num = 0
        eyeball_centers_moved = eyeball_centers.astype(dtype=np.float32, copy=True)
        eyeball_centers_moved[:, 1] = eyeball_centers_moved[:, 1] + 0.005
        for frameIndex in range(len(head_track[0])):
            print("...................")
            print("Frame: ", frameIndex)

            # Read the parameters for the current frame
            ball_pos = ball_track[frameIndex, :].reshape(3, 1)
            R = head_track[0][frameIndex, :, :]
            T = head_track[1][frameIndex, :].reshape(3, 1)

            # Validity criteria
            valid_head[frameIndex] = np.sum(T) != 0.0
            valid_ball[frameIndex] = np.sum(ball_pos) != 0
            valid_gaze[frameIndex] = valid_frames[frameIndex][1] == (b'OK' or b'BK') # Valid if OK or BlinKing
            valid_face[frameIndex] = np.sum(landmarks[frameIndex]) > 0.0

            # WCS to HCS system
            # Refer the ball center to the head coordinate system
            ball_HCS = np.dot(R.transpose(), ball_pos) - np.dot(R.transpose(), T)
            # Generate the ground truth gaze vectors
            gaze_vectors = ball_HCS.reshape(1, 3) - eyeball_centers
            gaze_vectors = gaze_vectors / (np.sqrt(np.sum(gaze_vectors ** 2, axis=1)).reshape(-1, 1))
            gaze_vector = np.mean(gaze_vectors, axis=0)  # we use the mean of the gaze vectors
            gaze_vector = gaze_vector / np.sqrt(np.sum(gaze_vector ** 2))
            gaze_angles = vector2angles(gaze_vector)

            # Valid angles according to (approx) max field of view when fixating in object
            valid_angles[frameIndex] = (-40 * np.pi / 180 < gaze_angles[0][0] < 40 * np.pi / 180) \
                           and (-30 * np.pi / 180 < gaze_angles[0][1] < 30 * np.pi / 180)

            valid_before = valid

            valid = valid_head[frameIndex] and valid_ball[frameIndex] and valid_gaze[frameIndex] \
                    and valid_face[frameIndex] and valid_angles[frameIndex]
            print("Valid: ", valid)
            validity_row = np.array((valid_head[frameIndex], valid_ball[frameIndex], valid_gaze[frameIndex],
                                     valid_angles[frameIndex], valid_face[frameIndex]))

            # Write validity file
            print("{};{};{}".format(session_str, format(frameIndex, '05'), np.array2string(validity_row, separator=';')),
                  file=validity_file)

            # Write sequence file
            # sequence number; video code;init frame; end frame;

            if not valid:
                if valid_before:
                    end_sequence = frameIndex - 1
                    print("{};{};{};{}".format(session_str, format(seq_num, '05'), format(start_sequence, '05'),
                                               format(end_sequence, '05')),
                          file=sequence_file)
                    seq_num = seq_num + 1
                    seq_num_t = seq_num_t + 1
                continue

            if not valid_before:
                start_sequence = frameIndex

            # WCS to CCS system
            eyeball_centers_WCS = np.dot(R, eyeball_centers.transpose()) + T
            eyeball_centers_moved_WCS = np.dot(R, eyeball_centers_moved.transpose()) + T

            eyeball_centers_CCS = np.dot(Rw.transpose(), eyeball_centers_WCS) - np.dot(Rw.transpose(), tw)
            eyeball_centers_moved_CCS = np.dot(Rw.transpose(), eyeball_centers_moved_WCS) - np.dot(Rw.transpose(), tw)

            ball_CCS = np.dot(Rw.transpose(), ball_pos) - np.dot(Rw.transpose(), tw)

            gaze_vectors_CCS = ball_CCS.reshape(1, 3) - eyeball_centers_CCS.transpose()

            gaze_vectors_CCS = gaze_vectors_CCS / (np.sqrt(np.sum(gaze_vectors_CCS ** 2, axis=1)).reshape(-1, 1))

            gaze_vector_CCS = np.mean(gaze_vectors_CCS, axis=0)

            gaze_vector_CCS = gaze_vector_CCS / np.sqrt(np.sum(gaze_vector_CCS ** 2))

            gaze_angles_CCS = vector2angles(gaze_vector_CCS)

            # Head direction in CCS
            aux_vector = np.zeros((3, 1), dtype=np.float32)
            aux_vector[2, 0] = 1.0
            head_vector_WCS = np.dot(R, aux_vector)
            head_vector_CCS = np.dot(Rw.transpose(), head_vector_WCS)
            head_vector_CCS = head_vector_CCS / np.sqrt(np.sum(head_vector_CCS ** 2))
            head_angle_CCS = vector2angles(head_vector_CCS)

            # Face normalization
            # we assume that center of face is 10 cm away from the center of the face in Z
            head = np.array([[0.0], [0.0], [0.1]]).reshape(1, 3)
            head_center_WCS = np.dot(head, R.transpose()) + T.transpose()

            head_center_CCS = np.dot(Rw.transpose(), head_center_WCS.transpose()) - np.dot(Rw.transpose(), tw)

            mean_face = head_center_CCS * 1000
            mean_face = mean_face.transpose()

            face_roi_size = [250, 250]
            face_patch_conv, face_patch_warp, face_patch_gaze = get_normalized_data(mean_face[0], R, face_roi_size,
                                                                                    calib)
            eyeball_centers_CCS1000 = eyeball_centers_moved_CCS * 1000

            eyes_roi_size = [70, 58]
            leye_patch_conv, leye_patch_warp, leye_patch_gaze = get_normalized_data(eyeball_centers_CCS1000[:,0], R,
                                                                                    eyes_roi_size, calib)
            reye_patch_conv, reye_patch_warp, reye_patch_gaze = get_normalized_data(eyeball_centers_CCS1000[:,1], R,
                                                                                    eyes_roi_size, calib)

            # Face Bounding box
            # Get max distance between landmarks
            max_dist = -1
            for l1 in landmarks[frameIndex, :, :2]:
                for l2 in landmarks[frameIndex,:,:2]:
                    if l1 is not l2:
                        dist = np.linalg.norm(l1 - l2)
                        if dist > max_dist:
                            max_dist = dist

            mean_landmarks = np.mean(landmarks[frameIndex,:,:2], axis=0)
            bb_height = max_dist
            bb_dims = np.empty([4,1])
            bb_dims[0] = mean_landmarks[0] - bb_height/2 # x
            bb_dims[1] = mean_landmarks[1] - bb_height/2 # y
            bb_dims[2] = bb_dims[3] = bb_height

            # Write annotations in files
            np.savetxt(gtv_cam_file, gaze_vector_CCS.reshape((1, 3)), fmt='%1.10f', delimiter='\t', newline='\n')
            np.savetxt(gt_cam_file, gaze_angles_CCS, fmt='%1.10f', delimiter='\t', newline='\n')
            np.savetxt(gtv_file, gaze_vector.reshape((1, 3)), fmt='%1.10f', delimiter='\t', newline='\n')
            np.savetxt(gt_file, gaze_angles, fmt='%1.10f', delimiter='\t', newline='\n')
            np.savetxt(gthv_cam_file, head_vector_CCS.reshape((1, 3)), fmt='%1.10f', delimiter='\t', newline='\n')
            np.savetxt(gth_cam_file, head_angle_CCS, fmt='%1.10f', delimiter='\t', newline='\n')

            img_file = os.path.join(frames_path, format(frameIndex, '05') + '.bmp')
            data_file.write(img_file + '\n')

            # 0 seq_num; 1 bb;
            # 2 face patch warp; 3 face patch conv; 4 face patch gaze;
            # 5 leye patch warp; 6 leye patch conv; 7 leye patch gaze;
            # 8 reye patch warp; 9 reye patch conv; 10 reye patch gaze;
            # 11 face roi size; 12 eye roi size; 13 - 80 3d landmarks
            facefeats_file.write(str(seq_num_t) + ';')
            facefeats_file.write(str(bb_dims[0][0]) + ',' + str(bb_dims[1][0]) + ',' + str(bb_dims[2][0]) + ','
                                 + str(bb_dims[3][0]) + ';')

            write_vector_to_file(facefeats_file, face_patch_warp.reshape(9,1))
            write_vector_to_file(facefeats_file, face_patch_conv.reshape(9,1))
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

            if (frameIndex + 1) == len(head_track[0]):
                end_sequence = frameIndex
                print("{};{};{};{}".format(session_str, format(seq_num, '05'), format(start_sequence, '05'),
                                           format(end_sequence, '05')),
                    file=sequence_file)
                seq_num = seq_num + 1
                seq_num_t = seq_num_t + 1

    data_file.close()
    gt_file.close()
    gtv_file.close()
    gt_cam_file.close()
    gtv_cam_file.close()
    gth_cam_file.close()
    gthv_cam_file.close()
    validity_file.close()
    sequence_file.close()
    facefeats_file.close()
