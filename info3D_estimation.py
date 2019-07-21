import numpy as np
import os
from ctypes import *
from numpy.ctypeslib import ndpointer

landmarks_model_num = 68
landmarks2D_size = 2*landmarks_model_num
landmarks3D_size = 3*landmarks_model_num
pose_size = 3
aus_intensity_size = 17
aus_presence_size = 18

landmarks2D_array = c_double * landmarks2D_size
landmarks3D_array = c_double * landmarks3D_size
head_pose_array = c_double * pose_size
aus_intensity_array = c_double * aus_intensity_size
aus_presence_array = c_double * aus_presence_size


class FACE_INFO(Structure):
    """
    Ctype data structure for OpenFaceWrapper shared library
    """
    _fields_ = [("certainty", c_float),
                ("detection_success", c_bool),
                ("landmarks2D", landmarks2D_array),
                ("landmarks3D", landmarks3D_array),
                ("head_rotation", head_pose_array),
                ("head_position", head_pose_array),
                ("aus_intensity", aus_intensity_array),
                ("aus_presence", aus_presence_array)]


class Info3DEstimation(object):
    """
    Info3DEstimation class. OpenFace wrapper https://github.com/TadasBaltrusaitis/OpenFace
    """
    def __init__(self, lib_path=""):
        """
        Loads OpenFace default models (FaceAnalyser and LandmarkDetector) from shared library (dll or so)
        :param lib_path: directory where openfacewrapper library and other needed files/libraries are located
        """
        os.environ['PATH'] = ';'.join([os.environ['PATH'], lib_path])

        if os.name == 'nt':
            self.info3D_model = cdll.LoadLibrary(os.path.join(lib_path, "OpenFaceDLL.dll"))
        elif os.name == 'posix':
            self.info3D_model = cdll.LoadLibrary(os.path.join(lib_path, "OpenFaceDLL.so"))
        else:
            RuntimeError("ERROR: Undefined operating system.")

        self.info3D_model.loadModel()

    def get_3Dinformation(self, image, calib_matrix, certainty_threshold = 0.5):
        """
        Calls trackFace method from shared library, which extracts FACE_INFO information
        :param image:
        :param calib_matrix:
        :param certainty_threshold:
        :return:
        """

        calib_matrix_np = np.array(calib_matrix).reshape((9, 1))
        self.info3D_model.trackFace.argtypes = [c_char_p, ndpointer(c_double, flags="C_CONTIGUOUS"), POINTER(FACE_INFO)]
        face_info = FACE_INFO(0.0, 0)
        success = self.info3D_model.trackFace(image.encode('utf-8'), calib_matrix_np, pointer(face_info))

        if success and face_info.certainty >= certainty_threshold:
            info3D = dict()

            # Landmark 2D
            info3D['landmarks2D'] = np.ndarray(buffer=landmarks2D_array.from_address(addressof(face_info.landmarks2D)),
                                     shape=(landmarks2D_size,)).reshape((2, landmarks_model_num)).transpose().copy()

            # Landmarks 3D
            info3D['landmarks3D'] = np.ndarray(buffer=landmarks3D_array.from_address(addressof(face_info.landmarks3D)),
                                     shape=(landmarks3D_size,)).reshape((3, landmarks_model_num)).transpose().copy()

            # Head pose
            info3D['R'] = np.ndarray(buffer=head_pose_array.from_address(addressof(face_info.head_rotation)),
                           shape=(pose_size,)).copy()

            # Head position
            info3D['T'] = np.ndarray(buffer=head_pose_array.from_address(addressof(face_info.head_position)),
                           shape=(pose_size,)).copy()

            # AUS intensity
            info3D['AUSint'] = np.ndarray(buffer=aus_intensity_array.from_address(addressof(face_info.aus_intensity)),
                                shape=(aus_intensity_size,)).copy()

            # AUS presence
            info3D['AUSpres'] = np.ndarray(buffer=aus_presence_array.from_address(addressof(face_info.aus_presence)),
                                 shape=(aus_presence_size,)).copy()

            return info3D

        print("WARNING from OpenFaceWrapper reading the input image or applying face model, or certainty obtained "
              "below threshold.")
        print("Please check image directory is correct.")
        return None
