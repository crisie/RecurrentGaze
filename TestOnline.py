import gaze_estimation
import os

if __name__ == '__main__':

    # Path to OpenFaceWrapper shared library and models
    openfacewrapper_path = "C://Users//Documents//OpenFace//x64//Release"

    # Load models - change to device = 'gpu' to run with cuda support
    ga = gaze_estimation.GazeEstimation(device='cpu', info3D_lib_path=openfacewrapper_path)

    dir = os.path.dirname(__file__)
    imgs_path = "Test//images" # change path
    img_files = os.listdir(os.path.join(dir, imgs_path))
    calib_matrix = dict()
    # Calibration matrix. If not given, gaze_estimation module computes a dummy version
    calib_matrix['intrinsics'] = [[1.2*720.0, 0.0, 360.0],
                                  [0.0, 1.2*720.0, 288.0],
                                  [0.0, 0.0, 1.0]]
    for file in img_files:
        img_path = os.path.join(dir, imgs_path, file)

        # Gaze inference
        gaze = ga.compute_gaze(img_path, calib_matrix = calib_matrix, draw=True)
        print(gaze)



