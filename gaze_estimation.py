from landmarks_estimation import *
from info3D_estimation import *
import string
from skimage.io import imsave
from Test import *
imageTuple = namedtuple('DataE', 'file, image')
import tempfile

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class modelType(Enum):
    StaticMultimodal = 'NFEL5836_2918'
    RecurrentMultimodal = 'NFEL5836GRU' # currently not working


class GazeEstimation(object):
    face_roi_size = [250, 250]
    eyes_roi_size = [70, 58]
    dummy_gaze = [[0.0], [0.0], [-1.0]]

    def __init__(self, model_type: modelType = modelType.StaticMultimodal,
                 device: str ='gpu', info3D_lib_path=""):
        """
        Initialize gaze estimation model
        :param model_type: Gaze estimation model (only NFEL5836_2918 currently available)
        :param device: gpu or cpu
        :param info3D_lib_path: path to directory where OpenFaceWrapper shared library and related models/libraries are located
        """
        if device == 'gpu':
            GPU = True
        elif device == 'cpu':
            GPU = False
        else:
            raise RuntimeError("Device type not found")

        if GPU:
            num_GPU = 1
            num_CPU = 1
        else:
            num_CPU = 1
            num_GPU = 0

        config = tf.ConfigProto(device_count={'CPU': num_CPU, 'GPU': num_GPU})
        session = tf.Session(config=config)
        K.set_session(session)

        self.batch_size = 1
        self.landmarks_model = LandmarksEstimation(device=device).get_model()
        self.gaze_model = ExperimentHelper.get_experiment(model_type.value)
        self.info3D_model = Info3DEstimation(lib_path=info3D_lib_path)

        dummy = DataTuple(x=[], y=[], feats=[], idxs=[], parts=None)
        self.gaze_model.init_data_gen_val(dummy, self.batch_size, None, False, False)  # , True)
        self.gaze_model.load_model()

        self.gaze_memory_window = 5
        self.gaze_memory_counter = 0
        self.last_prediction = GazeEstimation.dummy_gaze

    def compute_gaze(self, image_path, calib_matrix = None, draw=False):
        """
        Computes gaze of input image
        :param image: path to image
        :param calib_matrix: camera calibration matrix
        :param draw: Plot image with predicted vector overlaid if True
        :return: predicted 3D gaze vector
        """
        # Load image
        image = self.gaze_model.load_image(image_path)
        return self.compute_gaze_(image, calib_matrix, draw)

    def compute_gaze_(self, image, calib_matrix = None, draw=False):
        """
        Computes gaze of input image
        :param image: Loaded image
        :param calib_matrix: camera calibration matrix
        :param draw: Plot image with predicted vector overlaid if True
        :return: predicted 3D gaze vector
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, id_generator() + ".bmp")
            imsave(image_path, image)
            image_data = [imageTuple(file=image_path, image=image)]

            success = self.preprocess_image(image_data, calib_matrix)
            if success:
                prediction = self.predict_gaze()
                self.last_prediction = prediction
                self.gaze_memory_counter = 0
            elif not success and self.gaze_memory_counter < self.gaze_memory_window:
                prediction = self.last_prediction
                self.gaze_memory_counter += 1
            else:
                prediction = GazeEstimation.dummy_gaze

            if draw and success:
                img = load_image(self.gaze_model.val_data_generator.data[0][self.gaze_model.label_pos])
                projected_gaze = project_gaze(np.array(
                    list(self.gaze_model.val_data_generator.feats[0][self.gaze_model.label_pos][81])),
                    prediction[:, 0], np.array(calib_matrix['intrinsics']))
                fig, ax = plt.subplots()
                ax.imshow(img / 255)
                ax.plot(projected_gaze[:, 0], projected_gaze[:, 1], '-', linewidth=3, color='firebrick')
                plt.show()

        return prediction

    def preprocess_image(self, data: imageTuple, calib_matrix: np.array = None):
        """
        Preprocess the image and loads the extracted features and image to the data generator.
        Preprocessing includes:
        (1) calling OpenFace DLL to extract 3D face landmarks, rotation, and translation;
        (2) Calling Bulat et al code to extract "3D" landmarks.
        (3) Processing features from 1 and 2 to compute image normalization matrices compatible with gaze
        estimation network.
        Preprocessing may fail if Openface fails to extract correct 3D information from the face.
        :param data: temporary image file in imageTuple format
        :param calib_matrix: 3x3 camera calibration matrix, if none is passed, a dummy matrix is created
        :return: True if preprocessing has been successful, False otherwise.
        """

        if not calib_matrix:
            calib_matrix = dummy_calib(data[0].image.shape[1], data[0].image.shape[0])

        info3D = self.info3D_model.get_3Dinformation(data[0].file, calib_matrix['intrinsics'])
        if info3D is not None:

            face_info = compute_face_info(info3D['landmarks2D'], info3D['landmarks3D'], info3D['R'],
                                       GazeEstimation.face_roi_size, GazeEstimation.eyes_roi_size,
                                       calib_matrix)

            landmarks2D = self.landmarks_model.get_landmarks(data[0].file,
                             detected_faces=[[face_info['face']['bb'][0], face_info['face']['bb'][1],
                                             face_info['face']['bb'][0] + face_info['face']['bb'][2],
                                             face_info['face']['bb'][1] + face_info['face']['bb'][3]]])

            face_features = convert_face_info_to_features(face_info, landmarks2D[0],
                                                          GazeEstimation.face_roi_size, GazeEstimation.eyes_roi_size)

            gt = [[1.0, 1.0, 1.0] for i in range(len(data))]  # dummy GT for compatibility
            validation, _ = train_valtest_split([data[0].file], gt, [face_features], [])
            _, validation, _ = self.gaze_model.prepare_data(None, validation, {'max_look_back': 1}, train=False)
            self.gaze_model.val_data_generator.update_data(validation.x, validation.y, validation.feats)

            return True

        return False

    def predict_gaze(self):
        """
        Predict gaze vector for each frame that has been passed to val_data_generator, and denormalize result
        :return: predicted 3D gaze vector
        """

        input_x, _ = self.gaze_model.val_data_generator.__getitem__(0)
        normalized_predictions = self.gaze_model.model.predict(input_x)
        frame_feats = copy_face_features(self.gaze_model.val_data_generator.feats[0][self.gaze_model.label_pos])
        gaze_conv = get_gaze_conv(frame_feats)
        predictions = denormalize_gaze(gaze_conv, normalized_predictions)

        return predictions
