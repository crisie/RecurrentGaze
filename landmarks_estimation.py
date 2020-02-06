import face_alignment


class LandmarksEstimation(object):
    """
    LandmarksEstimation class. Wrapper of Bulat et al 2017 method
    https://github.com/1adrianb/face-alignment
    """
    def __init__(self, model_type=face_alignment.LandmarksType._3D, device='gpu'):
        if device == 'gpu':
            self.landmarks_model = face_alignment.FaceAlignment(model_type, flip_input=False, device='cuda')
        elif device == 'cpu':
            self.landmarks_model = face_alignment.FaceAlignment(model_type, flip_input=False, device='cpu')
        else:
            raise RuntimeError("Device type not found")

    # TODO: check if gpu/cpu selector works properly
    @classmethod
    def get_model(cls):
        return LandmarksEstimation.landmarks_model
