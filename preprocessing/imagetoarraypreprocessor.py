
import torch
import numpy as np

class ImageToTensorPreprocessor:
    def __init__(self, data_format='channels_first'):
        """
        data_format: 'channels_first' (C, H, W) or 'channels_last' (H, W, C)
        """
        assert data_format in ['channels_first', 'channels_last'], "data_format must be 'channels_first' or 'channels_last'"
        self.data_format = data_format

    def preprocess(self, image):

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if self.data_format == 'channels_first':

            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
        elif self.data_format == 'channels_last':

            pass


        image = image.astype(np.float32) / 255.0


        return torch.from_numpy(image)
