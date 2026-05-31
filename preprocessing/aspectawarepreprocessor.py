
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW, dH = 0, 0

        if w < h:
            r = self.width / float(w)
            dim = (self.width, int(h * r))
            image = cv2.resize(image, dim, interpolation=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            r = self.height / float(h)
            dim = (int(w * r), self.height)
            image = cv2.resize(image, dim, interpolation=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

