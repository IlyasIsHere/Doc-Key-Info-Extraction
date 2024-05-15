import tempfile
import cv2
import numpy as np
from PIL import Image


class OCRPreprocessor:
    """
    A class for preprocessing images before optical character recognition (OCR).

    This class provides methods to resize images, remove noise, and smoothen images
    to enhance OCR accuracy.

    Attributes:
        image_size (int): The desired size (in pixels) for the resized image. Default is 1800.
        binary_threshold (int): The threshold value for binary image conversion. Default is 180.
    """

    
    def __init__(self, image_size=1800, binary_threshold=180):
        """
        Initializes the OCRPreprocessor with default or user-defined parameters.

        Args:
            image_size (int, optional): The desired size (in pixels) for the resized image.
                Defaults to 1800.
            binary_threshold (int, optional): The threshold value for binary image conversion.
                Defaults to 180.
        """
        self.image_size = image_size
        self.binary_threshold = binary_threshold

    def process_image_for_ocr(self, file_path):
        temp_filename = self._set_image_dpi(file_path)
        im_new = self._remove_noise_and_smooth(temp_filename)
        return im_new

    def _set_image_dpi(self, file_path):
        try:
            im = Image.open(file_path)
            length_x, width_y = im.size
            factor = max(1, int(self.image_size / length_x))
            size = factor * length_x, factor * width_y
            im_resized = im.resize(size, Image.LANCZOS)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_filename = temp_file.name
            im_resized.save(temp_filename, dpi=(300, 300))
            return temp_filename
        except Exception as e:
            print("Error processing image:", e)
            return None

    def _image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img, self.binary_threshold, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def _remove_noise_and_smooth(self, file_name):
        try:
            img = cv2.imread(file_name, 0)
            filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            img = self._image_smoothening(img)
            or_image = cv2.bitwise_or(img, closing)
            return or_image
        except Exception as e:
            print("Error removing noise and smoothening:", e)
            return None