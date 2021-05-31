from typing import Tuple

import numpy as np
import wx
from cv2 import cv2
from tensorflow.keras.models import load_model
from wx.core import Bitmap

from utils import img_crop
from workers.imageQueue import ImageProcessorQueue
from workers.worker import WorkerThread

THRESHOLD = 0.35
CONFIDENCE = 0.10
CLASSIFICATION_CONFIDENCE = .4
SIZE = (224, 224)
MODEL = load_model('keras_m3.h5')
WEIGHTS_PATH = "CASPER3_tiny_l_best.weights"
CONFIG_PATH = "CASPER3_tiny_l_test.cfg"
LABELS = open("casper.names").read().strip().split("\n")


class ImageProcessor(WorkerThread):
    LAST_BUFFER = None
    IMAGE_CACHE = []

    def __init__(self, *args, **kwargs):
        print(LABELS)
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        ImageProcessor.LAST_BUFFER = None
        ImageProcessor.IMAGE_CACHE = []
        super(ImageProcessor, self).__init__(*args, **kwargs)

    def process_iteration(self):
        pass
        buffer: bytes = ImageProcessorQueue.get_input_image()
        if buffer:
            bmp: Bitmap = self._get_bitmap()
            bmp.CopyFromBuffer(buffer)

            img = bmp.ConvertToImage()
            arr = np.asarray(img.GetDataBuffer())
            image = np.copy(np.reshape(arr, (img.GetHeight(), img.GetWidth(), 3)))

            (H, W) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (224, 224),
                                         swapRB=False, crop=False)

            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.ln)

            rez = self.detect(image, layer_outputs, H, W)
            ImageProcessorQueue.array = rez

    @classmethod
    def _get_bitmap(cls) -> Bitmap:
        return wx.Bitmap(*cls._get_screen_size())

    @classmethod
    def _get_screen_size(cls) -> Tuple[int, int]:
        screen = wx.ScreenDC()
        size = screen.GetSize()
        return size.width, size.height

    @staticmethod
    def detect(image, layer_outputs, H, W):
        boxes = []
        confidences = []
        class_ids = []
        results = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                box = boxes[i]
                (x, y) = box[0], box[1]
                (w, h) = box[2], box[3]
                crop = img_crop(image, x, y, x + w, y + h)
                img = cv2.resize(crop, SIZE)
                input_image = np.expand_dims(np.array(img) / 255, axis=0)

                if not is_duplicate(results, [x, y, w, h]) and MODEL.predict(input_image)[0][
                    0] > CLASSIFICATION_CONFIDENCE:
                    results.append([x, y, w, h])

        return results


def byte_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])


def is_duplicate(list, new_item):
    for item in list:
        if item[0] == new_item[0] and item[1] == new_item[1] and item[2] == new_item[2] and item[3] == new_item[3]:
            return True
    return False
