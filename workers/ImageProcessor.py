import wx
from wx.core import Bitmap
from workers import WorkerThread
from workers import ImageProcessorQueue
from typing import Tuple
from tensorflow.keras.models import load_model
from cv2 import cv2
import numpy as np
from utils import img_crop

THRESHOLD = 0.45
CONFIDENCE = 0.25
CLASSIFICATION_CONFIDENCE = .4
SIZE = (200, 200)
MODEL = load_model('imgenet200x200.h5')
WEIGHTS_PATH = "casper_11000.weights"
CONFIG_PATH = "casper.cfg"
LABELS = open("casper.names").read().strip().split("\n")


class ImageProcessor(WorkerThread):

    def __init__(self, *args, **kwargs):
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        super(ImageProcessor, self).__init__(*args, **kwargs)

    def process_iteration(self):
        pass

        buffer: bytes = ImageProcessorQueue.get_input_image()
        if buffer:
            bmp: Bitmap = self._get_bitmap()
            bmp.CopyFromBuffer(buffer, wx.BitmapBufferFormat_RGB)

            img = wx.ImageFromBitmap(bmp)
            arr = np.asarray(img.GetDataBuffer())
            image = np.copy(np.reshape(arr, (img.GetHeight(), img.GetWidth(), 3)))

            (H, W) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (224, 224),
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.ln)
            rez = self.detect(image, layer_outputs, H, W)
            ImageProcessorQueue.put_processed_image(rez)

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
                if LABELS[int(class_ids[i])].startswith('image'):
                    box = boxes[i]
                    (x, y) = box[0], box[1]
                    (w, h) = box[2], box[3]
                    crop = img_crop(image, x, y, x + w, y + h)
                    img = cv2.resize(crop, SIZE)
                    input_image = np.expand_dims(np.array(img) / 255, axis=0)
                    if MODEL.predict(input_image)[0][0] > CLASSIFICATION_CONFIDENCE:
                        results.append([x, y, w, h])

        return results
