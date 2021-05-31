from typing import List

from textDetection.TextDetection import TextDetection
from workers.ImageProcessor import ImageProcessor
from workers.ScreenCapturer import ScreenCapturer
from workers.UIReactor import UIReactor
from workers.imageQueue import ImageProcessorQueue
from workers.worker import WorkerThread


class WorkerFactory(object):
    _workers: List[WorkerThread] = None

    @classmethod
    def get_workers(cls) -> List[WorkerThread]:
        if cls._workers is None:
            cls._workers = [ScreenCapturer(), ImageProcessor(), UIReactor(), TextDetection()]
        return cls._workers
