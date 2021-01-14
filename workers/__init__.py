from typing import List
from workers.imageQueue import ImageProcessorQueue
from workers.worker import WorkerThread
from workers.ScreenCapturer import ScreenCapturer
from workers.ImageProcessor import ImageProcessor
from workers.UIReactor import UIReactor


class WorkerFactory(object):

    _workers: List[WorkerThread] = None

    @classmethod
    def get_workers(cls) -> List[WorkerThread]:
        if cls._workers is None:
            cls._workers = [ScreenCapturer(), ImageProcessor(), UIReactor()]
        return cls._workers
