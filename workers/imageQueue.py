from wx.core import Image
from typing import Optional
from queue import Queue, Empty


class ImageProcessorQueue(object):
    _pre_processing: Optional[Queue] = None
    _post_processing: Optional[Queue] = None

    @classmethod
    def put_input_image(cls, image: bytes):
        cls._get_pre_process_queue().put_nowait(image)

    @classmethod
    def get_input_image(cls):
        try:
            return cls._get_pre_process_queue().get_nowait()
        except Empty:
            return None

    @classmethod
    def get_input_queue_size(cls) -> int:
        return cls._get_pre_process_queue().qsize()

    @classmethod
    def _get_pre_process_queue(cls) -> Queue:
        if cls._pre_processing is None:
            cls._pre_processing = Queue()
        return cls._pre_processing

    @classmethod
    def _get_post_process_queue(cls) -> Queue:
        if cls._post_processing is None:
            cls._post_processing = Queue()
        return cls._post_processing
