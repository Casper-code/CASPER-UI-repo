
import wx
import time
from wx.core import Bitmap
from workers import WorkerThread
from workers import ImageProcessorQueue
from typing import Tuple


class ImageProcessor(WorkerThread):
    #start_time = time.time()

    def process_iteration(self):
        pass
        #print(f"Queue size: {ImageProcessorQueue.get_input_queue_size()}")
        #if int(time.time() - self.start_time) > 0:
        #    print(f"Size: {ImageProcessorQueue.get_input_queue_size()} "
        #          f"Framerate {int(ImageProcessorQueue.get_input_queue_size() / int(time.time() - self.start_time))}")

        buffer: bytes = ImageProcessorQueue.get_input_image()
        if buffer:
            bmp: Bitmap = self._get_bitmap()
            bmp.CopyFromBuffer(buffer, wx.BitmapBufferFormat_RGB)
            bmp.SaveFile("output.bmp", wx.BITMAP_TYPE_BMP)

    @classmethod
    def _get_bitmap(cls) -> Bitmap:
        return wx.Bitmap(*cls._get_screen_size())

    @classmethod
    def _get_screen_size(cls) -> Tuple[int, int]:
        screen = wx.ScreenDC()
        size = screen.GetSize()
        return size.width, size.height
