import time
import wx
import numpy
import array
import io
from wx.core import DC, Bitmap, MemoryDC, Image
from workers import WorkerThread
from workers import ImageProcessorQueue
from typing import Tuple


class ScreenCapturer(WorkerThread):
    app = wx.App(False)

    def process_iteration(self):
        ImageProcessorQueue.put_input_image(self.take_screenshot())

    @classmethod
    def take_screenshot(cls) -> bytes:

        #start_time = time.time()
        bmp: Bitmap = cls._get_bitmap()
        memory_device_context: MemoryDC = cls._get_memory_device_context(bmp)

        cls._copy_device_context(wx.ScreenDC(), memory_device_context, *cls._get_screen_size())

        memory_device_context.SelectObject(wx.NullBitmap)

        width, height = cls._get_screen_size()

        buffer = bytes(width * height * 3)
        bmp.CopyToBuffer(buffer, wx.BitmapBufferFormat_RGB)

        #if time.time() - start_time > 0:
        #    print(f"-----1 {1 / (time.time() - start_time)}")

        return buffer

    @classmethod
    def _get_bitmap(cls) -> Bitmap:
        return wx.Bitmap(*cls._get_screen_size())

    @classmethod
    def _get_screen_size(cls) -> Tuple[int, int]:
        screen = wx.ScreenDC()
        size = screen.GetSize()
        return size.width, size.height

    @staticmethod
    def _get_memory_device_context(bmp: Bitmap) -> MemoryDC:
        memory_dc = wx.MemoryDC()
        memory_dc.SelectObject(bmp)
        return memory_dc

    @staticmethod
    def _copy_device_context(source_dc: DC, destination_dc: DC, width, height):
        destination_dc.Blit(
            0, 0,
            width, height,
            source_dc,
            0, 0,
        )
