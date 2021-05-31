from typing import Tuple

import wx
from wx.core import Bitmap, MemoryDC

from workers.imageQueue import ImageProcessorQueue
from workers.worker import WorkerThread


class UIReactor(WorkerThread):

    def __init__(self, *args, **kwargs):
        self.background = None
        self.bitmap = None
        self.memory_dc = None
        super(UIReactor, self).__init__(*args, **kwargs)

    def process_iteration(self):
        if self.memory_dc is None:
            if self.bitmap is None:
                screen_width, screen_height = self._get_screen_size()
                self.bitmap = self._get_bitmap()
                self.memory_dc = self._get_memory_device_context(self.bitmap)
                self.memory_dc.SetBrush(wx.Brush(wx.WHITE, wx.SOLID))
                self.memory_dc.DrawPolygon(((0, 0), (screen_width, 0),
                                            (screen_width, screen_height), (0, screen_height)))
        if self.memory_dc:
            boxes = ImageProcessorQueue.array
            dc = wx.ScreenDC()
            if boxes:
                for box in boxes:
                    dc.Blit(
                        box[0], box[1],
                        box[2], box[3],
                        self.memory_dc,
                        0, 0
                    )

    @staticmethod
    def _get_memory_device_context(bmp: Bitmap) -> MemoryDC:
        memory_dc = wx.MemoryDC()
        memory_dc.SelectObject(bmp)
        return memory_dc

    @classmethod
    def _get_bitmap(cls) -> Bitmap:
        return wx.Bitmap(*cls._get_screen_size())

    @classmethod
    def _get_screen_size(cls) -> Tuple[int, int]:
        screen = wx.ScreenDC()
        size = screen.GetSize()
        return size.width, size.height
