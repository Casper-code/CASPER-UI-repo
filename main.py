from typing import List, Optional

import wx
from wx import PaintEvent

from workers import WorkerFactory, WorkerThread


class DetectRepaint(wx.Window):
    pass


class Event(PaintEvent):
    pass


def main():
    app = wx.App()
    # TODO Code here
    workers: Optional[List[WorkerThread]] = WorkerFactory.get_workers()

    for worker in workers:
        worker.start()

    command: str = input("enter command")
    while command != 'stop':
        command: str = input("enter command")

    for worker in workers:
        worker.stop()
        worker.join()


if __name__ == '__main__':
    main()
