import wx
import sys
import time
import signal
from typing import List, Optional
from workers import WorkerFactory, WorkerThread
from wx import PaintEvent


class DetectRepaint(wx.Window):

    pass


class Event(PaintEvent):
    pass


def main():

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
