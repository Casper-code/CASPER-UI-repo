import time
from abc import abstractmethod
from threading import Thread


class WorkerThread(Thread):

    def __init__(self, *args, **kwargs):
        self.working = True
        super(WorkerThread, self).__init__(*args, **kwargs)

    @abstractmethod
    def process_iteration(self):
        pass

    def run(self):

        while self.working:
            self.process_iteration()
            time.sleep(1/1000)

    def stop(self):
        self.working = False

