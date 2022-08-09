from callbacks.callback import DefaultCallback
import time

class TimingCallback(DefaultCallback):

    def __init__(self) -> None:
        super().__init__()

        self.start = None
        self.n = 0
        self.total_duration = 0

    
    def on_main_loop_start(self):
        self.start = time.time()

    def on_main_loop_end(self):
        end = time.time()
        duration = end - self.start

        self.n += 1
        self.total_duration += duration
