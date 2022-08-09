

class BaseAlgorithm:

    def __init__(self, callback=None, verbose=False) -> None:
        self.verbose = verbose
        self.callback = callback

    def get_hook(self, name):
        return getattr(self.callback, name, None)

    def call_hook(self, name, *args, **kwargs):
        callback_op = self.get_hook(name)
        if callable(callback_op):
            callback_op(*args, **kwargs)