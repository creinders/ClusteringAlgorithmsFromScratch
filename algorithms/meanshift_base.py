import numpy as np
from algorithms.base_algorithm import BaseAlgorithm


class MeanShiftBase(BaseAlgorithm):

    def __init__(self, bandwidth, early_stop_threshold=0.01, cluster_threshold=0.1, verbose=False, callback=None) -> None:
        super().__init__(callback=callback, verbose=verbose)
        self.bandwidth = bandwidth
        self.early_stop_threshold = early_stop_threshold
        self.cluster_threshold = cluster_threshold

    def prepare(self, X):
        return X, X.copy()

    def _main_loop(self, X, clusters):        
        pass
    
    def tensor_to_numpy(self, t):
        return np.array(t)

    def finalize(self, centers, assignments):
        return self.tensor_to_numpy(centers), self.tensor_to_numpy(assignments)
    
    def fit(self, X):
        X, initial_centers = self.prepare(X)

        self.call_hook('on_main_loop_start')
        centers, assignments = self._main_loop(X, initial_centers)
        self.call_hook('on_main_loop_end')

        centers, assignments = self.finalize(centers, assignments)

        self.call_hook('on_epoch_end', self, X, centers, assignments)

        return centers, assignments
