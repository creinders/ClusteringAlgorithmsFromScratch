from tabnanny import verbose
import numpy as np
from .base_algorithm import BaseAlgorithm


class KMeansBase(BaseAlgorithm):

    def __init__(self, n_clusters, max_iter=100, early_stop_threshold=0.01, seed = None, callback=None, verbose=False) -> None:
        super().__init__(callback=callback, verbose=verbose)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.early_stop_threshold = early_stop_threshold
        self.rng = np.random.RandomState(seed)

    def init_clusters(self, X, n_clusters):
        n = X.shape[0]

        i = self.rng.permutation(n)[:n_clusters]
        centers = X[i]
        return centers

    def prepare(self, X):
        centers = self.init_clusters(X, self.n_clusters)
        return X, centers

    def _main_loop(self, X, centers):
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
