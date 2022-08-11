import numpy as np
from .kmeans_base import KMeansBase


class KMeansNumpy(KMeansBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _scatter_mean0(src, index, axis_size=None):
        """
        Scatter mean on 0-th axis
        """

        if axis_size is None:
            axis_size = np.max(index) + 1

        # Target shape is target size and remaining value shape without indexed dimension
        accumulator = np.zeros((axis_size,) + src.shape[1:], dtype=src.dtype)
        numerator = np.zeros(axis_size, dtype=int)

        np.add.at(accumulator, index, src)
        np.add.at(numerator, index, 1)
        return accumulator / numerator.reshape(axis_size, *((1,) * len(src.shape[1:])))

    def _main_loop(self, X, centers):

        for iteration in range(self.max_iter):
            distance = np.sum(np.square((X[:, :, None] - np.transpose(centers)[None, ...])), axis=1)
            assignments = np.argmin(distance, axis=1)

            new_centers = KMeansNumpy._scatter_mean0(X, assignments, axis_size=self.n_clusters)
            diff = np.sum(np.square(new_centers - centers))

            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff))

            if diff < self.early_stop_threshold:
                break
            centers = new_centers

            if self.get_hook('on_iteration_end'):
                self.call_hook('on_iteration_end', self, iteration, X, centers, assignments)

        return centers, assignments
