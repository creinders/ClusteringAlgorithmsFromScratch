import numpy as np
from .kmeans_base import KMeansBase


class KMeansNumpy(KMeansBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _main_loop(self, X, centers):

        for iteration in range(self.max_iter):
            distance = np.sum(np.square((X[:, :, None] - np.transpose(centers)[None, ...])), axis=1)
            assignments = np.argmin(distance, axis=1)

            new_centers = np.array([X[assignments == i].mean(0) for i in range(self.n_clusters)])
            
            diff = np.sum(np.square(new_centers - centers))

            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff))
            
            if diff < self.early_stop_threshold:
                break
            centers = new_centers
            
            if self.get_hook('on_iteration_end'):
                self.call_hook('on_iteration_end', self, iteration, X, centers, assignments)

        return centers, assignments
