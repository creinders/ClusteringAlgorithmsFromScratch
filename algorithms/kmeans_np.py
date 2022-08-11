import numpy as np
from algorithms.kmeans_base import KMeansBase
from algorithms.util_np import scatter_mean0


class KMeansNumpy(KMeansBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _main_loop(self, X, centers):

        for iteration in range(self.max_iter):
            distance = np.sum(np.square((X[:, :, None] - np.transpose(centers)[None, ...])), axis=1)
            assignments = np.argmin(distance, axis=1)

            # k-Means can assign no points to a cluster center, in that case keep old value
            center_means, assigned_counts = scatter_mean0(X, assignments, axis_size=self.n_clusters, return_counts=True)
            new_centers = np.copy(centers)
            new_centers[assigned_counts > 0] = center_means[assigned_counts > 0]

            diff = np.sum(np.square(new_centers - centers))

            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff))

            if diff < self.early_stop_threshold:
                break
            centers = new_centers

            if self.get_hook('on_iteration_end'):
                self.call_hook('on_iteration_end', self, iteration, X, centers, assignments)

        return centers, assignments
