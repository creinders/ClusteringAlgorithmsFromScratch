import numpy as np
from algorithms.meanshift_base import MeanShiftBase


class MeanShiftNumpy(MeanShiftBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare(self, X):
        return X, X.copy()

    def distance(self, a, b):
        # shape a: N x C
        # shape b: M x C

        d = np.sum(np.square(a[:,  None, :] - b[None, :, :]), axis=-1)
        return d

    def kernel(self, distances):
        return np.exp(-0.5 * ((distances / self.bandwidth ** 2)))

    def _main_loop(self, X, clusters):        
        
        iteration = 0
        while True:
            iteration += 1

            d = self.distance(clusters, X)
            w = self.kernel(d)

            new_centers = w[:, :, None] * X[None, :, :]
            w_sum = w.sum(1)
            new_centers = np.sum(new_centers, axis=1) / w_sum[:, None]

            diff = np.sum(np.square(new_centers - clusters))
            
            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff.item()))
            
            if diff < self.early_stop_threshold:
                break
            clusters = new_centers

            if self.get_hook('on_iteration_end'):
                _, plot_assignments = self._group_clusters(clusters)
                self.call_hook('on_iteration_end', self, iteration, X, clusters, plot_assignments)

        clusters, assignments = self._group_clusters(clusters)
        return clusters, assignments

    def _group_clusters(self, points):
        cluster_ids = []
        cluster_centers = []

        for point in points:
            add = True
            for cluster_index, cluster in enumerate(cluster_centers):
                dist = np.sum(np.square(point  - cluster), axis=-1)
                if dist < self.cluster_threshold:
                    cluster_ids.append(cluster_index)
                    add = False
                    break

            if add:
                cluster_ids.append(len(cluster_centers))
                cluster_centers.append(point)
        cluster_centers = np.stack(cluster_centers, axis=0)
        return cluster_centers, cluster_ids

