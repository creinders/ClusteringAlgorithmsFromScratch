import numpy as np
import tensorflow as tf
import math

from algorithms.meanshift_base import MeanShiftBase


class MeanShiftTensorflowEager(MeanShiftBase):

    def __init__(self, cuda=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cuda = cuda
        self.bandwidth = tf.constant(self.bandwidth)
        self.pi = tf.constant(math.pi)

    def prepare(self, X):
        X = tf.convert_to_tensor(X)
        return X, tf.identity(X)

    def distance(self, a, b):
        # shape a: N x C
        # shape b: M x C

        d = tf.reduce_sum(tf.square(tf.expand_dims(a, 1) - tf.expand_dims(b, 0)), axis=-1)
        return d

    def kernel(self, distances):
        return tf.exp(-0.5 * ((distances / self.bandwidth ** 2)))

    def _main_loop(self, X, clusters):

        iteration = 0
        while True:
            iteration += 1
            d = self.distance(clusters, X)
            w = self.kernel(d)

            new_centers = w[:, :, tf.newaxis] * X[tf.newaxis, :, :]
            w_sum = tf.reduce_sum(w, axis=1)
            new_centers = tf.reduce_sum(new_centers, axis=1) / w_sum[:, None]

            diff = tf.reduce_sum(tf.square(new_centers - clusters))

            if self.verbose:
                print('Iteration {}: {:.5f} difference'.format(iteration, float(diff)))

            if diff < self.early_stop_threshold:
                break
            clusters = new_centers

        clusters, assignments = self._group_clusters(clusters)
        return clusters, assignments

    def _group_clusters(self, points):
        cluster_ids = []
        cluster_centers = []

        for point in points:
            add = True
            for cluster_index, cluster in enumerate(cluster_centers):
                dist = tf.reduce_sum(tf.square(point  - cluster), axis=-1)
                if dist < self.cluster_threshold:
                    cluster_ids.append(cluster_index)
                    add = False
                    break

            if add:
                cluster_ids.append(len(cluster_centers))
                cluster_centers.append(point)

        cluster_centers = tf.stack(cluster_centers, axis=0)
        cluster_ids = tf.convert_to_tensor(cluster_ids)
        return cluster_centers, cluster_ids
    
    def tensor_to_numpy(self, t):
        return np.array(t)
