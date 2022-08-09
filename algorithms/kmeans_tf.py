import numpy as np
import tensorflow as tf

from .kmeans_base import KMeansBase


class KMeansTensorflow(KMeansBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    
    def prepare(self, X):
        centers = self.init_clusters(X, self.n_clusters)
        centers = tf.Variable(centers) # K x C
        X = tf.convert_to_tensor(X) # B x C

        return X, centers

    def _main_loop(self, X, centers):

        tf.function
        def step(iteration, centers, assignments, diff):
            distance = tf.reduce_sum(tf.square((tf.expand_dims(X, axis=2) - tf.expand_dims(tf.transpose(centers, perm=(1, 0)), axis=0))), axis=1)
            assignments = tf.math.argmin(distance, axis=1)

            new_centers = []
            for i in range(self.n_clusters):
                new_centers.append(tf.reduce_mean(X[assignments == i], axis=0))

            new_centers = tf.stack(new_centers, axis=0)

            diff = tf.reduce_sum(tf.square((new_centers - centers)))

            return iteration + 1, new_centers, assignments, diff
        
        c = tf.constant(self.early_stop_threshold, dtype=X.dtype)

        tf.function
        def cond(iteration, centers, assignments, diff):
            return iteration < self.max_iter and tf.math.greater(diff, c)

        iteration, centers, assignments, diff = tf.while_loop(cond, step, [0, centers, None, 100])
        return centers, assignments

    def tensor_to_numpy(self, t):
        return np.array(t)
