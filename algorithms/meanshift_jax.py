import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from algorithms.meanshift_base import MeanShiftBase


class MeanShiftJax(MeanShiftBase):

    def __init__(self, cuda=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not cuda:
            jax.config.update('jax_platform_name', 'cpu')

    def prepare(self, X):
        X = jnp.asarray(X)
        return X, jnp.copy(X)

    @partial(jit, static_argnums=0)
    def distance(self, a, b):
        # shape a: N x C
        # shape b: M x C

        d = jnp.sum(jnp.square(a[:,  None, :] - b[None, :, :]), axis=-1)
        return d

    @partial(jit, static_argnums=0)
    def kernel(self, distances):
        return jnp.exp(-0.5 * ((distances / self.bandwidth ** 2)))

    def _main_loop(self, X, clusters):

        @jit
        def step(arg):
            clusters, diff = arg
            d = self.distance(clusters, X)
            w = self.kernel(d)

            new_centers = w[:, :, None] * X[None, :, :]
            w_sum = w.sum(1)
            new_centers = jnp.sum(new_centers, axis=1) / w_sum[:, None]

            diff = jnp.sum(jnp.square(new_centers - clusters))

            return (new_centers, diff)
        
        @jit
        def cond(arg):
            clusters, diff = arg
            return diff > self.early_stop_threshold

        clusters, diff = jax.lax.while_loop(
            cond,
            step,
            (clusters, 1000)
        )

        clusters, assignments = self._group_clusters(clusters, self.cluster_threshold)
        return clusters, assignments

    def _group_clusters(self, points, t):
        cluster_ids = []
        cluster_centers = []

        for point in points:
            add = True
            for cluster_index, cluster in enumerate(cluster_centers):
                dist = jnp.sum(jnp.square(point  - cluster), axis=-1)
                if dist < t:
                    cluster_ids.append(cluster_index)
                    add = False
                    break

            if add:
                cluster_ids.append(len(cluster_centers))
                cluster_centers.append(point)

        cluster_centers = jnp.stack(cluster_centers, axis=0)
        cluster_ids = jnp.array(cluster_ids)
        return cluster_centers, cluster_ids
    
    def tensor_to_numpy(self, t):
        return np.array(t)
