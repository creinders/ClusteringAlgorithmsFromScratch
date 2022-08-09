import numpy as np
from .kmeans_base import KMeansBase
import jax
import jax.numpy as jnp
from jax import jit

@jit
def cluster_update(cluster_index, old_cluster, assignments, X):
    q = assignments == cluster_index
    mask = q.astype(jnp.int32)
    c = jnp.sum(mask)
    s = jnp.sum(X * mask[:, None], axis=0)
    m = s / c
    
    return m


@jit
def step(X, centers):
    cluster_update_vmap = jax.vmap(cluster_update, in_axes=(0, 0, None, None))

    distance = jnp.sum(jnp.square((X[:, :, None] - jnp.transpose(centers, (1, 0))[None, ...])), axis=1)
    assignments = jnp.argmin(distance, axis=1)

    a = jnp.arange(centers.shape[0])
    new_centers = cluster_update_vmap(a, centers, assignments, X)

    diff = jnp.sum(jnp.square((new_centers - centers)))
    return new_centers, diff, assignments


class KMeansJax(KMeansBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # jax.config.update('jax_platform_name', 'cpu')

    def prepare(self, X):
        centers = self.init_clusters(X, self.n_clusters)
        centers = jnp.asarray(centers) # K x C
        X = jnp.asarray(X) # B x C

        return X, centers


    def _main_loop(self, X, centers):
        
        @jit
        def while_step(arg):
            iteration, centers, assignments, diff = arg
            new_centers, diff, assignments = step(X, centers)

            return (iteration + 1, new_centers, assignments, diff)
        
        @jit
        def cond(arg):
            iteration, centers, assignments, diff = arg
            return (iteration < self.max_iter) & (diff > self.early_stop_threshold)

        assignments = jnp.zeros(X.shape[0], dtype=jnp.int32)

        iteration, centers, assignments, diff = jax.lax.while_loop(
            cond,
            while_step,
            (0, centers, assignments, 1000)
        )

        return centers, assignments

    def tensor_to_numpy(self, t):
        return np.array(t)
