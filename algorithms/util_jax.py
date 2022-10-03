import jax
import jax.numpy as jnp
from jax import jit

def connected_components_undirected(conn):
    """
    Find connected components in undirected graph connectivity matrix
    """

    assert conn.dtype == jnp.bool_
    assert len(conn.shape) == 2
    assert jnp.all(conn == conn.T)

    result = jnp.full(conn.shape[:1], -1, dtype=jnp.int32)

    curr_idx = 0

    n = conn.shape[0]

    for i in range(n):
        # Find next unassigned

        if result[i] >= 0:
            continue

        mask = jnp.zeros(len(conn), dtype=jnp.bool_)
        mask = mask.at[i].set(True)
        mask_sum = 1

        while True:
            mask = jnp.logical_or(mask, jnp.any(conn[mask], axis=0))
            s = jnp.sum(mask).item()

            if s == mask_sum:
                break
            mask_sum = s

        assert jnp.all(result[mask] == -1)
        result = result.at[mask].set(curr_idx)

        curr_idx += 1

    assert jnp.all(result > -1)
    return curr_idx, result


@jit
def calculate_mean(cluster_index, assignments, X):
    q = assignments == cluster_index
    mask = q.astype(jnp.int32)
    c = jnp.sum(mask)
    s = jnp.sum(X * mask[:, None], axis=0)
    m = s / c
    
    return m


def scatter_mean0(src, index):
    """
    Scatter mean on 0-th axis
    """

    index_max = jnp.max(index) + 1
    # count = jnp.zeros(index_max, dtype=jnp.int32)
    # count = count.at[index].add(1)
    
    calculate_mean_vmap = jax.vmap(calculate_mean, in_axes=(0, None, None))
    indices = jnp.arange(index_max)
    clusters = calculate_mean_vmap(indices, index, src)
    
    return clusters

