import numpy as np


def connected_components_undirected(conn):
    """
    Find connected components in undirected graph connectivity matrix
    """

    assert conn.dtype == bool
    assert len(conn.shape) == 2
    assert np.all(conn == conn.T)

    result = np.full(len(conn), -1, dtype=int)

    curr_idx = 0

    while True:
        # Find next unassigned
        cand = np.flatnonzero(result == -1)

        if len(cand) == 0:
            break

        mask = np.zeros(len(conn), dtype=bool)
        mask[cand[0]] = True
        mask_sum = 1

        while True:
            mask = np.logical_or(mask, np.any(conn[mask], axis=0))
            s = np.sum(mask)

            if s == mask_sum:
                break
            mask_sum = s

        assert np.all(result[mask] == -1)
        result[mask] = curr_idx

        curr_idx += 1

    assert np.all(result > -1)
    return curr_idx, result


def scatter_mean0(src, index, axis_size=None, return_counts=False):
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

    result = accumulator / numerator.reshape(axis_size, *((1,) * len(src.shape[1:])))

    if return_counts:
        return result, numerator
    else:
        return result
