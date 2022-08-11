import torch


def connected_components_undirected(conn):
    """
    Find connected components in undirected graph connectivity matrix
    """

    assert conn.dtype == torch.bool
    assert len(conn.shape) == 2
    assert torch.all(conn == conn.T)

    result = torch.full(conn.shape[:1], -1, dtype=torch.int64, device=conn.device)

    curr_idx = 0

    while True:
        # Find next unassigned
        cand = torch.nonzero(result == -1).flatten()

        if len(cand) == 0:
            break

        mask = torch.zeros(len(conn), dtype=torch.bool, device=conn.device)
        mask[cand[0]] = True
        mask_sum = 1

        while True:
            mask = torch.logical_or(mask, torch.any(conn[mask], dim=0))
            s = torch.sum(mask).item()

            if s == mask_sum:
                break
            mask_sum = s

        assert torch.all(result[mask] == -1)
        result[mask] = curr_idx

        curr_idx += 1

    assert torch.all(result > -1)
    return curr_idx, result


def scatter_mean0(src, index, axis_size=None, return_counts=False):
    """
    Scatter mean on 0-th axis
    """

    if axis_size is None:
        axis_size = torch.max(index) + 1

    # Target shape is target size and remaining value shape without indexed dimension
    accumulator = torch.zeros((axis_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    numerator = torch.zeros(axis_size, dtype=torch.int64, device=src.device)

    torch.index_put_(accumulator, (index,), src, accumulate=True)
    ones = torch.ones(len(index), dtype=torch.int64, device=numerator.device)
    torch.index_put_(numerator, (index,), ones, accumulate=True)

    result = accumulator / numerator.reshape(axis_size, *((1,) * len(src.shape[1:])))

    if return_counts:
        return result, numerator
    else:
        return result
