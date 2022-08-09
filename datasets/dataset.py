import numpy as np


def normalize(d, epsilon=1e-15):
    assert len(d.shape) == 2
    if not np.issubdtype(d.dtype, np.floating):
        raise ValueError("Invalid dtype: {}".format(d.dtype))
    min = np.min(d, axis=0)
    max = np.max(d, axis=0)
    return (d - min) / np.maximum(max - min, epsilon)

def load_text_data(path, shuffle=True):
    total = np.loadtxt(path)

    if shuffle:
        np.random.shuffle(total)

    # Last column should be label, subtract minimum to always have 0 indexing
    total[:, -1] = total[:, -1] - np.min(total[:, -1])
    x, y = total[..., :-1], total[..., -1].astype(int)

    return x, y


def load_moons(n_total=2000):
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n_total, noise=.05)


def load_pa(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    result = []
    is_start = False

    for l in lines:
        if is_start:
            result.append(int(l))
        elif l.startswith('---'):
            is_start = True

    return np.array(result) - np.min(result)


def load_s4(n_total=1000):
    import os

    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    x = np.loadtxt(os.path.join(data_folder, 's4.txt'))
    y = load_pa(os.path.join(data_folder, 's4-label.pa'))
    assert len(x) == len(y)

    if n_total is not None:
        idxs = np.random.choice(np.arange(len(x)), n_total)
        x, y = x[idxs], y[idxs]

    return x, y