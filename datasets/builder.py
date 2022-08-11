from datasets.dataset import load_moons, normalize, load_text_data, load_s4

def build_dataset(cfg):
    import os

    res_path = os.path.join(os.path.dirname(__file__), '../data')

    dataset_name = cfg.dataset

    if dataset_name == 'moons':
        X, _ = load_moons(500, random_state=cfg.seed)
    elif dataset_name == 'aggregation':
        X, _ = load_text_data(os.path.join(res_path, 'Aggregation.txt'), random_state=cfg.seed)
    elif dataset_name == 'jain':
        X, _ = load_text_data(os.path.join(res_path, 'jain.txt'), random_state=cfg.seed)
    elif dataset_name == 's4':
        X, _ = load_s4(random_state=cfg.seed)
    else:
        raise ValueError('unknown dataset: {}'.format(dataset_name))
    # make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=cfg.seed)
    X = normalize(X)

    return X
