from functools import partial

def build_algorithm(cfg, callback=None, verbose=False):
    algorithm_name = cfg.algorithm_name
    framework_name = cfg.framework

    if algorithm_name == 'kmeans':

        if framework_name == 'numpy':
            from algorithms.kmeans_np import KMeansNumpy
            clustering_class = KMeansNumpy
        elif framework_name == 'pytorch':
            from algorithms.kmeans_pt import KMeansPytorch
            clustering_class = partial(KMeansPytorch, cuda=cfg.cuda)
        elif framework_name == 'jax':
            from algorithms.kmeans_jax import KMeansJax
            clustering_class = KMeansJax
        elif framework_name == 'tensorflow':
            from algorithms.kmeans_tf import KMeansTensorflow
            clustering_class = KMeansTensorflow
        elif framework_name == 'tensorflow_eager':
            from algorithms.kmeans_tf_eager import KMeansTensorflowEager
            clustering_class = KMeansTensorflowEager    
        else:
            raise ValueError('Unknown framework or not implemented: {}'.format(framework_name))

        n_clusters = cfg.n_clusters if cfg.n_clusters is not None else cfg.default_n_clusters
        clustering = clustering_class(n_clusters=n_clusters, max_iter=cfg.max_iter, early_stop_threshold=cfg.early_stop_threshold, verbose=verbose, callback=callback, seed=cfg.seed)

    elif algorithm_name == 'meanshift':

        if framework_name == 'numpy':
            from algorithms.meanshift_np import MeanShiftNumpy
            clustering_class = MeanShiftNumpy
        elif framework_name == 'pytorch':
            from algorithms.meanshift_pt import MeanShiftPytorch
            clustering_class = partial(MeanShiftPytorch, cuda=cfg.cuda)
        elif framework_name == 'jax':
            from algorithms.meanshift_jax import MeanShiftJax
            clustering_class = MeanShiftJax
        elif framework_name == 'tensorflow':
            from algorithms.meanshift_tf import MeanShiftTensorflow
            clustering_class = MeanShiftTensorflow
        elif framework_name == 'tensorflow_eager':
            from algorithms.meanshift_tf_eager import MeanShiftTensorflowEager
            clustering_class = MeanShiftTensorflowEager          
        else:
            raise ValueError('Unknown framework or not implemented: {}'.format(framework_name))
        
        bandwidth = cfg.bandwidth if cfg.bandwidth is not None else cfg.default_bandwidth
        clustering = clustering_class(bandwidth=bandwidth, early_stop_threshold=cfg.early_stop_threshold, cluster_threshold=cfg.cluster_threshold, verbose=verbose, callback=callback)

    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm_name))


    return clustering