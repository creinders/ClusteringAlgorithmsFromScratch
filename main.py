import seaborn as sns
import numpy as np
from callbacks.timing_callback import TimingCallback

import hydra
from omegaconf import  DictConfig
from datasets.builder import build_dataset
from algorithms.builder import build_algorithm
from callbacks.visualization_callback import VisualizationCallback
from callbacks.visualization_gif_callback import VisualizationGifCallback


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    
    X = build_dataset(cfg)
    X = X.astype(np.float32)

    verbose = cfg.verbose
    n = 1

    if cfg.time:
        if cfg.plot_gif or cfg.plot:
            print('WARNING: cannot plot and time at the same time')
        callback = TimingCallback()
        verbose = False
        n = cfg.time_repeats

    elif cfg.plot_gif:
        if cfg.framework != 'numpy':
            print('WARNING: set framework to numpy for generating gifs')

        callback = VisualizationGifCallback(cfg.algorithm_name, plot_png=cfg.plot)

    elif cfg.plot:
        
        callback = VisualizationCallback()

    else:
        callback = None

    clustering = build_algorithm(cfg, callback=callback, verbose=verbose)
    for _ in range(n):
        clustering.fit(X.copy())

    if cfg.time:
        print('{}s per iteration (n={})'.format(callback.total_duration / callback.n, callback.n))


if __name__ == "__main__":
    main()
