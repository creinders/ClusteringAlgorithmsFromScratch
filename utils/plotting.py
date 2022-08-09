
def plot_clustering(points, assignments=None, centers=None, fig=None, ax=None, labels=None,
                    alpha=1., center_size=1., center_marker='o', point_size=1., point_marker='x',
                    palette=None, pfx=True, equal_axis_scale=True, center_per_point=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns

    sns.set_style("white")

    if pfx:
        import matplotlib.patheffects as path_effects
        path_effects = [path_effects.withStroke(linewidth=2, foreground='black')]
    else:
        path_effects = None

    if assignments is None:
        assignments = np.zeros(len(points), dtype=int)

    assert len(points) == len(assignments)

    if len(points.shape) != 2 or points.shape[-1] != 2:
        raise ValueError("Invalid points shape: {}".format(points.shape))

    def plot_wrap(fn, pts, *args, **kwargs):
        # Ensure always a 2D array even if just one point
        if len(pts.shape) == 1:
            pts = np.array([pts])

        return fn(pts[:, 0], pts[:, 1], *args, **kwargs)

    if centers is not None:
        n_centers = len(centers)
        assert np.max(assignments) < n_centers
    else:
        n_centers = np.max(assignments) + 1

    if palette is None:
        colors = plt.get_cmap('tab10').colors
        palette = [colors[i % 10] for i in range(n_centers)]

    unique = np.unique(assignments)

    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    
    
    
    if ax is None:
        ax = fig.add_subplot(111)

    ax.cla()

    for u in unique:
        label = labels[u] if labels is not None else None

        p_color = palette[u]
        p_points = points[assignments == u]
        plot_wrap(ax.scatter, p_points, marker=point_marker, s=point_size * rcParams['lines.markersize'] ** 2,
                  color=p_color, label=label, alpha=alpha, path_effects=path_effects)

        if centers is not None:
            if center_per_point:
                p_cluster = centers[assignments == u, :]
            else:
                p_cluster = centers[u]
                
            plot_wrap(ax.scatter, p_cluster, marker=center_marker, color=palette[u],
                    s=center_size * rcParams['lines.markersize'] ** 2, alpha=alpha, path_effects=path_effects)

    if equal_axis_scale:
        # Ensure X and Y axis have equal scale in visualization
        plt.gca().set_aspect('equal', adjustable='box')

    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    

    plt.tight_layout()
    return ax