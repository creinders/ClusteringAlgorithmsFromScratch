import torch

from algorithms.meanshift_base import MeanShiftBase


class MeanShiftPytorch(MeanShiftBase):

    def __init__(self, cuda=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cuda = cuda

        self.pi = torch.asin(torch.tensor(1.))
        if cuda:
            self.pi = self.pi.cuda()

    def prepare(self, X):
        X = torch.from_numpy(X)

        if self.cuda:
            X = X.cuda()

        return X, X.clone()

    def distance(self, a, b):
        # shape a: N x C
        # shape b: M x C

        d = torch.sum(torch.square(a[:,  None, :] - b[None, :, :]), dim=-1)
        return d

    def kernel(self, distances):
        return torch.exp(-0.5 * ((distances / self.bandwidth ** 2)))

    def _main_loop(self, X, clusters):        

        iteration = 0
        while True:
            iteration += 1
            d = self.distance(clusters, X)
            w = self.kernel(d)

            new_centers = w[:, :, None] * X[None, :, :]
            w_sum = w.sum(1)
            new_centers = torch.sum(new_centers, dim=1) / w_sum[:, None]

            diff = torch.sum(torch.square(new_centers - clusters))

            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff.item()))

            if diff < self.early_stop_threshold:
                break
            clusters = new_centers

        clusters, assignments = self._group_clusters(clusters)
        return clusters, assignments

    def _group_clusters(self, points):
        from algorithms.util_pt import connected_components_undirected, scatter_mean0

        _, cluster_ids = connected_components_undirected(self.distance(points, points) < self.cluster_threshold)
        cluster_centers = scatter_mean0(points, cluster_ids)
        return cluster_centers, cluster_ids
    
    def tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()
