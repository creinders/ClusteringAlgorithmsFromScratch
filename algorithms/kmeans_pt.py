from algorithms.kmeans_base import KMeansBase
from algorithms.util_pt import scatter_mean0
import torch


class KMeansPytorch(KMeansBase):

    def __init__(self, *args, cuda=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cuda = cuda

    def prepare(self, X):
        centers = self.init_clusters(X, self.n_clusters)

        X = torch.from_numpy(X)
        centers = torch.from_numpy(centers)

        if self.cuda:
            X = X.cuda()
            centers = centers.cuda()

        return X, centers

    def _main_loop(self, X, centers):
        for iteration in range(self.max_iter):

            distance = (X[:, :, None] - centers.permute((1, 0))[None, ...]).square().sum(1)
            assignments = torch.argmin(distance, dim=1)

            # k-Means can assign no points to a cluster center, in that case keep old value
            center_means, assigned_counts = scatter_mean0(X, assignments, axis_size=self.n_clusters, return_counts=True)
            new_centers = torch.clone(centers)
            new_centers[assigned_counts > 0] = center_means[assigned_counts > 0]

            diff = (new_centers - centers).square().sum()
            if self.verbose:
                print('Iteration {}: {} difference'.format(iteration, diff))

            if diff < self.early_stop_threshold:
                break

            centers = new_centers

        return centers, assignments

    def tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()