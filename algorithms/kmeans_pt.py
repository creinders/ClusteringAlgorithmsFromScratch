from math import dist
import numpy as np
from .kmeans_base import KMeansBase
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

        for _ in range(self.max_iter):

            distance = (X[:, :, None] - centers.permute((1, 0))[None, ...]).square().sum(1)
            assignments = torch.argmin(distance, dim=1)

            new_centers = centers
            for i in range(self.n_clusters):
                new_centers[i] = X[assignments == i].mean(0)

            diff = (new_centers - centers).square().sum()
            if diff < self.early_stop_threshold:
                break

            centers = new_centers

        return centers, assignments

    def tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()