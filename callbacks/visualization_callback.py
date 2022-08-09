import matplotlib.pyplot as plt
from callbacks.callback import DefaultCallback
from utils.plotting import plot_clustering

class VisualizationCallback(DefaultCallback):

    def __init__(self, filename='visualization.png') -> None:
        self.filename = filename


    def on_epoch_end(self, method, X, clusters, assignments):
        X = method.tensor_to_numpy(X)
        
        plot_clustering(X, assignments=assignments, centers=clusters)
        plt.savefig(self.filename)

