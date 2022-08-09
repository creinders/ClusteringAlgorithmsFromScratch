import matplotlib.pyplot as plt
from callbacks.callback import DefaultCallback
from utils.plotting import plot_clustering
import os
import imageio

class VisualizationGifCallback(DefaultCallback):

    def __init__(self, algorithm_name, filename='visualization.png', filename_gif='visualization.gif', plot_png=True, tmp_folder='output', fps=2, repeat_last_result=5) -> None:
        self.tmp_folder = tmp_folder
        self.filename_gif = filename_gif
        self.fps = fps
        self.repeat_last_result = repeat_last_result
        self.center_per_cluster = True if algorithm_name == 'meanshift' else False
        self.plot_png = plot_png
        self.filename = filename

        os.makedirs(self.tmp_folder, exist_ok=True)

        self.files = []
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
    
    
    def on_iteration_end(self, method, iteration, X, clusters, assignments):

        plot_clustering(X, assignments=assignments, centers=clusters, center_per_point=self.center_per_cluster, fig=self.fig, ax=self.ax)
        plot_path = os.path.join(self.tmp_folder, 'iteration-{}.png'.format(iteration))
        plt.savefig(plot_path)
        self.files.append(plot_path)

    def on_epoch_end(self, method, X, clusters, assignments):
        if len(self.files) == 0:
            print('no images for gif')
            return 

        X = method.tensor_to_numpy(X)
        
        plot_clustering(X, assignments=assignments, centers=clusters, fig=self.fig, ax=self.ax)
        plot_path = os.path.join(self.tmp_folder, 'iteration-final.png')
        plt.savefig(plot_path)

        for _ in range(self.repeat_last_result):
            self.files.append(plot_path)

        if self.plot_png:
            plt.savefig(self.filename)

        print('creating gif')
        with imageio.get_writer(self.filename_gif, mode='I', fps=self.fps) as writer:
            for filename in self.files:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        print('cleaning up temporary images')
        for filename in set(self.files):
            os.remove(filename)
