# Clustering


Implementation of unsupervised clustering algorithms from scratch in different machine learning frameworks. The goal is to demonstrate the similarities and differences of the frameworks. 
If you have an idea to improve an implementation (e.g., a more elegant or faster solution) or would like to implement a different algorithm/framework, please feel free to contribute.

Clustering algorithms
- K-Means
- Mean shift

Machine learning frameworks
- [NumPy](https://numpy.org)
- [PyTorch](https://pytorch.org)
- [TensorFlow (Eager and Graph Mode)](https://www.tensorflow.org)
- [JAX](https://jax.readthedocs.io/)


## Algorithms

| Algorithm    |  Framework  |   |
|  :---------  | :------  | :------ |
| K-Means       | NumPy         |  [kmeans_np.py](algorithms/kmeans_np.py) |
|               | PyTorch       |  [kmeans_pt.py](algorithms/kmeans_pt.py) |
|               | TensorFlow 2 (Eager)   |  [kmeans_tf_eager.py](algorithms/kmeans_tf_eager.py) |
|               | TensorFlow 2 (Graph)   |  [kmeans_tf.py](algorithms/kmeans_tf.py) |
|               | JAX    |  [kmeans_jax.py](algorithms/kmeans_jax.py) |
| Mean shift      | NumPy           |  [meanshift_np.py](algorithms/meanshift_np.py) |
|                 | PyTorch         |  [meanshift_pt.py](algorithms/meanshift_pt.py) |
|                 | JAX         |  [meanshift_jax.py](algorithms/meanshift_jax.py) |
|                 | TensorFlow 2 (Eager)         |  [meanshift_tf_eager.py](algorithms/meanshift_tf_eager.py) |
|                 | TensorFlow 2 (Graph)         |  [meanshift_tf.py](algorithms/meanshift_tf.py) |


![Mean shift on Aggregation](images/meanshift.gif)

## Usage

Please follow the [installation guide](#installation).

You can simply run the following command to execute a `mean shift clustering` on `aggregation` with `JAX`
```
python main.py 
```
To select different algorithms, datasets, or frameworks, r

The algorithm, dataset, and framework can be selected via command like options, set
- `algorithm` to `kmeans` or `meanshift`
- `dataset` to `aggregation`, `jain`, `moons`, `s4`, or `meanshift`
- `framework` to `numpy`, `pytorch`, `jax`, `tensorflow_eager`, or `tensorflow`

For example
```
python main.py algorithm=kmeans dataset=moons framework=pytorch
```
For all options, please see `configs/base.yaml`.

For timing, set `time=true`
```
python main.py time=true
```
Plot result
```
python main.py plot=true
```

Plot gif
```
python main.py plot_gif=true
```

## Installation

Clone repository
```bash
git clone git@github.com:creinders/ClusteringAlgorithmsFromScratch.git
cd ClusteringAlgorithmsFromScratch
```

Install anaconda environment and dependencies
```
conda create -n clustering python=3.9
conda activate clustering

# Install PyTorch (follow https://pytorch.org/get-started)
conda install pytorch -c pytorch

# Install TensorFlow (follow https://www.tensorflow.org/install/pip)
pip install tensorflow

# Install JAX (follow https://github.com/google/jax#installation)
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt
```

If you want to use the datasets `aggregation`, `jain`, or `s4`, please download the data
```
./download_datasets.sh
```
