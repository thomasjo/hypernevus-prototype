## Copyright 2017 Yufei HAN, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


import numpy as np
import scipy as sp
from numpy.linalg import *
from scipy.linalg import *
from numpy.random import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from scipy.sparse import csgraph


# ---------


import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

np.random.seed(42)

l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

# #############################################################################
# 4 circles
img = circle1 + circle2 + circle3 + circle4

# We use a mask that limits to the foreground: the problem that we are
# interested in here is not separating the objects from the background,
# but separating them one from the other.
mask = img.astype(bool)

img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())


# ---------


def sampling_vector(p, n):
    values = np.array([-1.0 / np.sqrt(p), 0, 1.0 / np.sqrt(p)])
    value_probs = np.array([p / 2, 1 - p, p / 2])

    return np.random.choice(values, n, p=value_probs)

def stochastic_gradient_manifold(L, p, n_proj, W):
    n, k = W.shape
    G = np.zeros((n, k))

    for i in range(n_proj):
        r_i = sampling_vector(p, n)
        print(r_i)
        G += (1.0 / n_proj) * np.matmul(np.matmul(L, np.outer(r_i, r_i)), W)

    return np.matmul(np.eye(n) - np.matmul(W, W.T), G)


affinity_matrix = graph.toarray()
laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)

print(affinity_matrix.shape)
print(laplacian.shape)

n_samples = affinity_matrix.shape[0]
n_clusters = 4

n_iter = 50
p = 2 / n_samples
n_proj = 5
step_size = 0.025
epsilon = 10e-6

W = [None] * n_iter
W[0] = sp.linalg.orth(np.random.randn(n_samples, n_clusters))

M = [None] * n_iter
M[0] = np.zeros((n_samples, n_clusters))

for t in range(1, n_iter):
    print('iteration:', t)
    H_tilde = stochastic_gradient_manifold(laplacian, p, n_proj, W[t - 1])
    M[t] = M[t - 1] + H_tilde * H_tilde
    H_hat = H_tilde / (epsilon + np.sqrt(M[t]))
    Q, _ = np.linalg.qr(W[t - 1] - step_size * H_hat)
    W[t] = Q

labels = MiniBatchKMeans(n_clusters=n_clusters, n_init=50).fit_predict(W[-1])

# ---------


# Force the solver to be arpack, since amg is numerically
# unstable on this example
# labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')


label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

plt.savefig('output/toy-mbsc-simple.png')

# #############################################################################
# 2 circles
# img = circle1 + circle2
# mask = img.astype(bool)
# img = img.astype(float)

# img += 1 + 0.2 * np.random.randn(*img.shape)

# graph = image.img_to_graph(img, mask=mask)
# graph.data = np.exp(-graph.data / graph.data.std())

# labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
# label_im = np.full(mask.shape, -1.)
# label_im[mask] = labels

# plt.matshow(img)
# plt.matshow(label_im)

# plt.show()
