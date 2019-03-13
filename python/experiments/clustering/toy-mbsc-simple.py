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


def make_blob(center, radius):
    x, y = np.indices((100, 100))
    return x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2

center1, radius1 = (28, 24), 16
center2, radius2 = (40, 50), 14
center3, radius3 = (67, 58), 15
center4, radius4 = (24, 70), 14

circle1 = make_blob(center1, radius1)
circle2 = make_blob(center2, radius2)
circle3 = make_blob(center3, radius3)
circle4 = make_blob(center4, radius4)

img = circle1 + circle2 + circle3 + circle4
img_mask = img.astype(bool)
img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
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
        G += (1.0 / n_proj) * np.matmul(np.matmul(L, np.outer(r_i, r_i)), W)

    return np.matmul(np.eye(n) - np.matmul(W, W.T), G)


affinity_matrix = graph.toarray()
laplacian = csgraph.laplacian(affinity_matrix, normed=True)

print(affinity_matrix.shape)
print(laplacian.shape)

n_samples = affinity_matrix.shape[0]
n_clusters = 4

n_iter = 100
p = 10 / n_samples
n_proj = 5
step_size = 0.025
epsilon = 1e-6

W = sp.linalg.orth(np.random.randn(n_samples, n_clusters))
M = np.zeros((n_samples, n_clusters))

for t in range(1, n_iter):
    print('iteration:', t)
    H_tilde = stochastic_gradient_manifold(laplacian, p, n_proj, W)
    M += np.power(H_tilde, 2)
    H_hat = H_tilde / (epsilon + np.sqrt(M))
    Q, _ = np.linalg.qr(W - step_size * H_hat)
    W = Q

labels = MiniBatchKMeans(n_clusters=n_clusters, n_init=50).fit_predict(W)

# ---------


# Force the solver to be arpack, since amg is numerically
# unstable on this example
# labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')


img_labeled = np.full_like(img_mask, -1.0)
img_labeled[img_mask] = labels

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
