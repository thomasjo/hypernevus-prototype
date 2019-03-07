import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import image
from sklearn.utils.extmath import _deterministic_vector_sign_flip

import mbsc


def generate_graph():
    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14
    # radius1, radius2, radius3, radius4 = 8, 7, 8, 7

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

    return graph, img, mask


if __name__ == '__main__':
    eigensolver = mbsc.ManifoldEigensolver()

    graph, img, mask = generate_graph()
    affinity_matrix = graph.toarray()
    laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)

    np.fill_diagonal(laplacian, 1)

    # W, V = eigensolver.fit(laplacian, k=4, lr=0.05, m=100, Nr=100, maxiter=20000, epsilon=1E-9, tol=1E-4)

    W, V = eigensolver.fit(laplacian, k=5, lr=0.05, m=100, Nr=100, maxiter=20000, epsilon=1E-9, tol=1E-5)
    sort_idx = np.argsort(V)[::-1]
    W = W[:, sort_idx[1:]]

    W = (W.T * dd).T
    W = _deterministic_vector_sign_flip(W)

    # labels = MiniBatchKMeans(n_clusters=4, n_init=50).fit_predict(W)
    labels = KMeans(n_clusters=4, n_init=50).fit_predict(W)

    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    fig = plt.figure()
    # ax1, ax2 = fig.subplots(1, 2)
    ax2 = fig.subplots(1, 1)
    # ax1.matshow(img)
    ax2.matshow(label_im)

    plt.savefig('output/example-toy.png')
