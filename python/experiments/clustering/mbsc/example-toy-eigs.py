import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import pdist, squareform

from sklearn.feature_extraction import image

import mbsc


def generate_graph():
    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    # radius1, radius2, radius3, radius4 = 16, 14, 15, 14
    radius1, radius2, radius3, radius4 = 8, 6, 7, 6

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

    # #############################################################################
    # 4 circles
    # img = circle1 + circle2 + circle3 + circle4
    # img = circle1 + circle2
    img = circle1

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
    mu1 = np.array([0, 0], dtype=float)
    mu2 = np.array([5, 5], dtype=float)
    N = 2000

    S = np.array([[1, 0], [0, 1]], dtype=float)
    X = np.concatenate(
        (np.random.multivariate_normal(mu1, S, N), np.random.multivariate_normal(mu2, S, N)),
        axis=0)

    y = np.concatenate(
        (np.zeros((N,1)), np.ones((N,1)))
    )

    # MBSC
    dist = pdist(X)
    sigma = 0.15*np.median(dist)
    K = squareform(np.exp(-1/(2*sigma**2)*dist**2)) + np.eye(2*N)

    # graph, img, mask = generate_graph()
    # affinity_matrix = graph.toarray()
    # laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)
    K = csgraph.laplacian(K, normed=True, return_diag=False)

    eigensolver = mbsc.ManifoldEigensolver()

    t = time.time()
    W, V = eigensolver.fit(K, k=2, lr=0.025, m=5, Nr=40, maxiter=10000, epsilon=1E-6, tol=1E-6)

    idx = np.argsort(V)[::-1]
    W = W[:, idx]
    V = V[idx]
    print('Time, MBSC: %.4f seconds' % (time.time() - t))

    t = time.time()
    Vtrue, Wtrue = eigs(K, k=2, which='LM')
    print('Time, eigendecomposition: %.4f seconds' % (time.time() - t))

    print('Eigenvalues MBSC:')
    print(V)
    print('Eigenvalues eigs:')
    print(np.real(Vtrue))

    cm = plt.get_cmap('RdYlBu')
    colors = cm(np.linspace(0,4,4*laplacian.shape[0]))

    plt.figure()
    plt.scatter(W[:,0], W[:,1])
    plt.title('Eigenvectors MBSC')

    plt.figure()
    plt.scatter(Wtrue[:,0], Wtrue[:,1])
    plt.title('True eigenvectors')
    plt.show()
