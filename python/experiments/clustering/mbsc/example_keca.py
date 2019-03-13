import logging
import numpy as np
import matplotlib.pyplot as plt
import time

from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel

import mbsc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def keca_frey():
    data = loadmat('/home/sigurd/phd/data/frey_rawface.mat')
    X = data['ff'].T/255.0
    X = normalize(X)

    # MBSC
    dist = pdist(X)
    sigma = 0.15*np.median(dist)

    manifold_keca = mbsc.ManifoldKECA()

    t = time.time()
    Z = manifold_keca.fit(X, k=3, sigma=sigma, kernel='rbf', lr=0.005, m=5,
                          Nr=40, maxiter=1000, epsilon=1E-9, tol=1E-4)
    logger.info('Time, MBSC: %.4f seconds' % (time.time() - t))

    t = time.time()
    K = rbf_kernel(X, gamma=0.5/sigma**2)
    _, _ = np.linalg.eig(K)
    logger.info('Time, EIG: %.4f seconds' % (time.time() - t))

    # Plot
    plot_result(Z)


def plot_result(Z):
    cm = plt.get_cmap('RdYlBu')
    colors = cm(np.linspace(0, 1, 2*Z.shape[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors)
    plt.title('KECA MBSC')
    plt.show()


def eig_comparison_frey():
    data = loadmat('/home/sigurd/phd/data/frey_rawface.mat')
    X = data['ff'].T/255.0
    X = normalize(X)

    # MBSC
    k = 3
    dist = pdist(X)
    sigma = 0.15*np.median(dist)
    eigensolver = mbsc.ManifoldKECA()

    t = time.time()
    W = eigensolver.fit(X, sigma=sigma, kernel='rbf', k=k, lr=0.005, m=5,
                        Nr=40, maxiter=1000, epsilon=1E-9, tol=1E-6)
    logger.info('Time, MBSC: %.4f seconds' % (time.time() - t))

    plot_result(W)


if __name__ == '__main__':
    keca_frey()
    eig_comparison_frey()
