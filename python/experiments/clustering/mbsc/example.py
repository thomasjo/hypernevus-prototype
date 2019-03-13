import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.sparse.linalg import eigs
from scipy.spatial.distance import pdist, squareform

import mbsc

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

    eigensolver = mbsc.ManifoldEigensolver()

    t = time.time()
    W, V = eigensolver.fit(K, k=2, lr=0.005, m=5, Nr=40, maxiter=10000, epsilon=1E-9, tol=1E-4)

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
    colors = cm(np.linspace(0,1,2*N))

    plt.figure()
    plt.scatter(W[:,0], W[:,1], c=colors)
    plt.title('Eigenvectors MBSC')

    plt.figure()
    plt.scatter(Wtrue[:,0], Wtrue[:,1], c=colors)
    plt.title('True eigenvectors')
    plt.show()
