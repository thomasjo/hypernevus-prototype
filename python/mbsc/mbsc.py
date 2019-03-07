import logging
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

logger = logging.getLogger(__name__)


class ManifoldEigensolver(object):
    def fit(self, L, k, m=5, Nr=40, lr=0.02, maxiter=1000,
            epsilon=1E-3, tol=1E-9):
        W = self._optimize(L, k, m, Nr, lr, maxiter, epsilon, tol)
        W, V = self._eigs(L, W)

        return W, V
        # return W

    def _optimize(self, L, k, m, Nr, lr, maxiter, epsilon, tol):
        N = L.shape[0]
        M = np.zeros((N, k), dtype=float)
        W = self._initialize_W(N, k)

        l = m/float(Nr)
        self._p = l/N

        iter_idx = 0
        max_difference = 2*tol

        logger.info('Start optimization')
        while iter_idx < maxiter and max_difference > tol:
            # print('iteration:', iter_idx + 1)

            # Calculate gradient (in tangent space)
            H = self._htilde(L, m, Nr, W)

            # Adaptive learning rate (ADAM)
            M += np.power(H, 2)
            H /= (np.sqrt(M) + epsilon)

            # Update
            # W += lr*H
            W -= lr*H  # Minimize

            # Project back on manifold (approximate exponential map)
            W_temp = W
            W, _ = np.linalg.qr(W, mode='reduced')

            max_difference = np.max(np.abs(W_temp - W))

            loss = np.trace(0.5 * W.T.dot(L).dot(W))
            print('loss:', loss)

            iter_idx += 1

        logger.info('Optimization done in %d iterations' % iter_idx)
        if max_difference > tol:
            logger.warning('Optimization did not converge. Max difference: %.2E, Tolerance: %.2E' % (max_difference, tol))

        return W

    def _eigs(self, L, W):
        logger.info('Rotate vectors and calculate eigenvalues')

        C = W.T.dot(L).dot(W)

        # Rotate W such that C is diagonal.
        V, U = np.linalg.eig(C)

        return W.dot(U), V

    def _htilde(self, L, m, Nr, W):
        N, _ = L.shape

        # Gradient
        G = np.zeros(W.shape, dtype=float)

        # Stochastic gradient
        for i in range(Nr):
            # Draw components
            r, indeces = self._draw_components(m, N)

            # G += np.dot(L[:, indeces].dot(r),
            #             r.T.dot(W[indeces, :]))*1.0/(self._p*Nr)
            G += np.dot(L[:, indeces],
                        W[indeces, :])*1.0/(self._p*Nr)

        H = G - W.dot(W.T.dot(G))

        return H

    def _draw_components(self, m, N):
        r = np.random.rand(m, 1)
        r[r < 0.5] = -1
        r[r >= 0.5] = 1

        indeces = np.random.choice(N, size=m, replace=False)

        return r, indeces

    def _initialize_W(self, N, k):
        logger.info('Initialize')

        # Random, orthogonal matrix W
        W = np.random.randn(N, k)
        W, _ = np.linalg.qr(W, mode='reduced')

        return W


class ManifoldKECA(object):
    def fit(self, X, k=2, sigma=0.1, kernel='rbf', m=5, Nr=40,
            lr=0.01, maxiter=1000, epsilon=1E-3, tol=1E-9):
        # Kernel setup
        if kernel == 'rbf':
            logger.info('Calculate kernel')

            K = rbf_kernel(X, gamma=0.5/sigma**2)
        elif kernel == 'precomputed':
            logger.info('Kernel precomputed')

            K = X
        else:
            raise ValueError('Unknown kernel type')

        # Calculate 10 times as many eigenvectors as needed (just in case)
        k_eigs = min(10*k, K.shape[0])

        # Find eigenvectors
        eigensolver = ManifoldEigensolver()
        W, V = eigensolver.fit(K, k_eigs, m, Nr, lr, maxiter, epsilon, tol)

        # Sort
        idx = np.argsort(V)[::-1]
        W = W[:, idx]
        V = V[idx]

        return self._entropy_components(W, V, k)

    def _entropy_components(self, W, V, k):
        logger.info('Find max kernel entropy components')

        # Sort entropy components
        entropy_contribution = self._entropy_contribution(W, V)

        # Descending
        component_idx = np.argsort(entropy_contribution)[::-1]

        logger.info('Kernel entropy components:\n %s' % component_idx)

        # Calculate entropy components
        logger.info('Embed data')

        return W[:, component_idx[:k]].dot(np.diag(np.sqrt(V[component_idx[:k]])))
        #return W[:, component_idx[:k]]

    def _entropy_contribution(self, W, V):
        logger.info('Calculate information potential contributions')
        return np.power(np.sum(W, axis=0), 2)*V
