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



#### stochastic approximation to matrix product
def StochasticAverageMPFast(A,B,nelem,sampleIter,sample_mat):
    rowA = A.shape[0]
    colA = A.shape[1]
    rowB = B.shape[0]
    colB = B.shape[1]
    AB_prod = np.zeros((rowA,colB),dtype=float)
    ### generate sampling matrix
    #rvec = np.random.random_integers(0,high=(colA-1),size=(nelem,sampleIter))
    ### count unique elements
    #unique_rvec = np.unique(rvec)
    ### avoid repeated calculation
    rvec_list =[]
    #rvec_ind = list(range(colA))
    p = float(nelem) / float(colA)
    for iter in range(0,sampleIter):
        #np.random.shuffle(rvec_ind)
        rvec = sample_mat[:,iter].ravel().astype(int)
        #print rvec
        #rvec = rvec_ind[:nelem]
        rvec_list.extend(list(rvec))
        print(rvec)
        AB_prod += np.dot(A[:,list(rvec)],B[list(rvec),:]) * (1./p) * (1./float(sampleIter))
        #AB_prod += np.dot(A[:,rvec[:,iter]],B[rvec[:,iter],:]) * (1./p) * (1./float(sampleIter))

    return AB_prod,len(np.unique(rvec_list))
    
#### Riemannian Gradient Calculation using stochastic approximation 
def RiemannianGrad(C_off_diag,X,nsamples,nrounds,sample_mat):
    #### -(I-XX^T)CX
    #### Step.1 stochastic approximation to CX
    CX, nnz_len = StochasticAverageMPFast(C_off_diag,X,nsamples,nrounds,sample_mat)
    CX = CX - X
    #### Step.2 XCX
    XCX = np.dot(X.T,CX) ## do we need to use stochastic approximation to XCX
    #### Step.3 XXCX'
    XXCX = np.dot(X,XCX)
    G = CX - XXCX
    G = -1. * G
    return G, nnz_len

#### Spectral decomposition using Nystrom approximation
def nystromSP(train_data, nsample, sigma, num_clusters):
    # num_clusters = np.unique(train_label).shape[0]
    data_ind = np.array(range(train_data.shape[0]))
    np.random.shuffle(data_ind)
    sampled_ind = data_ind[:nsample]
    other_ind = data_ind[nsample:]

    A = rbf_kernel(train_data[sampled_ind,:], train_data[sampled_ind,:], gamma=sigma)
    B = rbf_kernel(train_data[sampled_ind,:], train_data[other_ind,:], gamma=sigma)
    d1 = np.sum(A, axis=1) + np.sum(B, axis=1)
    d2 = np.sum(B.T, axis=1) + np.dot(B.T, np.dot(np.linalg.inv(A), np.sum(B, axis=1)))
    dhat = np.reshape(np.sqrt(1.0 / np.concatenate([d1, d2])), [train_data.shape[0], 1])
    A = np.multiply(A, np.dot(dhat[0:nsample], dhat[0:nsample].T))
    m = train_data.shape[0] - nsample
    B1 = np.dot(dhat[0:nsample, :], dhat[nsample:(nsample+m), :].T)
    B = np.multiply(B, B1)
    Asi = sp.linalg.sqrtm(np.linalg.inv(A))
    BBT  = np.dot(B, B.T)
    W = np.zeros((A.shape[0] + B.shape[1], A.shape[1]))
    W[0:A.shape[0], :] = A
    W[A.shape[0]:, :] = B.T
    R = A + np.dot(Asi,np.dot(BBT, Asi))
    R = (R + R.T) / 2
    S,U = np.linalg.eigh(R) ### ascending order of eigenvalues
    S = np.diag(S[::-1])
    U = U[:, ::-1]

    W = np.dot(W,Asi)
    V = np.dot(np.dot(W, U[:, :num_clusters]), np.linalg.inv(np.sqrt(S[:num_clusters, :][:, :num_clusters])))

    sq_sum = np.sqrt(np.sum(np.multiply(V, V), axis=1)) + 1e-20
    sq_sum_mask = np.zeros((len(sq_sum), num_clusters), dtype=float)
    for k in range(num_clusters):
        sq_sum_mask[:, k] = sq_sum

    Umat = np.divide(V, sq_sum_mask)
    X = np.zeros((Umat.shape[0], Umat.shape[1]))
    X[data_ind, :] = Umat
    return X
    
#### Mini-batch based spectral decomposition using Stochastic Gradient on Manifold    
def StochasticRiemannianOpt(C,X,ndim,master_stepsize,auto_corr,outer_max_iter,nsamples,nrounds):
    ### AdaGrad
    ### obj func X = argmin_{X} (0.5) * X.T L X, s.t. X.TX = I
    ### Semi-sotchastic gradient descent to control variance reduction
    fudge_factor = 1e-6
    historical_grad = 0.
    k = ndim
    n = C.shape[0]
    ### orthonormalisation
    X = Orthonormalisation(X)
    ### extract diagonal part
    for k in range(C.shape[0]):
        C[k, k] = 0.

    nnz_list = [] ## recording how many no-zero entries sampled each iteration
    X_sto_list = []

    print('generate sampling templates')
    sample_mat_list = []
    rvec_idx = list(range(C.shape[0]))
    for itr in range(0, outer_max_iter):
        rvec_mat = np.zeros((nsamples,nrounds), dtype=float)
        for k in range(nrounds):
            np.random.shuffle(rvec_idx)
            rvec = rvec_idx[:nsamples]
            rvec_mat[:, k] = rvec

        sample_mat_list.append(rvec_mat)

    print(sample_mat_list[0])

    print('the iteration begins ')
    for itr in range(0, outer_max_iter):
        ### stochastic gradient
        stoG,nnz = RiemannianGrad(C, X, nsamples, nrounds, sample_mat_list[itr])
        nnz_list.append(nnz)
        historical_grad += auto_corr * historical_grad + (1.0 - auto_corr) * np.power(stoG, 2)
        adjusted_grd = stoG / (fudge_factor + np.sqrt(historical_grad))
        X = X - master_stepsize * adjusted_grd
        X = Orthonormalisation(X)
        X_sto_list.append(X)
        print('iteration: ' + str(itr))

    return X, nnz_list, X_sto_list

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X, mode='reduced')
    return Q

def Orthonormalisation(X):
    #### orthonormalisation
    if X.shape[1] == 1:
       target_norm = np.linalg.norm(X)
       X_hat = X / (target_norm + 1e-6)
    else:
        X_hat = gram_schmidt_columns(X)

    return X_hat



train_data = img[mask].reshape((-1, 1))
print(train_data.shape)

gamma_value = 5.0
# affinity_matrix = rbf_kernel(train_data, gamma=gamma_value)
affinity_matrix = graph.toarray()
laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)

print(affinity_matrix.shape)

nclass = 4
nsample = train_data.shape[0]

#### Configuring AdaGrad
print('mini batch size = 100')
# master_stepsize = 0.0025
master_stepsize = 0.025
outer_iter = 6000
nsampleround = 50
ncols = 2
auto_corr = 0.0
ndim = nclass
print('nsampleround: ' + str(nsampleround))
print('ncols: ' + str(ncols))
print('master_stepsize: ' + str(master_stepsize))

# nmi_sgd_set = []
num_repeat_exp = 1
for repeatExp in range(num_repeat_exp):
    print('iteration id: ' + str(repeatExp))
    X = nystromSP(train_data, 10, gamma_value, nclass)
    X_sto1, nnz_list, X_sto_list = StochasticRiemannianOpt(laplacian, X, ndim,master_stepsize, auto_corr, outer_iter, ncols, nsampleround)
    # nmi_sgd = []
    for i in range(len(X_sto_list)):
        if i % 5 ==0:
            X_sto_tmp = X_sto_list[i].T * dd
            X_sto_tmp = _deterministic_vector_sign_flip(X_sto_tmp)
            cluster_id = MiniBatchKMeans(n_clusters=nclass, n_init=50).fit(X_sto_tmp.T).labels_
            # nmi = normalized_mutual_info_score(train_label,cluster_id, average_method="geometric") ### measuring NMI score per iteration
            # nmi_sgd.append(nmi)

    # nmi_sgd_set.append(nmi_sgd)

# ---------


# Force the solver to be arpack, since amg is numerically
# unstable on this example
# labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
labels = cluster_id
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

plt.savefig('output/toy-mbsc.png')

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
