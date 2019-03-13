import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA


output_dir = Path("/root/output")

# Load pre-processed lesion cube.
npy_file = output_dir / "processed.npy"
lesion_cube = np.load(str(npy_file))
m, n, k = lesion_cube.shape
print(lesion_cube.shape)

# Represent cube as pixels.
# lesion_pixels = lesion_cube.flatten()
lesion_pixels = lesion_cube.reshape((-1, k))
print("lesion pixels: ", lesion_pixels.shape)

# PCA dimensionality reduction...
pca = PCA(n_components=10)
lesion_pixels = pca.fit_transform(lesion_pixels)
print(pca.explained_variance_ratio_)

# Simple k-means clustering stuff...
kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)
labels = kmeans.fit_predict(lesion_pixels)

# Plot RGB image with cluster labels.
rgb_image = plt.imread(str(output_dir / "rgb.png"))
plt.figure()
# plt.imshow(rgb_image)
plt.imshow(labels.reshape((m, n)))
plt.savefig(str(output_dir / "kmeans-pca.png"))
