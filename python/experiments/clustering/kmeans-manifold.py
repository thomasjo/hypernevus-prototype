import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE


output_dir = Path("/root/output")

# Load pre-processed lesion cube.
npy_file = output_dir / "processed.npy"
lesion_cube = np.load(str(npy_file))
m, n, k = lesion_cube.shape
print(lesion_cube.shape)

crop_x, crop_y = 320, 300
crop_size = 150
crop_slice = (
    slice(crop_y, crop_y + crop_size),
    slice(crop_x, crop_x + crop_size),
)

lesion_cube = lesion_cube[crop_slice]
m, n, k = lesion_cube.shape

# Represent cube as pixels.
# lesion_pixels = lesion_cube.flatten()
lesion_pixels = lesion_cube.reshape((-1, k))
print("lesion pixels: ", lesion_pixels.shape)

# PCA dimensionality reduction...
pca = PCA(n_components=10)
lesion_pixels = pca.fit_transform(lesion_pixels)

# Simple manifold learning...
embedding = SpectralEmbedding(n_components=8, random_state=0, eigen_solver="amg")
lesion_pixels = embedding.fit_transform(lesion_pixels)

# Simple k-means clustering stuff...
kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)
labels = kmeans.fit_predict(lesion_pixels)

# Plot embedding...
plt.figure()
# plt.imshow(rgb_image)
plt.imshow(embedding.reshape((m, n)))
plt.savefig(str(output_dir / "kmeans-tsne.png"))
