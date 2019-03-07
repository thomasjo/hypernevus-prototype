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

# Simple k-means clustering stuff...
# kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)
kmeans = KMeans(n_clusters=8, random_state=0)
labels = kmeans.fit_predict(lesion_pixels)

rgb_image = plt.imread(str(output_dir / "rgb.png"))
rgb_image = rgb_image[crop_slice]

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)
ax1.imshow(rgb_image)
ax2.imshow(labels.reshape((m, n)))
fig.savefig(str(output_dir / "kmeans.png"))
