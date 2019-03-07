import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA



u = np.random.randn(10)
v = np.random.randn(10)
print(np.sqrt(((u-v)**2).sum()))

output_dir = Path("/root/output")

# Load pre-processed lesion cube.
npy_file = output_dir / "processed.npy"
lesion_cube = np.load(str(npy_file))
# lesion_cube = lesion_cube[300:400, 300:400, ...]
m, n, k = lesion_cube.shape
print("cube: ", lesion_cube.shape)

crop_x, crop_y = 320, 300
crop_size = 150
crop_slice = (
    slice(crop_y, crop_y + crop_size),
    slice(crop_x, crop_x + crop_size),
)

lesion_cube = lesion_cube[crop_slice]
m, n, k = lesion_cube.shape

# Represent cube as pixels.
lesion_pixels = lesion_cube.reshape((-1, k))
print("lesion pixels: ", lesion_pixels.shape)
print("-" * 72)

# print(pdist(lesion_pixels[0:2]))

# Define function for the Euclidean cumulative spectrum distance.
def ecs_dist(spectra):
    # print("spectra: ", spectra.shape)
    cumulative_spectra = np.cumsum(spectra, axis=1)
    # print("cumulative spectra: ", cumulative_spectra.shape)
    dist = np.array([euclidean(cumulative_spectra[0], cumulative_spectra[1])])
    # dist = pdist(cumulative_spectra)
    # print(dist.shape)

    return dist

def _ecs_dist(a, b):
    return ecs_dist(np.vstack((a, b)))

# Sanity check the ECS distance function...
print("ECS: ", ecs_dist(np.vstack((lesion_pixels[0], lesion_pixels[0]))))
print()
print("ECS: ", ecs_dist(np.vstack((lesion_pixels[0], lesion_pixels[1]))))
print()
print("ECS: ", ecs_dist(np.vstack((lesion_pixels[1], lesion_pixels[0]))))
print()
print("ECS: ", ecs_dist(np.vstack((lesion_pixels[0], lesion_pixels[2]))))

# agg = AgglomerativeClustering(n_clusters=8, affinity=ecs_dist, linkage="single")
# agg = AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="complete", memory="/root/output")
# labels = agg.fit_predict(lesion_pixels)

cumulative_spectra = np.cumsum(lesion_pixels, axis=1)
del lesion_pixels
del lesion_cube
dist_matrix = 1 / (1 + squareform(pdist(cumulative_spectra)))
print('dist_matrix', dist_matrix.shape)
# dist_matrix = squareform(pdist(lesion_pixels, _ecs_dist))

labels = spectral_clustering(dist_matrix, n_clusters=8)

rgb_image = plt.imread(str(output_dir / "rgb.png"))
rgb_image = rgb_image[crop_slice]

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)
ax1.imshow(rgb_image)
ax2.imshow(labels.reshape((m, n)))
plt.savefig(str(output_dir / "sc-ecs.png"))
