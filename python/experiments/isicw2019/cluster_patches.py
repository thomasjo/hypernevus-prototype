import sys

from hashlib import sha1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean, pdist, squareform
from skimage.segmentation import mark_boundaries
from sklearn.cluster import spectral_clustering

PATCH_SIZE = 125

# Make experiments reproducible.
np.random.seed(42)

data_dir = Path("/root/output/processed")
output_dir = Path("/root/output/isicw2019/cluster_patches")
cache_dir = output_dir / ".cache"

# Ensure output directories exists.
output_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)


def find_image_file(shasum):
    # Search for image file based on shasum fragment.
    image_glob = data_dir.rglob("{}*.npy".format(shasum))
    image_file = next(image_glob, None)
    if not image_file:
        print("No matches for {}".format(shasum))
        exit(1)
    if next(image_glob, None):
        print("Found multiple matches for {}".format(shasum))
        exit(1)

    return image_file


def load_image(image_file, crop_xy=None):
    # Load hyperspectral cube.
    image_data = np.load(image_file)
    image_data = image_data[..., :-5]  # Discard noisy bands

    # Load RGB image.
    rgb_image = plt.imread(str(image_file.with_suffix(".png")))
    rgb_image = rgb_image[..., :3]  # Discard alpha channel

    # Crop if necessary.
    if crop_xy:
        image_data = crop_image(image_data, crop_xy)
        rgb_image = crop_image(rgb_image, crop_xy)

    return image_data, rgb_image


def crop_image(image_data, xy):
    # Crop cube based on patch size and xy coordinates.
    col_slice = slice(xy[0], xy[0] + PATCH_SIZE)
    row_slice = slice(xy[1], xy[1] + PATCH_SIZE)
    image_data = image_data[row_slice, col_slice]

    return image_data


def segment_pixels(pixels, dist_fn, n_clusters=3):
    dist_matrix = dist_fn(pixels)
    dist_matrix = 1 / (1 + dist_matrix)
    labels = spectral_clustering(dist_matrix, n_clusters=n_clusters)

    return labels


def _euclidean(pixels):
    return squareform(pdist(pixels))


def _ecs(pixels):
    return squareform(pdist(np.cumsum(pixels, axis=1)))


def _cache_id(*args):
    arg_string = "+".join([str(arg) for arg in args])
    shasum = sha1(arg_string.encode())
    shasum = shasum.hexdigest()

    return shasum[:7]


if __name__ == "__main__":
    shasum_crop = {
        "775d2a3": (340, 360),
        "1425595": (364, 316),
        "f19607d": (260, 318),
        "766c5be": (169, 428),
    }

    distances = {"euclidean": _euclidean, "ecs": _ecs}
    dist_colors = {"euclidean": (1, 0, 0), "ecs": (0, 1, 1)}

    # shasum, xy = next(iter(shasum_crop.items()))
    for shasum, crop_xy in shasum_crop.items():
        image_file = find_image_file(shasum)
        image_data, rgb_image = load_image(image_file, crop_xy)

        print("Processing: ", image_file)

        # NOTE: Check that we've cropped the correct region. ------------------
        debug_file = output_dir / "{}-debug.png".format(image_file.stem)
        plt.imsave(str(debug_file), image_data[..., -1])
        # ---------------------------------------------------------------------

        m, n, k = image_data.shape
        pixels = image_data.reshape((-1, k))

        comparison_image = rgb_image.copy()
        print(comparison_image.shape)

        for name, dist_fn in distances.items():
            print("  Segmenting with {} metric...".format(name))

            cache_id = _cache_id(shasum, crop_xy, name)
            cache_file = cache_dir / "{}.npy".format(cache_id)
            if cache_file.exists():
                print("    Using cached cluster labels")
                labels = np.load(cache_file)
            else:
                print("    Computing cluster labels...")
                labels = segment_pixels(pixels, dist_fn)
                np.save(cache_file, labels)

            label_image = labels.reshape((m, n))

            color = dist_colors.get(name, (1, 1, 0))
            comparison_image = mark_boundaries(comparison_image, label_image, color)

            # TODO: Extract as function.
            fig = plt.figure()
            ax1, ax2 = fig.subplots(1, 2)
            ax1.imshow(rgb_image)
            ax2.imshow(label_image)
            plt.savefig(str(output_dir / "{}-{}.png".format(image_file.stem, name)))

        comparison_file = output_dir / "{}-cmp.png".format(image_file.stem)
        plt.imsave(str(comparison_file), comparison_image)
