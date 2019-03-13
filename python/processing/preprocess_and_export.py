import warnings

import fpipy as fpi
import numpy as np

import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from pathlib import Path


data_dir = Path("/root/data/HSPC1/MM")
output_dir = Path("/root/output")

# Select a hyperspectral skin lesion image.
cube_dir = data_dir / "a965ccdcc83d466386649b1a21a927b1078a71bb"
# cube_dir = data_dir / "775d2a3131b92553009b2165e2cc4ee323b84c71"
# cube_dir = data_dir / "7bf233f5848c413fca645cc38b99283ab6d88641"

# Load the raw measurements of the lesion image, and convert to radiance.
raw_file = cube_dir / "RawMeasurementCube.hdr"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_dataset = fpi.read_ENVI_cfa(str(raw_file))

rad_dataset = fpi.raw_to_radiance(raw_dataset)
rad_cube = rad_dataset.radiance.values

print('rad_cube:')
print(rad_cube.min(), rad_cube.max())
print(rad_cube[200:1000, 500:1500].min(), rad_cube[200:1000, 500:1500].max())
print()

# Load the white reference image captured prior to capturing the lesion image,
# and convert to radiance.
raw_file = cube_dir / "WhiteReference.hdr"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_dataset = fpi.read_ENVI_cfa(str(raw_file))

ref_dataset = fpi.raw_to_radiance(raw_dataset)
ref_cube = ref_dataset.radiance.values

print('ref_cube:')
print(ref_cube.min(), ref_cube.max())
print(ref_cube[160:1160, 500:1500].min(), ref_cube[160:1160, 500:1500].max())
print()

# Crop the lesion image to remove lens cover, etc.
cropped = rad_cube[160:1160, 500:1500]

print('cropped:')
print(cropped.min(), cropped.max())
print(np.unravel_index(cropped.argmin(), cropped.shape))
print(np.unravel_index(cropped.argmax(), cropped.shape))
print()

# Calculate the normalized cube using the white reference cube.
normalized = cropped / ref_cube[160:1160, 500:1500]

print('normalized:')
print(normalized.min(), normalized.max())
print(np.unravel_index(normalized.argmin(), normalized.shape))
print(np.unravel_index(normalized.argmax(), normalized.shape))
print()

# Remove noisy wavelengths in the NIR range.
normalized = normalized[..., 0:115]

print('normalized (subset):')
print(normalized.min(), normalized.max())
print(np.unravel_index(normalized.argmin(), normalized.shape))
print(np.unravel_index(normalized.argmax(), normalized.shape))
print()

# Threshold normalized values to [0, 1].
normalized[normalized < 0] = 0
normalized[normalized > 1] = 1

print('normalized (clamped):')
print(normalized.min(), normalized.max())
print(np.unravel_index(normalized.argmin(), normalized.shape))
print(np.unravel_index(normalized.argmax(), normalized.shape))
print()

# Plot RGB image of skin lesion.
file_path = cube_dir / 'RGB_Image.png'
rgb_image = plt.imread(str(file_path))
rgb_cropped = rgb_image[160:1160, 500:1500]

# plt.figure()
# plt.imshow(rgb_cropped)
# plt.savefig(str(output_dir / 'rgb.png'))
plt.imsave(str(output_dir / 'rgb.png'), rgb_cropped)

npy_file = output_dir / 'processed.npy'
np.save(str(npy_file), normalized)
