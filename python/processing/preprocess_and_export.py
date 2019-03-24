"""
preprocess_and_export.py

Usage:
  preprocess_and_export.py [--data-dir=<path>] [--output-dir=<path>]
                           [--overwrite]

Options:
  -h --help            Show this screen.
  --data-dir=<path>    The source directory to recursively glob for raw
                       hyperspectral cubes to process.
                       [default: /root/data/HSPC1]
  --output-dir=<path>  The output directory for all processed cubes.
                       [default: /root/output/processed]
  --overwrite          Overwrite all previously exported files.
"""

import sys
import warnings

import fpipy as fpi
import numpy as np

import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from docopt import docopt
from pathlib import Path


# Crop slices for images by row and column axes.
ROW_CROP = slice(160, 1160)
COL_CROP = slice(500, 1500)


def _radiance_cube(hdr_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_dataset = fpi.read_ENVI_cfa(str(hdr_file))

    rad_dataset = fpi.raw_to_radiance(raw_dataset)
    return rad_dataset.radiance.values[ROW_CROP, COL_CROP].copy()


def process_directory(data_dir, output_dir, overwrite=False):
    for raw_file in data_dir.rglob("RawMeasurementCube.hdr"):
        cube_dir = raw_file.parents[0]
        category = raw_file.parents[1].name

        output_file = output_dir / category / "{}.npy".format(cube_dir.name)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and overwrite is False:
            print("Skipping: {}".format(raw_file))
            continue

        print("Processing: {}".format(raw_file))

        # Load the raw measurements of the lesion image, and convert to
        # radiance.
        rad_cube = _radiance_cube(raw_file)

        # Load the white reference image captured prior to capturing the lesion
        # image, and convert to radiance.
        ref_file = cube_dir / "WhiteReference.hdr"
        ref_cube = _radiance_cube(ref_file)

        # Calculate the normalized cube using the white reference cube.
        normalized = (rad_cube / ref_cube).copy()
        del rad_cube
        del ref_cube

        # Remove noisy wavelengths in the NIR range.
        # normalized = normalized[..., 0:115]

        # Threshold normalized values to [0, 1].
        # normalized[normalized < 0] = 0
        # normalized[normalized > 1] = 1

        # Save processed cube as NumPy array.
        np.save(str(output_file), normalized)

        # Save RGB image of skin lesion.
        rgb_file = cube_dir / "RGB_Image.png"
        if rgb_file.exists():
            rgb_image = plt.imread(str(rgb_file))
            rgb_cropped = rgb_image[ROW_CROP, COL_CROP]
            rgb_output_file = output_file.with_suffix(".png")
            plt.imsave(str(rgb_output_file), rgb_cropped)


if __name__ == "__main__":
    args = docopt(__doc__)
    data_dir = Path(args["--data-dir"])
    output_dir = Path(args["--output-dir"])
    overwrite = args["--overwrite"]

    print()
    print("Source directory: {}".format(data_dir))
    print("Output directory: {}".format(output_dir))

    print("-" * 72)
    process_directory(data_dir, output_dir, overwrite)
    print("-" * 72)
