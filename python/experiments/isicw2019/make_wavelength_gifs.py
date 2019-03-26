from pathlib import Path

import imageio
import matplotlib
import numpy as np

from cluster_patches import find_image_file, load_image

data_dir = Path("/root/output/processed")
output_dir = Path("/root/output/isicw2019/gifs")

def convert_to_rgb(scalar_image):
    cmap = matplotlib.cm.get_cmap(name="viridis")
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap)

    return mapper.to_rgba(scalar_image, bytes=True, norm=False)


def make_wavelength_gif(image_file, crop_xy=None):
    image_data, _ = load_image(image_file, crop_xy)
    gif_file = output_dir / "{}.gif".format(image_file.stem)
    with imageio.get_writer(gif_file, mode="I", duration=0.1) as writer:
        for k in range(image_data.shape[2]):
            rgb_frame = convert_to_rgb(image_data[..., k])
            writer.append_data(rgb_frame)

    return gif_file


if __name__ == "__main__":
    # Ensure output directories exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    shasum_crop = {
        "775d2a3": (340, 360),
        "1425595": (364, 316),
        "f19607d": (260, 318),
        "766c5be": (169, 428),
    }

    for shasum, crop_xy in shasum_crop.items():
        image_file = find_image_file(shasum)
        gif_file = make_wavelength_gif(image_file, crop_xy)
        print(gif_file)
