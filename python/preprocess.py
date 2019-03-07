import fpipy as fpi
import numpy as np

from pathlib import Path

from preprocessing import (
    convert_to_radiance_cube,
    crop,
    extract_truecolor_image,
    load_raw_cube,
    normalize,
)


def main():
    data_dir = Path("/root/data/HSPC1/MM")
    cube_dir = data_dir / "a965ccdcc83d466386649b1a21a927b1078a71bb"
    # cube_dir = data_dir / "775d2a3131b92553009b2165e2cc4ee323b84c71"
    # cube_dir = data_dir / "7bf233f5848c413fca645cc38b99283ab6d88641"

    raw_hdr_file = cube_dir / "RawMeasurementCube.hdr"
    raw_cube = load_raw_cube(str(raw_hdr_file))

    # truecolor_image = extract_truecolor_image(raw_cube)
    # truecolor_image = crop(truecolor_image)
    # truecolor_image = truecolor_image / truecolor_image.max()

    radiance_cube = convert_to_radiance_cube(raw_cube)
    radiance_cube = crop(radiance_cube)

    # ---

    flat_hdr_file = cube_dir / "WhiteReference.hdr"
    flat_cube = load_raw_cube(str(flat_hdr_file))

    flat_radiance_cube = convert_to_radiance_cube(flat_cube)
    flat_radiance_cube = crop(flat_radiance_cube)

    print('flat radiance cube:',
          flat_radiance_cube.images.min(),
          flat_radiance_cube.images.max())

    normalized_cube = normalize(radiance_cube, flat_radiance_cube)

    print('normalized cube:',
          normalized_cube.images.min(),
          normalized_cube.images.max())

    # normalized_image = normalized_cube.images[..., image_index]
    # normalized_image = normalized_image / normalized_image.max()


if __name__ == "__main__":
    main()
