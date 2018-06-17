import matplotlib.pyplot as plt

from preprocessing import *
from pathlib import Path


def main():
    data_dir = Path("/root/data/examples")
    # cube_dir = data_dir / "a965ccdcc83d466386649b1a21a927b1078a71bb"
    # cube_dir = data_dir / "d5d370809f55d2c427930e8d8bd123295013d594"
    cube_dir = data_dir / "efb2352b7dca13d17962a9f9e0a6c94c13e849a3"
    raw_hdr_file = cube_dir / "RawMeasurementCube.hdr"
    raw_cube = load_raw_cube(str(raw_hdr_file))

    truecolor_image = extract_truecolor_image(raw_cube)
    truecolor_image = crop(truecolor_image)
    truecolor_image = truecolor_image / truecolor_image.max()

    plt.figure()
    plt.imshow(truecolor_image)
    plt.show()

    radiance_cube = convert_to_radiance_cube(raw_cube)
    radiance_cube = crop(radiance_cube)

    print(
        'radiance cube:',
        radiance_cube.images.min(),
        radiance_cube.images.max(),
    )

    image_index = 80
    radiance_image = radiance_cube.images[..., image_index]
    radiance_image = radiance_image / radiance_image.max()
    wavelength = radiance_cube.wavelengths[image_index]

    print('wavelength:', wavelength)

    plt.figure()
    plt.imshow(radiance_image)
    plt.show()

    # ---

    flat_hdr_file = cube_dir / "WhiteReference.hdr"
    flat_cube = load_raw_cube(str(flat_hdr_file))

    flat_radiance_cube = convert_to_radiance_cube(flat_cube)
    flat_radiance_cube = crop(flat_radiance_cube)

    print(
        'flat radiance cube:',
        flat_radiance_cube.images.min(),
        flat_radiance_cube.images.max(),
    )

    normalized_cube = normalize(radiance_cube, flat_radiance_cube)

    print(
        'normalized cube:',
        normalized_cube.images.min(),
        normalized_cube.images.max(),
    )

    normalized_image = normalized_cube.images[..., image_index]
    normalized_image = normalized_image / normalized_image.max()

    plt.figure()
    plt.imshow(normalized_image)
    plt.show()


if __name__ == '__main__':
    main()
