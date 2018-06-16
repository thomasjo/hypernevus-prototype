import numpy as np
import spectral.io.envi as envi

from collections import namedtuple
from configparser import ConfigParser
from pathlib import Path


RawCube = namedtuple("RawCube", ["data", "metadata"])
RawCubeMetadata = namedtuple("RawCubeMetadata", ["wavelengths"])
BandData = namedtuple("BandData", ["images", "wavelengths"])
RadianceCube = namedtuple("RadianceCube", ["images", "wavelengths"])


def load_raw_cube(hdr_file):
    data = envi.open(str(hdr_file)).asarray(np.float)
    metadata = _read_metadata(hdr_file)
    cube = RawCube(data, metadata)

    return cube


def extract_truecolor_image(raw_cube):
    # The RGB exposure is stored at the end of the cube, depth-wise.
    rgb_frame = raw_cube.data[:, :, -1]
    rgb_image = _demosaic(rgb_frame)

    return rgb_image


# def subtract_dark_frame(cube):
#     dark_frame = cube[..., 0]
#     data = cube.data
#     data[..., 1:-1] -= dark_frame
#
#     return cube


def convert_to_radiance_cube(raw_cube):
    dark_frame = raw_cube.data[..., 0]

    images = []
    wavelengths = []

    for layer_index in range(1, 85):
        band_data = bands_from_layer(raw_cube, layer_index)
        if band_data is None:
            continue

        images.append(band_data.images)
        wavelengths.extend(band_data.wavelengths)

    images = np.dstack(images)
    wavelengths = wavelengths.sort()

    radiance_cube = RadianceCube(images, wavelengths)

    return radiance_cube


def crop(target):
    if isinstance(target, np.ndarray):
        cropped = target[60:560, 220:720]
    elif isinstance(target, RadianceCube):
        images = target.images[60:560, 220:720]
        cropped = RadianceCube(images, target.wavelengths)
    else:
        raise Error("Unsupported type for argument 'target'.")

    return cropped


def normalize():
    pass


def _read_metadata(hdr_file):
    hdt = ConfigParser(converters={'array': convert_to_array})
    hdt_file = str(hdr_file).replace('.hdr', '.hdt')
    hdt.read(hdt_file)

    return hdt


def _demosaic(bayer_frame):
    r = (bayer_frame[0::2, 0::2])
    g = (bayer_frame[1::2, 0::2] + bayer_frame[0::2, 1::2]) / 2
    b = (bayer_frame[1::2, 1::2])

    rgb = np.dstack((r, g, b))

    return rgb


def bands_from_layer(cube, layer_index):
    hdt_section = cube.metadata['Image{}'.format(layer_index + 1)]
    npeaks = hdt_section.getint('Npeaks')
    if npeaks < 1:
        return None

    exposure = hdt_section.getfloat('Exposure time (ms)')
    wavelengths = hdt_section.getarray('Wavelengths')
    sinvs = hdt_section.getarray('Sinvs').reshape((3, 3))

    bayer_frame = cube.data[..., layer_index]
    rgb_frame = _demosaic(bayer_frame)

    usable_wavelengths = wavelengths[0:npeaks]
    band_images = np.matmul(rgb_frame, sinvs.T[:, 0:npeaks])
    band_images /= exposure

    band_data = BandData(band_images, usable_wavelengths)

    return band_data


def convert_to_array(raw_string):
    return np.array([float(x) for x in raw_string.strip('"').split()])


# TODO: Temporary test. Remove ASAP.
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_dir = Path("/root/data/examples")
    cube_dir = data_dir / "d5d370809f55d2c427930e8d8bd123295013d594"
    raw_hdr_file = cube_dir / "RawMeasurementCube.hdr"
    raw_cube = load_raw_cube(raw_hdr_file)

    truecolor_image = extract_truecolor_image(raw_cube)
    truecolor_image = crop(truecolor_image)
    truecolor_image = truecolor_image / truecolor_image.max()

    plt.figure()
    plt.imshow(truecolor_image)
    plt.show()

    radiance_cube = convert_to_radiance_cube(raw_cube)
    radiance_cube = crop(radiance_cube)
    print(radiance_cube.wavelengths)

    image_index = 40
    radiance_image = radiance_cube.images[..., image_index]
    wavelength = radiance_cube.wavelengths[image_index]
    print(wavelength)

    plt.figure()
    plt.imshow(radiance_image)
    plt.show()
