import numpy as np
import spectral.io.envi as envi

from collections import namedtuple
from configparser import ConfigParser


RawCube = namedtuple("RawCube", ["data", "metadata"])
RawCubeMetadata = namedtuple("RawCubeMetadata", ["wavelengths"])
BandData = namedtuple("BandData", ["images", "wavelengths"])
RadianceCube = namedtuple("RadianceCube", ["images", "wavelengths"])


def load_raw_cube(hdr_file):
    data = envi.open(hdr_file).asarray()
    data = data.astype(np.float64)
    metadata = _read_metadata(hdr_file)
    cube = RawCube(data, metadata)

    return cube


def extract_truecolor_image(raw_cube):
    # The RGB exposure is stored at the end of the cube, depth-wise.
    rgb_frame = raw_cube.data[:, :, -1]
    rgb_image = _demosaic(rgb_frame)

    return rgb_image


def _subtract_dark_frame(cube):
    all_frames = cube.data.copy()

    # Remove the dark frame from all frames except the dark frame itself, and
    # the truecolor (RGB) frame; first and last frame, respectively.
    dark_frame = np.atleast_3d(all_frames[..., 0])
    all_frames[..., 1:-1] -= dark_frame

    # Negative values are physically impossible, so take care of offenders.
    all_frames[all_frames < 0] = 0

    denoised_cube = RawCube(all_frames, cube.metadata)

    return denoised_cube


def convert_to_radiance_cube(raw_cube):
    # Denoise the cube by subtracting the dark frame.
    raw_cube = _subtract_dark_frame(raw_cube)

    images = []
    wavelengths = []

    for layer_index in range(1, 85):
        band_data = bands_from_layer(raw_cube, layer_index)
        if band_data is None:
            continue

        images.extend(band_data.images)
        wavelengths.extend(band_data.wavelengths)

    sort_idx = np.argsort(wavelengths).squeeze()
    images = np.dstack(images)[..., sort_idx].squeeze()
    wavelengths = np.sort(np.hstack(wavelengths)).squeeze()

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


def normalize(cube, reference_cube):
    normalized_cube = RadianceCube(
        cube.images / reference_cube.images,
        cube.wavelengths,
    )

    return normalized_cube


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
    band_images = np.matmul(rgb_frame, sinvs.T[..., 0:npeaks]) / exposure

    band_data = BandData(
        np.split(band_images, npeaks, axis=2),
        np.split(usable_wavelengths, npeaks, axis=0),
    )

    return band_data


def convert_to_array(raw_string):
    return np.array([float(x) for x in raw_string.strip('"').split()])
