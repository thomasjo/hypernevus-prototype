import configparser
import math
import numpy as np
import spectral.io.envi as envi

import matplotlib.pyplot as plt


def process_image(image_hdr_path):
    raw_cube, hdt = load_raw_cube(image_hdr_path)
    raw_ref_cube, ref_hdt = load_reference_cube(image_hdr_path)

    x, y = (230, 45)
    output_width, output_height = 530, 530

    height, width, nlayers = raw_cube.shape
    # output_height = math.ceil(height / 2)
    # output_width = math.ceil(width / 2)

    rgb_layer = get_truecolor_image(raw_cube)
    ref_rgb_layer = get_truecolor_image(raw_ref_cube)

    rgb_layer = rgb_layer[y:(y + output_height), x:(x + output_width)]
    ref_rgb_layer = ref_rgb_layer[y:(y + output_height), x:(x + output_width)]
    assert rgb_layer.shape == ref_rgb_layer.shape

    nlayers = nlayers - 1  # Skip the last layer, which is RGB.
    max_bands = nlayers * 3

    # dark_layer = get_dark_layer(raw_cube, hdt)
    # cube = raw_cube - np.atleast_3d(dark_layer)
    # cube = raw_cube
    cube = raw_cube[..., 1:] - np.atleast_3d(raw_cube[..., 0])
    cube[cube < 0] = 0

    # dark_layer = get_dark_layer(ref_cube, hdt)
    # ref_cube = ref_cube - np.atleast_3d(dark_layer)
    ref_cube = raw_ref_cube[..., 1:] - np.atleast_3d(raw_ref_cube[..., 0])
    ref_cube[ref_cube < 0] = 0

    wavelengths = np.zeros(max_bands)
    corrected_images = np.ma.empty((output_height, output_width, max_bands))
    uncorrected_images = np.ma.empty_like(corrected_images)
    reference_images = np.ma.empty_like(corrected_images)

    assert corrected_images.shape[0:2] == rgb_layer.shape[0:2]

    band_index_offset = 0

    for layer_index in range(nlayers):
        layer_wavelengths, layer_band_images = bands_from_layer(
            cube, hdt, layer_index)
        ref_wavelengths, ref_band_images = bands_from_layer(
            ref_cube, ref_hdt, layer_index)

        if layer_wavelengths is None:
            continue

        assert np.allclose(layer_wavelengths, ref_wavelengths)

        npeaks = layer_band_images.shape[2]

        layer_band_images = layer_band_images[y:y+output_height, x:x+output_width]
        ref_band_images = ref_band_images[y:y+output_height, x:x+output_width]

        for peak_index in range(npeaks):
            idx = band_index_offset + peak_index
            wavelengths[idx] = layer_wavelengths[peak_index]
            uncorrected_images[..., idx] = layer_band_images[..., peak_index]
            reference_images[..., idx] = ref_band_images[..., peak_index]
            corrected_images[..., idx] = (layer_band_images[..., peak_index] /
                                          ref_band_images[..., peak_index])

        band_index_offset += npeaks

    nbands = band_index_offset
    wavelengths = wavelengths[0:nbands]
    corrected_images = corrected_images[..., 0:nbands]
    uncorrected_images = uncorrected_images[..., 0:nbands]
    reference_images = reference_images[..., 0:nbands]

    sorting_indices = np.argsort(wavelengths)

    wavelengths = wavelengths[sorting_indices]
    corrected_images = corrected_images[..., sorting_indices]
    uncorrected_images = uncorrected_images[..., sorting_indices]
    reference_images = reference_images[..., sorting_indices]

    return (wavelengths,
            corrected_images,
            uncorrected_images,
            rgb_layer,
            reference_images,
            ref_rgb_layer)


def load_raw_cube(hdr_path):
    cube = envi.open(hdr_path).asarray(float)
    hdt = hdt_for_cube(hdr_path)

    return cube, hdt


def load_reference_cube(hdr_path):
    hdr_path = hdr_path.replace('RawMeasurementCube', 'WhiteReference')
    cube, hdt = load_raw_cube(hdr_path)

    return cube, hdt


def hdt_for_cube(hdr_path):
    hdt = configparser.ConfigParser(converters={'array': convert_to_array})
    hdt_path = hdr_path.replace('.hdr', '.hdt')
    hdt.read(hdt_path)

    return hdt


def convert_to_array(raw_string):
    return np.array([float(x) for x in raw_string.strip('"').split()])


def get_dark_layer(cube, hdt):
    dark_layer = cube[..., 0]
    if hdt.getboolean('Header', 'Dark Layer Included', fallback=False):
        return dark_layer

    return np.zeros_like(dark_layer)


def bands_from_layer(cube, hdt, layer_index):
    hdt_section = hdt['Image{}'.format(layer_index + 1)]
    npeaks = hdt_section.getint('Npeaks')
    if npeaks < 1:
        return None, None

    exposure = hdt_section.getfloat('Exposure time (ms)')
    wavelengths = hdt_section.getarray('Wavelengths')
    sinvs = hdt_section.getarray('Sinvs').reshape((3, 3))

    bayer_frame = cube[..., layer_index]
    rgb_frame = demosaic(bayer_frame)

    usable_wavelengths = wavelengths[0:npeaks]
    band_images = np.matmul(rgb_frame, sinvs.T[:, 0:npeaks])
    band_images /= exposure

    return usable_wavelengths, band_images


def demosaic(bayer_frame):
    r = bayer_frame[0::2, 0::2]
    g = (bayer_frame[1::2, 0::2] + bayer_frame[0::2, 1::2]) / 2.0
    b = bayer_frame[1::2, 1::2]

    rgb = np.dstack((r, g, b))

    return rgb


def get_truecolor_image(cube):
    return demosaic(cube[..., -1])


def circular_mask(xy, width, height, radius):
    xs, ys = np.ogrid[0:height, 0:width]

    dist_from_xy = np.sqrt(np.power(xs - xy[1], 2) + np.power(ys - xy[0], 2))

    mask = dist_from_xy > radius
    mask = mask.reshape((height, width))

    return mask
