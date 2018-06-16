# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:00:11 2017

@author: lauri.kangas
"""

import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import configparser


def get_dark_frame(cube, hdt=None):
    if hdt and check_dark_included(hdt):
        return None

    return cube[:, :, 0]


def check_dark_included(hdt):
    try:
        dark_included = (hdt['Header']['Dark Layer Included'].lower() == 'true')
    except KeyError:
        return False

    return dark_included


def subtract_dark(cube):
    dark = get_dark_frame(cube)

    # make cube an ndarray in case the input cube was BSQFile and not ndarray
    cube_ar = cube[:,:,:]

    return cube_ar - np.broadcast_to(dark, cube_ar.shape)


def demosaic(cfa, method='bin'):
    if method == 'bin':
        red = cfa[::2, ::2].astype(float)

        green1 = cfa[1::2, ::2]
        green2 = cfa[::2, 1::2]
        green = (green1 + green2)/2

        blue = cfa[1::2, 1::2].astype(float)

        rgb = np.dstack((red, green, blue))

        return rgb


def get_bands_from_layer(cube, hdt, index):

    # one-based indices according to HDT file sections
    section = "Image{}".format(index)

    Npeaks = int(hdt[section]['Npeaks'])

    if Npeaks < 1:
        return (None, None)

    # e.g. Image6 found at layer 5
    frame = cube[:,:,index-1]

    sinvs_str = hdt[section]['Sinvs'] # "1 2 3 0 0 0 0 0 0"
    sinvs_str = sinvs_str.strip('"') # 1 2 3 0 0 0 0 0 0
    sinvs = [float(x) for x in sinvs_str.split()] # [1,2,3,0,0,0,0,0,0]
    sinvs = np.array(sinvs) # array([1,2,3,0,0,0,0,0,0])
    sinvs = sinvs.reshape((3,3)) # array([[1,2,3],[0,0,0],[0,0,0])

    wavelengths_str = hdt[section]['Wavelengths'] # "450 650 0"
    wavelengths_str = wavelengths_str.strip('"')
    frame_wavelengths = [float(x) for x in wavelengths_str.split()] # [450, 650, 0]



    rgb = demosaic(frame) # (N, M, 3) shape

    # reshape to (3, N*M) so matrix product can be applied directly
    # -> all spatial pixels on single row, for each R, G, B
    rgb2 = rgb.reshape((np.product(rgb.shape[:2]),3)).T

    # all spatial pixels on single row, for each wavelength band (1 or more)
    S = sinvs.dot(rgb2)

    usable_wavelengths = []
    band_images = []

    for k in range(Npeaks):
        usable_wavelengths.append(frame_wavelengths[k])

        image_vector = S[k] # image data for single wavelength band, in one long vector
        image_array = image_vector.reshape(rgb.shape[:2]) # (M, N) b/w image

        band_images.append(image_array)

    return usable_wavelengths, band_images


def get_truecolor_image(cube):
    return demosaic(cube[:,:,-1])

# -----------------------------------------------------------------------------
prefix = '/root/data/examples'
file_name = 'd5d370809f55d2c427930e8d8bd123295013d594'
file_path = '{0}/{1}/RawMeasurementCube.hdr'.format(prefix, file_name)
# -----------------------------------------------------------------------------

raw_cube = envi.open(file_path)
raw_hdt = configparser.ConfigParser()
raw_hdt.read(file_path.replace('.hdr', '.hdt'))

# -----------------------------------------------------------------------------
file_path = file_path.replace('RawMeasurementCube', 'WhiteReference')
# -----------------------------------------------------------------------------

flat_cube = envi.open(file_path)
flat_hdt = configparser.ConfigParser()
flat_hdt.read(file_path.replace('.hdr', '.hdt'))

#raw_dark = get_dark_frame(raw_cube, raw_hdt)
#flat_dark = get_dark_frame(flat_cube, flat_hdt)

assert check_dark_included(raw_hdt)
assert check_dark_included(flat_hdt)

raw = subtract_dark(raw_cube)
flat = subtract_dark(flat_cube)

wavelengths = []
corrected_images = []
uncorrected_images = []

for k in range(1,86):
    layer_wavelengths, layer_band_images \
        = get_bands_from_layer(raw, raw_hdt, k)
    flat_layer_wavelengths, flat_layer_band_images \
        = get_bands_from_layer(flat, flat_hdt, k)

    print(k, layer_wavelengths)

    if not layer_wavelengths:
        continue

    # make sure flat data has same bands
    assert layer_wavelengths == flat_layer_wavelengths

    uncorrected_images_short = []
    flat_corrected_images_short = []

    # iterate over the wavelength of a single image (0 to 3 bands)
    for image, flat_image in zip(layer_band_images, flat_layer_band_images):
        uncorrected_images_short.append(image)
        flat_corrected_images_short.append(image/flat_image)

    # append python lists (no summation)
    wavelengths += layer_wavelengths
    corrected_images += flat_corrected_images_short
    uncorrected_images += uncorrected_images_short

cube = np.dstack(corrected_images)
noflat_cube = np.dstack(uncorrected_images)

sorted_wavelength_indices = np.argsort(wavelengths)
cube = cube[:,:,sorted_wavelength_indices]
noflat_cube = noflat_cube[:,:,sorted_wavelength_indices]

wavelengths = np.sort(wavelengths)

# rgb image is fetched from the last layer of the raw cube (not wavelength cube)
rgb = get_truecolor_image(raw_cube)

# NOTE: Not code from Revenio. Added to aid with debugging.
rgb = rgb / rgb[:, ].max()
plt.figure()
plt.imshow(rgb)
# plt.savefig('/images/fig.png')
plt.show()
