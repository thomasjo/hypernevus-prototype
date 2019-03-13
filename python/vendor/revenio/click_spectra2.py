# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:19:33 2017

@author: lauri.kangas
"""

"""

Script for plotting HSC spectra stats.

First, run hsc_reader. Running this script will plot an RGB image in figure 1,
and invite to click 4 points on figure 1. The smallest rectangle that fits
around the points will be the stat area.

A mean spectrum curve, together with +/- 1 std and min/max limits will be
plotted in figure 2.

The script can be run multiple times to get multiple spectra in figure 2.

"""

import matplotlib.pyplot as plt
import numpy as np

scale = rgb[:, ].max()

plt.figure(1)
plt.clf()
plt.imshow((rgb/scale)**(1/1.8))

plt.figure(1)

# pause and wait for 4 mouse clicks
points = plt.ginput(4)

# get smallest rectangle containing all four points
min_corner = np.array(points).min(axis=0).round()
max_corner = np.array(points).max(axis=0).round()

# slice full spectra from selected rectangle
spectra = cube[min_corner[1]:max_corner[1], min_corner[0]:max_corner[0], :]

axis = (0,1)

spectra_mean = spectra.mean(axis=axis)
spectra_std = spectra.std(axis=axis)

plt.figure(2)

lines, = plt.plot(wavelengths, spectra_mean, '.')

c = lines.get_color() # for plotting limits with same color

plt.fill_between(wavelengths, spectra_mean-spectra_std, spectra_mean+spectra_std, color=c, alpha = 0.5)
plt.fill_between(wavelengths, spectra.min(axis=axis), spectra.max(axis=axis), color=c, alpha=0.1)
