# -*- coding: iso-8859-1 -*-
"""
Python sample script for wavelet analysis and the statistical approach
suggested by Torrence and Compo (1998) using the wavelet module. To run
this script successfully, the matplotlib module has to be installed

"""
from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import kpywavelet as wavelet

# This script allows different sample data sets to be analysed. Simply comment
# and uncomment the respective fname, title, label, t0, dt and units variables
# to see the different results. t0 is the starting time, dt is the temporal
# sampling step

sample = 'NINO3' # Either NINO3, MAUNA, MONSOON, SUNSPOTS or SOI
if sample == 'NINO3':
    title = 'NINO3 Sea Surface Temperature (seasonal)'
    fname = 'sst_nino3.dat'
    label='NINO3 SST'
    t0=1871
    dt=0.25
    units='^{\circ}C'
elif sample == 'MAUNA':
    title = 'Mauna Loa Carbon Dioxide'
    fname = 'mauna.dat'
    label = 'Mauna Loa $CO_{2}$'
    t0=1958.0
    dt=0.08333333
    units='ppm'
elif sample == 'MONSOON':
    title = 'All-India Monsoon Rainfall'
    fname = 'monsoon.dat'
    label = 'Rainfall'
    t0 = 1871.0
    dt = 0.25
    units = 'mm'
elif sample == 'SUNSPOTS':
    title = 'Wolf\'s Sunspot Number'
    fname = 'sunspot.dat'
    label = 'Sunspots'
    t0 = 1748
    dt = 0.25
    units = ''
elif sample == 'SOI':
    title = 'Southern Oscillation Index'
    fname = 'soi.dat'
    label = 'SOI'
    t0 = 1896
    dt = 0.25
    units = 'mb'
else:
    raise Warning, 'No valid dataset chosen.'

data = np.loadtxt(fname)
sig_level = 0.95                        # Significance level

std = data.std()                      # Standard deviation
std2 = std ** 2                      # Variance
data = (data - data.mean()) / std       # Calculating anomaly and normalizing

N = data.size                         # Number of measurements
time = np.arange(0, N) * dt + t0  # Time array in correct units

dj = 1/12                            # Four sub-octaves per octaves
s0 = -1                      # Starting scale, here 6 months
J = -1                      # Seven powers of two with dj sub-octaves
alpha, g, _ = wavelet.ar1(data)

mother = wavelet.Morlet()          # Morlet mother wavelet with wavenumber=6
#mother = wavelet.Mexican_hat()       # Mexican hat wavelet, or DOG with m=2
#mother = wavelet.Paul()             # Paul wavelet with order m=4
#mother = wavelet.DOG()              # Derivative of the Gaussian, with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt, dj, s0, J,
                                                      mother)
power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=sig_level, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95                # Where ratio > 1, power is significant

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)
dof = N - scales                     # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                       significance_level=sig_level, dof=dof, wavelet=mother)

# Scale average between avg1 and avg2 periods and significance level
#sel = np.argwhere((period >= avg1) & (period < avg2))
Cdelta = mother.cdelta

# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.

plt.ion()

#fontsize = 'medium'
#params = {'text.fontsize': fontsize,
#          'xtick.labelsize': fontsize,
#          'ytick.labelsize': fontsize,
#          'axes.titlesize': fontsize,
#          'text.usetex': True
#         }
#mpl.rcParams.update(params)
fig = plt.figure(figsize=(11, 8), dpi=90)

# First sub-plot, the original time series.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, data, 'k', linewidth=1.5)
ax.set_title('a) %s' % (title, ))
if units != '':
  ax.set_ylabel(r'%s [$%s$]' % (label, units,))
else:
  ax.set_ylabel(r'%s' % (label, ))

# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area. Nonunifrom image was used just in case.
bx = plt.axes([0.1, 0.11, 0.65, 0.60], sharex=ax)           
im = NonUniformImage(bx, interpolation='nearest')
im.set_data(time, period, power)
bx.images.append(im)
bx.contour(time, period, sig95, colors='k',
           linewidths=2, origin='lower', interpolation='nearest', aspect = 'auto')

bx.fill(np.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
        time[:1]-dt, time[:1]-dt]), np.concatenate([[1e-9], coi,
        [1e-9], period[-1:], period[-1:], [1e-9]]), 'k', alpha='0.3',
        hatch='x')
bx.set_ylabel('Period (years)')

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = plt.axes([0.77, 0.11, 0.2, 0.60], sharey=bx)
cx.plot(glbl_signif, period, 'k--')
cx.plot(glbl_power, period, 'k-', linewidth=1.5)
if units != '':
  cx.set_xlabel(r'Power [$%s^2$]' % (units, ))
else:
  cx.set_xlabel(r'Power')
cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim([0,coi.max()])

plt.colorbar(im, cax=plt.axes([0.1, 0.05, 0.65, 0.01]), orientation='horizontal',extend='both')

plt.show()