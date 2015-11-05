"""
Python sample script for wavelet analysis and the statistical approach
suggested by Torrence and Compo (1998) using the wavelet module. To run
this script successfully, the matplotlib module has to be installed

DISCLAIMER
    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/
    and on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find
from matplotlib.image import NonUniformImage

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

var = np.loadtxt(fname)
avg1, avg2 = (2, 8)                  # Range of periods to average
slevel = 0.95                        # Significance level

std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (var - var.mean()) / std       # Calculating anomaly and normalizing

N = var.size                         # Number of measurements
time = np.arange(0, N) * dt + t0  # Time array in years

dj = 1/12                            # Twelve sub-octaves per octaves
s0 = -1#2 * dt                      # Starting scale, here 6 months
J = -1#7 / dj                      # Seven powers of two with dj sub-octaves
alpha = 0.0                          # Lag-1 autocorrelation for white noise
#alpha, _, _ = wavelet.ar1(var) # Lag-1 autocorrelation for red noise

mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)
power = (np.abs(wave)) ** 2  # Normalized wavelet power spectrum
period = 1/ freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95                # Where ratio > 1, power is significant
power /= scales[:, None]

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)
dof = N - scales                     # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                       significance_level=slevel, dof=dof, wavelet=mother)

# Scale average between avg1 and avg2 periods and significance level
sel = find((period >= avg1) & (period < avg2))
Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
# As in Torrence and Compo (1998) equation 24
scale_avg = power / scale_avg
scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                            significance_level=slevel, dof=[scales[sel[0]],
                            scales[sel[-1]]], wavelet=mother)

# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.
fig = plt.figure()

# First sub-plot, the original time series anomaly.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
ax.set_title('a) %s' % (title, ))
if units != '':
  ax.set_ylabel(r'%s [$%s$]' % (label, units,))
else:
  ax.set_ylabel(r'%s' % (label, ))

extent = [time.min(),time.max(),0,max(period)]
# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
<<<<<<< HEAD:pycwt/sample/sample.py
bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
im = NonUniformImage(bx, interpolation='bilinear', extent=extent)
im.set_data(time, period, power/scales[:, None])
bx.images.append(im)
bx.contour(time, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
bx.fill(np.concatenate([time, time[-1:]+dt, time[-1:]+dt,time[:1]-dt, time[:1]-dt]),
        (np.concatenate([coi,[1e-9], period[-1:], period[-1:], [1e-9]])),
        'k', alpha=0.3,hatch='x')

bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
bx.set_ylabel('Period (years)')

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, (period), 'k--')
cx.plot(glbl_power, (period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
if units != '':
  cx.set_xlabel(r'Power [$%s^2$]' % (units, ))
else:
  cx.set_xlabel(r'Power')
cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim(([period.min(), period.max()/2]))
plt.setp(cx.get_yticklabels(), visible=False)

# Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
# avg1 and avg2 parameters
dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(time, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) $%d$-$%d$ year scale-averaged power' % (avg1, avg2))
dx.set_xlabel('Time (year)')
if units != '':
  dx.set_ylabel(r'Average variance [$%s$]' % (units, ))
else:
  dx.set_ylabel(r'Average variance')

ax.set_xlim([time.min(), time.max()])
plt.show()
