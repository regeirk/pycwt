"""
Python sample script for wavelet analysis and the statistical approach
suggested by Torrence and Compo (1998) using the wavelet module. To run
this script successfully, the matplotlib module has to be installed


Disclaimer
----------
This module is based on routines provided by C. Torrence and G. P. Compo
available at <http://paos.colorado.edu/research/wavelets/>, on routines
provided by A. Grinsted, J. Moore and S. Jevrejeva available at
<http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence>, and
on routines provided by A. Brazhe available at
<http://cell.biophys.msu.ru/static/swan/>.

This software is released under a BSD-style open source license. Please read
the license file for furter information. This routine is provided as is
without any express or implied warranties whatsoever.

Authors
-------
Sebastian Krieger, Nabil Freij

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy
from matplotlib import pyplot

import pycwt as wavelet
from pycwt.helpers import find

from dataset import Dataset

# Here we use the dataset class to load the data. Valid datasets are either
# NINO3, MAUNA, MONSOON, SUNSPOTS or SOI. If your `matplotlib` allows LaTeX
# text formatting, change the `usetex` parameter to `True`.
sample = 'NINO3'
usetex = True
ds = Dataset(sample, usetex=usetex)
dat = ds.load()

avg1, avg2 = (2, 8)                  # Range of periods to average
slevel = 0.95                        # Significance level

std = dat.std()                      # Standard deviation
std2 = std ** 2                      # Variance
dat = (dat - dat.mean()) / std       # Calculating anomaly and normalizing

N = dat.size                            # Number of measurements
time = numpy.arange(0, N) * ds.dt + ds.t0  # Time array in years

dj = 1 / 12                          # Twelve sub-octaves per octaves
s0 = -1  # 2 * dt                    # Starting scale, here 6 months
J = -1  # 7 / dj                     # Seven powers of two with dj sub-octaves
#  alpha = 0.0                       # Lag-1 autocorrelation for white noise
try:
    alpha, _, _ = wavelet.ar1(dat)   # Lag-1 autocorrelation for red noise
except Warning:
    # When the dataset is too short, or there is a strong trend, ar1 raises a
    # warning. In this case, we assume a white noise background spectrum.
    alpha = 1.0

mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat, ds.dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, ds.dt, dj, mother)

# Normalized wavelet and Fourier power spectra
power = (numpy.abs(wave)) ** 2
fft_power = numpy.abs(fft) ** 2
period = 1 / freqs

# Significance test. Where ratio power/sig95 > 1, power is significant.
signif, fft_theor = wavelet.significance(1.0, ds.dt, scales, 0, alpha,
                                         significance_level=slevel,
                                         wavelet=mother)
sig95 = numpy.ones([1, N]) * signif[:, None]
sig95 = power / sig95

# Power rectification as of Liu et al. (2007). TODO: confirm if significance
# test ratio should be calculated first.
# power /= scales[:, None]

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = power.mean(axis=1)
dof = N - scales                     # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2, ds.dt, scales, 1, alpha,
                                        significance_level=slevel, dof=dof,
                                        wavelet=mother)

# Scale average between avg1 and avg2 periods and significance level
sel = find((period >= avg1) & (period < avg2))
Cdelta = mother.cdelta
scale_avg = (scales * numpy.ones((N, 1))).transpose()
# As in Torrence and Compo (1998) equation 24
scale_avg = power / scale_avg
scale_avg = std2 * dj * ds.dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(std2, ds.dt, scales, 2, alpha,
                                             significance_level=slevel,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)

# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.
pyplot.close('all')
pyplot.ioff()
params = {
          'font.size': 13.0,
          'text.usetex': usetex,
          'text.fontsize': 12,
          'axes.grid': True,
         }
pyplot.rcParams.update(params)
figprops = dict(figsize=(11, 8), dpi=72)
fig = pyplot.figure(**figprops)

# First sub-plot, the original time series anomaly and inverse wavelet
# transform.
ax = pyplot.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, dat, 'k', linewidth=1.5)
ax.set_title('a) {}'.format(ds.title))
if ds.units != '':
    ax.set_ylabel(r'{} [{}]'.format(ds.label, ds.units))
else:
    ax.set_ylabel(r'{}'.format(ds.label))

# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area. Note that period scale is
# logarithmic.
bx = pyplot.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(time, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
            extend='both', cmap=pyplot.cm.viridis)
extent = [time.min(), time.max(), 0, max(period)]
bx.contour(time, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
bx.fill(numpy.concatenate([time, time[-1:] + ds.dt, time[-1:] + ds.dt,
                           time[:1] - ds.dt, time[:1] - ds.dt]),
        numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                           numpy.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(ds.label,
                                                        mother.name))
bx.set_ylabel('Period (years)')
#
Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                           numpy.ceil(numpy.log2(period.max())))
bx.set_yticks(numpy.log2(Yticks))
bx.set_yticklabels(Yticks)
# bx.invert_yaxis()

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = pyplot.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, numpy.log2(period), 'k--')
cx.plot(std2 * fft_theor, numpy.log2(period), '--', color='#cccccc')
cx.plot(std2 * fft_power, numpy.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(std2 * glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
if ds.units2 != '':
    cx.set_xlabel(r'Power [{}]'.format(ds.units2))
else:
    cx.set_xlabel(r'Power')
cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim(numpy.log2([period.min(), period.max()]))
cx.set_yticks(numpy.log2(Yticks))
cx.set_yticklabels(Yticks)
pyplot.setp(cx.get_yticklabels(), visible=False)
# cx.invert_yaxis()

# Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
# avg1 and avg2 parameters
dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(time, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) ${}$--${}$ year scale-averaged power'.format(avg1, avg2))
dx.set_xlabel('Time (year)')
if ds.units != '':
    dx.set_ylabel(r'Average variance [{}]'.format(ds.units))
else:
    dx.set_ylabel(r'Average variance')
ax.set_xlim([time.min(), time.max()])

fig.savefig('sample_{}.png'.format(sample), dpi=96)

pyplot.show()
