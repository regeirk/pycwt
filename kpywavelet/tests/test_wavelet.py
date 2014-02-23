from __future__ import division, absolute_import

import numpy as np
import kpywavelet as wavelet
import matplotlib.pyplot as plt
from kpywavelet.helpers import boxpdf
#
#sample = 'NINO3' # Either NINO3, MAUNA, MONSOON, SUNSPOTS or SOI
#if sample == 'NINO3':
#    title = 'NINO3 Sea Surface Temperature (seasonal)'
#    fname = 'sst_nino3.dat'
#    label='NINO3 SST'
#    t0=1871
#    dt=0.25
#    units='^{\circ}C'
#elif sample == 'MAUNA':
#    title = 'Mauna Loa Carbon Dioxide'
#    fname = 'mauna.dat'
#    label = 'Mauna Loa $CO_{2}$'
#    t0=1958.0
#    dt=0.08333333
#    units='ppm'
#elif sample == 'MONSOON':
#    title = 'All-India Monsoon Rainfall'
#    fname = 'monsoon.dat'
#    label = 'Rainfall'
#    t0 = 1871.0
#    dt = 0.25
#    units = 'mm'
#elif sample == 'SUNSPOTS':
#    title = 'Wolf\'s Sunspot Number'
#    fname = 'sunspot.dat'
#    label = 'Sunspots'
#    t0 = 1748
#    dt = 0.25
#    units = ''
#elif sample == 'SOI':
#    title = 'Southern Oscillation Index'
#    fname = 'soi.dat'
#    label = 'SOI'
#    t0 = 1896
#    dt = 0.25
#    units = 'mb'
#else:
#    raise ValueError('No valid dataset chosen.')
#
#data = data_orig = np.loadtxt(fname)
##avg1, avg2 = (2, 8) # Range of periods to average
#slevel = 0.95 # Significance level
#
#std = data.std() # Standard deviation
#std2 = std ** 2 # Variance
#data = (data - data.mean()) / std # Calculating anomaly and normalizing
#
#N = data.size # Number of measurements
#time = np.arange(0, N) * dt + t0 # Time array in years
#
#dj = 0.1 # Four sub-octaves per octaves
#s0 = -1 # 2 * dt # Starting scale, here 6 months
#J = -1 # 7 / dj # Seven powers of two with dj sub-octaves
##alpha = 0.0 # Lag-1 autocorrelation for white noise
#alpha, _, _ = wavelet.ar1(data)
#
#mother = wavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6
##mother = wavelet.Mexican_hat() # Mexican hat wavelet, or DOG with m=2
##mother = wavelet.Paul(4) # Paul wavelet with order m=4
##mother = wavelet.DOG(6) # Derivative of the Gaussian, with m=6
#
## The following routines perform the wavelet transform and siginificance
## analysis for the chosen data set.
#wave, scales, s0, J, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt=dt, dj=dj, s0=s0, J=J, wavelet=mother)
#power = (np.abs(wave)) ** 2 # Normalized wavelet power spectrum
#period = 1. / freqs
#
#signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha, wavelet=mother)
#sig95 = np.ones([1, data.size]) * signif[:, None]
#sig95 = power / sig95 # Where ratio > 1, power is significant
#
## Calculates the global wavelet spectrum and determines its significance level.
#glbl_power = std2  * power.mean(axis=1)
#dof = data.size - scales # Correction for padding at edges
#glbl_signif, tmp = wavelet.significance(std2 , dt, scales, 1, alpha,
#                                        dof=dof, wavelet=mother)
#
## First sub-plot, the original time series anomaly.
#ax = plt.axes([0.1, 0.75, 0.65, 0.2])
#ax.plot(time, data_orig, 'k', linewidth=1.5)
#ax.set_title('a) %s' % (title, ))
#ax.set_ylabel('%s' % (label, ))
#
## Second sub-plot, the normalized wavelet power spectrum and significance level
## contour lines and cone of influece hatched area.
#bx = plt.axes([0.1, 0.15, 0.65, 0.45], sharex=ax)
#levels=50
#step = (np.max(power) - np.min(power)**2) / levels
#levels = np.arange(levels) * step + np.min(power)
#im = bx.contourf(time, period, power, levels)
#bx.contour(time, period, sig95, [-99, 1], colors='k',
#           linewidths=2.)
#bx.fill(np.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
#        time[:1]-dt, time[:1]-dt]), np.concatenate([[1e-9], coi,
#        [1e-9], period[-1:], period[-1:], [1e-9]]), 'k', alpha=0.3,
#        hatch='x')
#bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
#bx.set_ylabel('Period (Minutes)')
#cbarax = plt.axes([0.1, 0.05, 0.65, 0.05])
#plt.colorbar(im,cax=cbarax, ax=bx,orientation='horizontal')
#
## Third sub-plot, the global wavelet and Fourier power spectra and theoretical
## noise spectra.
#cx = plt.axes([0.77, 0.15, 0.2, 0.45], sharey=bx)
#cx.plot(glbl_signif, period, 'k--')
##cx.plot(fft_power, 1./fftfreqs, '-', color=[0.7, 0.7, 0.7],
##        linewidth=1.)
#cx.plot(glbl_power, period, 'k-', linewidth=1.5)
#cx.set_title('c) Global Wavelet Spectrum')
#cx.set_xlabel('Power')
#cx.set_xlim([0, glbl_power.max() + std2])
#cx.set_ylim([period.min(), period.max()])
#plt.setp(cx.get_yticklabels(), visible=False)
#
#plt.show()


####
"""
sample_xwt
"""
#data1 = dict(
#    name = 'Arctic Oscillation',
#    nick = 'AO',
#    file = 'jao.dat',
#    alpha = 0.0)
#data2 = dict(
#    name = 'Baltic Sea ice extent',
#    nick = 'BMI',
#    file='jbaltic.dat',
#    alpha = 0.0)
#
#t1, s1 = np.loadtxt(data1['file'], unpack=True)
#t2, s2 = np.loadtxt(data2['file'], unpack=True)
#dt = np.diff(t1)[0]
#dt2 = np.diff(t1)[0]
#n1 = t1.size
#n2 = t2.size
#n = min(n1, n2)

t1 = time = np.linspace(0,500,500)
s1 = np.sin(np.pi * 2 * time / 100) + np.random.random(len(time))
s2 = np.cos(np.pi * 2 * time / 100) + np.random.random(len(time))
dt = np.diff(time)[0]
dt2 = np.diff(time)[0]
n1 = time.size
n2 = time.size
n = min(n1, n2)


# Change the probablity density function (PDF) of the data. The time series
# of Baltic Sea ice extent is highly bi-modal and we therefore transform the
# timeseries into a series of percentiles. The transformed series probably
# reacts 'more linearly' to climate.
#s2, _, _ = boxpdf(s2)
s2 = s2[0:len(s1)]
# Calculates the standard deviatio of each time series for later
# normalization.
std1 = s1.std()
std2 = s2.std()

# Calculate the CWT of both normalized time series. The function wavelet.cwt
# returns a a list with containing [wave, scales, freqs, coi, fft, fftfreqs]
# variables.
#mother = 'morlet'
#cwt1 = wavelet.cwt(s1 / std1, dt, wavelet=mother)
#sig1 = wavelet.significance(1.0, dt, cwt1[1], 0, data1['alpha'],
#    wavelet=mother)
#cwt2 = wavelet.cwt(s2 / std2, dt, wavelet=mother)
#sig2 = wavelet.significance(1.0, dt, cwt2[1], 0, data1['alpha'],
#    wavelet=mother)

# Calculate the cross wavelet transform (XWT). The XWT finds regions in time
# frequency space where the time series show high common power. Torrence and
# Compo (1998) state that the percent point function -- PPF (inverse of the
# cumulative distribution function) of a chi-square distribution at 95%
# confidence and two degrees of freedom is Z2(95%)=3.999. However, calculating
# the PPF using chi2.ppf gives Z2(95%)=5.991. To ensure similar significance
# intervals as in Grinsted et al. (2004), one has to use confidence of 86.46%.
W12, sj, freqs, coi, dj, s0, J, signif = wavelet.xwt(s1, s2, dt, significance_level=0.8646, dj=1./12, s0=-1, J=-1,
        wavelet='morlet', normalize=True)

# Calculate the wavelet coherence (WTC). The WTC finds regions in time
# frequency space where the two time seris co-vary, but do not necessarily have
# high power.
WCT, coi, freqs, sig, aWCT = wavelet.wct(s1, s1, dt, significance_level=0.8646, dj=dj, s0=s0, J=J,
        wavelet='morlet', normalize=False)

power = (np.abs(WCT)) ** 2 # Normalized wavelet power spectrum
period = 1. / freqs

# Do the plotting!
da = [3, 3]
levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]

angle = 0.5 * np.pi - np.angle(W12)
u, v = np.cos(angle), np.sin(angle)

fig = plt.figure()
plt.contourf(t1, period, power, levels)
plt.contour(t1, period, sig[1], [-99, 1], colors='k',
    linewidths=2.)
#q = plt.quiver(t1[::da[1]], np.log2(period)[::da[0]], u[::da[0], ::da[1]],
#    v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid',
#    linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
#    headaxislength=5, minshaft=2, minlength=5)
#plt.fill(np.concatenate([t1[:1]-dt, t1, t1[-1:]+dt, t1[-1:]+dt, t1[:1]-dt,
#    t1[:1]-dt]), np.log2(np.concatenate([[1e-9], coi, [1e-9],
#    period[-1:], period[-1:], [1e-9]])), 'k', alpha='0.3', hatch='x')
#Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
#    np.ceil(np.log2(period.max())))
#ax.set_yticks(np.log2(Yticks))
#ax.set_yticklabels(Yticks)
#ax.set_xlim([t.min(), t.max()])
#ax.set_ylim(np.log2([period.min(), min([coi.max(), period.max()])]))
#ax.invert_yaxis()
#cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
#cbar.ax.set_yticklabels(labels)