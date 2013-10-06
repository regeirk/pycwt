from __future__ import division
import numpy as np
from scipy import io
import kpywavelet as wavelet
import matplotlib.pyplot as plt

title = 'IDL signal'
fname = 'halpha.sav'
label = 'Area'
tlabel = 'Minutes'
t0 = 0
dt = 2.19/60.
units = 'km^5'

data = data_orig = io.idl.readsav(fname)['data_area_1']
std2 = data.std() ** 2
data = (data - data.mean()) / data.std() # Calculating anomaly and normalizing
time = np.arange(0, data.size) * dt + t0 # Time array in time units of your choice 
alpha, _, _ = wavelet.ar1(data) # Lag-1 autocorrelation for white noise
mother = wavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6

wave, scales, freqs, coi, dj, s0, J = wavelet.cwt(data, dt, dj=1./100, s0=-1, J=-1, wavelet=mother)
power = (np.abs(wave)) ** 2 # Normalized wavelet power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha, wavelet=mother)
sig95 = np.ones([1, data.size]) * signif[:, None]
sig95 = power / sig95 # Where ratio > 1, power is significant

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2  * power.mean(axis=1)
dof = data.size - scales # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2 , dt, scales, 1, alpha, 
                                        dof=dof, wavelet=mother)

# First sub-plot, the original time series anomaly.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, data_orig, 'k', linewidth=1.5)
ax.set_title('a) %s' % (title, ))
ax.set_ylabel('%s' % (label, ))

# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
bx = plt.axes([0.1, 0.15, 0.65, 0.45], sharex=ax)
levels=250
step = (np.max(power)**0.5 - np.min(power)**2) / levels
levels = np.arange(levels) * step + np.min(power)
im = bx.contourf(time, period, power, levels,
            extend='both')
bx.contour(time, period, sig95, [-99, 1], colors='k',
           linewidths=2.)
bx.fill(np.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
        time[:1]-dt, time[:1]-dt]), np.concatenate([[1e-9], coi,
        [1e-9], period[-1:], period[-1:], [1e-9]]), 'k', alpha=0.3,
        hatch='x')
bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
bx.set_ylabel('Period (Minutes)')
#cbarax = plt.axes([0.1, 0.05, 0.65, 0.05])
#plt.colorbar(im,cax=cbarax, ax=bx)

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = plt.axes([0.77, 0.15, 0.2, 0.45], sharey=bx)
cx.plot(glbl_signif, period, 'k--')
#cx.plot(fft_power, 1./fftfreqs, '-', color=[0.7, 0.7, 0.7],
#        linewidth=1.)
cx.plot(glbl_power, period, 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
cx.set_xlabel('Power')
cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim([period.min(), period.max()])
plt.setp(cx.get_yticklabels(), visible=False)

plt.show()