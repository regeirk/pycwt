from __future__ import division
import numpy as np
import kpywavelet as wavelet
import matplotlib.pyplot as plt

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
    raise ValueError('No valid dataset chosen.')

data = data_orig = np.loadtxt(fname)
#avg1, avg2 = (2, 8) # Range of periods to average
slevel = 0.95 # Significance level

std = data.std() # Standard deviation
std2 = std ** 2 # Variance
data = (data - data.mean()) / std # Calculating anomaly and normalizing

N = data.size # Number of measurements
time = np.arange(0, N) * dt + t0 # Time array in years

dj = 0.1 # Four sub-octaves per octaves
s0 = -1 # 2 * dt # Starting scale, here 6 months
J = -1 # 7 / dj # Seven powers of two with dj sub-octaves
#alpha = 0.0 # Lag-1 autocorrelation for white noise
alpha, _, _ = wavelet.ar1(data)

mother = wavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6
#mother = wavelet.Mexican_hat() # Mexican hat wavelet, or DOG with m=2
#mother = wavelet.Paul(4) # Paul wavelet with order m=4
#mother = wavelet.DOG(6) # Derivative of the Gaussian, with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt=dt, dj=dj, s0=s0, J=J, wavelet=mother)
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
levels=50
step = (np.max(power) - np.min(power)**2) / levels
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
cbarax = plt.axes([0.1, 0.05, 0.65, 0.05])
plt.colorbar(im,cax=cbarax, ax=bx,orientation='horizontal')

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