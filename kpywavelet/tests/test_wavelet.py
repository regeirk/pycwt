import numpy as np
import kpywavelet as wavelet
import matplotlib.pyplot as plt

sample = 'JBALTIC' # Either NINO3, MAUNA, MONSOON, SUNSPOTS, SOI, JAO or JBALTIC
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
elif sample == 'JAO':
    title = 'Arctic Oscillation'
    fname = 'jao.dat'
    label = 'AO'
    t0 = 1851
    dt = 1
    units = ''
elif sample == 'JBALTIC':
    title = 'Baltic Sea ice extent'
    fname = 'jbaltic.dat'
    label = 'BMI'
    t0 = 1720
    dt = 1
    units = ''
else:
    raise Warning, 'No valid dataset chosen.'

data = np.loadtxt(fname)
if data.shape[1] == 2: # For the last two data sets which contain time and data
    data = np.asarray(zip(*data)[1])  

data = (data - data.mean()) / data.std # Calculating anomaly and normalizing
time = np.arange(0, data.size) * dt + t0 # Time array in time units of your choice 
alpha, _, _ = wavelet.ar1(data) # Lag-1 autocorrelation for white noise
mother = wavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6

wave, scales, freqs, coi, dj, s0, J = wavelet.cwt(data, mother)
power = (np.abs(wave)) ** 2 # Normalized wavelet power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha, wavelet=mother)
sig95 = np.ones([1, data.size]) * signif[:, None]
sig95 = power / sig95 # Where ratio > 1, power is significant

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = data.std ** 2  * power.mean(axis=1)
dof = data.size - scales # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(data.std ** 2 , dt, scales, 1, alpha, 
                                        dof=dof, wavelet=mother)

# First sub-plot, the original time series anomaly.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, data, 'k', linewidth=1.5)
ax.set_title('a) %s' % (title, ))
if units != '':
  ax.set_ylabel(r'%s [$%s$]' % (label, units,))
else:
  ax.set_ylabel(r'%s' % (label, ))

# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
bx = pylab.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(time, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
            extend='both')
bx.contour(time, numpy.log2(period), sig95, [-99, 1], colors='k',
           linewidths=2.)
bx.fill(numpy.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
        time[:1]-dt, time[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi,
        [1e-9], period[-1:], period[-1:], [1e-9]])), 'k', alpha='0.3',
        hatch='x')
bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
bx.set_ylabel('Period (years)')
Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                           numpy.ceil(numpy.log2(period.max())))
bx.set_yticks(numpy.log2(Yticks))
bx.set_yticklabels(Yticks)
bx.invert_yaxis()

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = pylab.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, numpy.log2(period), 'k--')
cx.plot(fft_power, numpy.log2(1./fftfreqs), '-', color=[0.7, 0.7, 0.7],
        linewidth=1.)
cx.plot(glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
if units != '':
  cx.set_xlabel(r'Power [$%s^2$]' % (units, ))
else:
  cx.set_xlabel(r'Power')
#cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim(numpy.log2([period.min(), period.max()]))
cx.set_yticks(numpy.log2(Yticks))
cx.set_yticklabels(Yticks)
pylab.setp(cx.get_yticklabels(), visible=False)
cx.invert_yaxis()

# Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
# avg1 and avg2 parameters
dx = pylab.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(time, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) $%d$-$%d$ year scale-averaged power' % (avg1, avg2))
dx.set_xlabel('Time (year)')
if units != '':
  dx.set_ylabel(r'Average variance [$%s$]' % (units, ))
else:
  dx.set_ylabel(r'Average variance')
#
ax.set_xlim([time.min(), time.max()])