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

REVISION
    2 (2011-01-11 15:21)
    1 (2010-12-25 20:09)

REFERENCES
    [1] Torrence, Christopher and Compo, Gilbert P. (1998). A Practical
        Guide to Wavelet Analysis

"""


## -*- coding: iso-8859-1 -*-
#"""
#Continuous wavelet transform plot module for Python.
#
#DISCLAIMER
#    This module is based on routines provided by C. Torrence and G.
#    Compo available at http://paos.colorado.edu/research/wavelets/, on
#    routines provided by Aslak Grinsted, John Moore and Svetlana
#    Jevrejeva and available at
#    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
#    on routines provided by A. Brazhe available at
#    http://cell.biophys.msu.ru/static/swan/.
#
#    This software may be used, copied, or redistributed as long as it
#    is not sold and this copyright notice is reproduced on each copy
#    made. This routine is provided as is without any express or implied
#    warranties whatsoever.
#
#AUTHOR
#    Sebastian Krieger
#    email: sebastian@nublia.com
#
#REVISION
#    1 (2013-02-15 17:51 -0300)
#
#REFERENCES
#    [1] Mallat, S. (2008). A wavelet tour of signal processing: The
#        sparse way. Academic Press, 2008, 805.
#    [2] Addison, P. S. (2002). The illustrated wavelet transform
#        handbook: introductory theory and applications in science,
#        engineering, medicine and finance. IOP Publishing.
#    [3] Torrence, C. and Compo, G. P. (1998). A Practical Guide to
#        Wavelet Analysis. Bulletin of the American Meteorological
#        Society, American Meteorological Society, 1998, 79, 61-78.
#    [4] Torrence, C. and Webster, P. J. (1999). Interdecadal changes in
#        the ENSO-Monsoon system, Journal of Climate, 12(8), 2679-2690.
#    [5] Grinsted, A.; Moore, J. C. & Jevrejeva, S. (2004). Application
#        of the cross wavelet transform and wavelet coherence to
#        geophysical time series. Nonlinear Processes in Geophysics, 11,
#        561-566.
#    [6] Liu, Y.; Liang, X. S. and Weisberg, R. H. (2007). Rectification
#        of the bias in the wavelet power spectrum. Journal of
#        Atmospheric and Oceanic Technology, 24(12), 2093-2102.
#
#"""
#
#__version__ = '$Revision: 1 $'
## $Source$
#
from __future__ import division, absolute_import
import numpy
import pylab
from matplotlib import pyplot
import kpywavelet as wavelet

class wavplot(object):
    def __init__(self,show=False):
        fontsize = 'medium'
        params = {'font.family': 'serif',
                  'font.sans-serif': ['Helvetica'],
                  'font.size': 18,
                  'font.stretch': 'ultra-condensed',
                  'text.fontsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'ytick.labelsize': fontsize,
                  'axes.titlesize': fontsize,
                  'text.usetex': True,
                  'text.latex.unicode': True,
                  'timezone': 'UTC'
                 }
        pyplot.rcParams.update(params)
        pyplot.ion()


    def figure(self,fp=dict(), ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95,
        wspace=0.10, hspace=0.10), orientation='landscape'):
        """Creates a standard figure.

        PARAMETERS
            fp (dictionary, optional) :
                Figure properties.
            ap (dictionary, optional) :
                Adjustment properties.

        RETURNS
            fig : Figure object

        """

        self.__init__()

        if 'figsize' not in fp.keys():
            if orientation == 'landscape':
                fp['figsize'] = [11, 8]
            elif orientation == 'portrait':
                fp['figsize'] = [8, 11]
            elif orientation == 'squared':
                fp['figsize'] = [8, 8]
            elif orientation == 'worldmap':
                fp['figsize'] = [9, 5.0625] # Widescreen aspect ratio 16:9
            else:
                raise Warning, 'Orientation \'%s\' not allowed.' % (orientation, )

        fig = pyplot.figure(**fp)
        fig.subplots_adjust(**ap)

        return fig


    def cwt(self, t, f, cwt, sig, rectify=False, **kwargs):
        """Plots the wavelet power spectrum.

        It rectifies the bias in the wavelet power spectrum as noted by
        Liu et al. (2007) dividing the power by the wavelet scales.

        PARAMETERS
            t (array like) :
                Time array.
            f (array like) :
                Function array.
            cwt (list) :
                List containing the results from wavelet.cwt function (e.g.
                wave, scales, freqs, coi, fft, fftfreqs)
            sig (list) :
                List containig the results from wavelet.significance funciton
                (e.g. signif, fft_theor)
            rectify (boolean, optional) :
                Sets wether to rectify the wavelet power by dividing by the
                wavelet scales, according to Liu et al. (2007).

        RETURNS
            A list with the figure and axis objects for the plot.

        """
        # Sets some parameters and renames some of the input variables.
        N = len(t)
        dt = numpy.diff(t)[0]
        wave, scales, freqs, coi, fft, fftfreqs = cwt
        signif, fft_theor = sig
        #
        if 'std' in kwargs.keys():
            std = kwargs['std']
        else:
            std = f.std() # calculates standard deviation ...
        std2 = std ** 2   # ... and variance
        #
        period = 1. / freqs
        power = (abs(wave)) ** 2 # normalized wavelet power spectrum
        fft_power = std2 * abs(fft) ** 2 # FFT power spectrum
        sig95 = numpy.ones([1, N]) * signif[:, None]
        sig95 = power / sig95 # power is significant where ratio > 1
        if rectify:
            scales = numpy.ones([1, N]) * scales[:, None]
            levels = numpy.arange(0, 2.1, 0.1)
            labels = levels
        else:
            levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']

        result = []

        if 'fig' in kwargs.keys():
            fig = kwargs['fig']
        else:
            fig = pylab.figure()
        result.append(fig)

        # Plots the normalized wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.
        if 'fig' in kwargs.keys():
            ax = kwargs['ax']
        else:
            ax = fig.add_subplot(1,1,1)
        cmin, cmax = power.min(), power.max()
        rmin, rmax = min(levels), max(levels)
        if 'extend' in kwargs.keys():
            extend = kwargs['extend']
        elif (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        if rectify:
            cf = ax.contourf(t, numpy.log2(period), power / scales, levels,
                extend=extend)
        else:
            cf = ax.contourf(t, numpy.log2(period), numpy.log2(power),
            numpy.log2(levels), extend=extend)
        ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k',
            linewidths=2.)
        ax.fill(numpy.concatenate([t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt,
            t[:1]-dt]), numpy.log2(numpy.concatenate([coi, [1e-9],
            period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
        Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
            numpy.ceil(numpy.log2(period.max())))
        ax.set_yticks(numpy.log2(Yticks))
        ax.set_yticklabels(Yticks)
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
        ax.invert_yaxis()
        if rectify:
            cbar = fig.colorbar(cf)
        if not rectify:
            cbar = fig.colorbar(cf, ticks=numpy.log2(levels))
            cbar.ax.set_yticklabels(labels)
        result.append(ax)

        return result

    def xwt(self, *args, **kwargs):
        """Plots the cross wavelet power spectrum and phase arrows.
        function.

        The relative phase relationship convention is the same as adopted
        by Torrence and Webster (1999), where in phase signals point
        upwards (N), anti-phase signals point downwards (S). If X leads Y,
        arrows point to the right (E) and if X lags Y, arrow points to the
        left (W).

        PARAMETERS
            xwt (array like) :
                Cross wavelet transform.
            coi (array like) :
                Cone of influence, which is a vector of N points containing
                the maximum Fourier period of useful information at that
                particular time. Periods greater than those are subject to
                edge effects.
            freqs (array like) :
                Vector of Fourier equivalent frequencies (in 1 / time units)
                that correspond to the wavelet scales.
            signif (array like) :
                Significance levels as a function of Fourier equivalent
                frequencies.
            da (list, optional) :
                Pair of integers that the define frequency of arrows in
                frequency and time, default is da = [3, 3].

        RETURNS
            A list with the figure and axis objects for the plot.

        SEE ALSO
            wavelet.xwt

        """
        # Sets some parameters and renames some of the input variables.
        xwt, t, coi, freqs, signif = args[:5]
        if 'scale' in kwargs.keys():
            scale = kwargs['scale']
        else:
            scale = 'log2'

        N = len(t)
        dt = t[1] - t[0]
        period = 1. / freqs
        power = abs(xwt)
        sig95 = numpy.ones([1, N]) * signif[:, None]
        sig95 = power / sig95 # power is significant where ratio > 1

        # Calculates the phase between both time series. The phase arrows in the
        # cross wavelet power spectrum rotate clockwise with 'north' origin.
        if 'angle' in kwargs.keys():
            angle = 0.5 * numpy.pi - kwargs['angle']
        else:
            angle = 0.5 * numpy.pi - numpy.angle(xwt)
        u, v = numpy.cos(angle), numpy.sin(angle)

        result = []

        if 'da' in kwargs.keys():
            da = kwargs['da']
        else:
            da = [3, 3]
        if 'fig' in kwargs.keys():
            fig = kwargs['fig']
        else:
            fig = pylab.figure()
        result.append(fig)

        if 'fig' in kwargs.keys():
            ax = kwargs['ax']
        else:
            ax = fig.add_subplot(1, 1, 1)

        # Plots the cross wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.
        if 'crange' in kwargs.keys():
            levels = labels = kwargs['crange']
        else:
            levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']
        cmin, cmax = power.min(), power.max()
        rmin, rmax = min(levels), max(levels)
        if 'extend' in kwargs.keys():
            extend = kwargs['extend']
        elif (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        if scale == 'log2':
            Power = numpy.log2(power)
            Levels = numpy.log2(levels)
        else:
            Power = power
            Levels = levels

        cf = ax.contourf(t, numpy.log2(period), Power, Levels, extend=extend)
        ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k',
            linewidths=2.)
        q = ax.quiver(t[::da[1]], numpy.log2(period)[::da[0]], u[::da[0], ::da[1]],
            v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid',
            linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
            headaxislength=5, minshaft=2, minlength=5)
        ax.fill(numpy.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt,
            t[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi, [1e-9],
            period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
        Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
            numpy.ceil(numpy.log2(period.max())))
        ax.set_yticks(numpy.log2(Yticks))
        ax.set_yticklabels(Yticks)
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
        ax.invert_yaxis()
        cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
        cbar.ax.set_yticklabels(labels)
        result.append(ax)

        return result

# This script allows different sample data sets to be analysed. Simply comment
# and uncomment the respective fname, title, label, t0, dt and units variables
# to see the different results. t0 is the starting time, dt is the temporal
# sampling step
#
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

var = numpy.loadtxt(fname)
avg1, avg2 = (2, 8)                  # Range of periods to average
slevel = 0.95                        # Significance level

std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (var - var.mean()) / std       # Calculating anomaly and normalizing

N = var.size                         # Number of measurements
time = numpy.arange(0, N) * dt + t0  # Time array in years

dj = 0.25                            # Four sub-octaves per octaves
s0 = -1 #2 * dt                      # Starting scale, here 6 months
J = -1  #7 / dj                      # Seven powers of two with dj sub-octaves
alpha = 0.0                          # Lag-1 autocorrelation for white noise
#alpha, _, _ = wavelet.ar1(var)

mother = wavelet.Morlet(6.)          # Morlet mother wavelet with wavenumber=6
#mother = wavelet.Mexican_hat()       # Mexican hat wavelet, or DOG with m=2
#mother = wavelet.Paul(4)             # Paul wavelet with order m=4
#mother = wavelet.DOG(6)              # Derivative of the Gaussian, with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)
power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = numpy.ones([1, N]) * signif[:, None]
sig95 = power / sig95                # Where ratio > 1, power is significant

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)
dof = N - scales                     # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                       significance_level=slevel, dof=dof, wavelet=mother)

# Scale average between avg1 and avg2 periods and significance level
sel = pylab.find((period >= avg1) & (period < avg2))
Cdelta = mother.cdelta
scale_avg = (scales * numpy.ones((N, 1))).transpose()
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
pylab.close('all')
pylab.ion()
fontsize = 'medium'
params = {'text.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'text.usetex': True
         }
pylab.rcParams.update(params)          # Plot parameters
figprops = dict(figsize=(11, 8), dpi=72)
fig = pylab.figure(**figprops)

# First sub-plot, the original time series anomaly.
ax = pylab.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
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
bx.fill(numpy.concatenate([time, time[-1:]+dt, time[-1:]+dt,time[:1]-dt, time[:1]-dt]),
        numpy.log2(numpy.concatenate([coi,[1e-9], period[-1:], period[-1:], [1e-9]])),
        'k', alpha=0.3,hatch='x')

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
#
pylab.draw()
pylab.show()

###
###
"""
"""
# Important parameters
data1 = dict(
    name = 'Arctic Oscillation',
    nick = 'AO',
    file = 'jao.dat',
    alpha = 0.0
)
data2 = dict(
    name = 'Baltic Sea ice extent',
    nick = 'BMI',
    file='jbaltic.dat',
    alpha = 0.0
)
mother = 'morlet'
save = False

# Functions used in the module
def boxpdf(x):
    """
    Forces the probability density function of the input data to have
    a boxed distribution.

    PARAMETERS
        x (array like) :
            Input data

    RETURNS
        X (array like) :
            Boxed data varying between zero and one.
        Bx, By (array like) :
            Data lookup table

    """
    x = numpy.asarray(x)
    n = x.size

    # Kind of 'unique'
    i = numpy.argsort(x)
    d = (numpy.diff(x[i]) != 0)
    I = pylab.find(numpy.concatenate([d, [True]]))
    X = x[i][I]

    I = numpy.concatenate([[0], I+1])
    Y = 0.5 * (I[0:-1] + I[1:]) / n
    bX = numpy.interp(x, X, Y)

    return bX, X, Y


# Loads the data to be analysed.
t1, s1 = numpy.loadtxt(data1['file'], unpack=True)
t2, s2 = numpy.loadtxt(data2['file'], unpack=True)
dt = numpy.diff(t1)[0]
n1 = t1.size
n2 = t2.size
n = min(n1, n2)

# Change the probablity density function (PDF) of the data. The time series
# of Baltic Sea ice extent is highly bi-modal and we therefore transform the
# timeseries into a series of percentiles. The transformed series probably
# reacts 'more linearly' to climate.
s2, _, _ = boxpdf(s2)

# Calculates the standard deviatio of each time series for later
# normalization.
std1 = s1.std()
std2 = s2.std()

# Calculate the CWT of both normalized time series. The function wavelet.cwt
# returns a a list with containing [wave, scales, freqs, coi, fft, fftfreqs]
# variables.
cwt1 = wavelet.cwt(s1 / std1, dt, wavelet=mother)
sig1 = wavelet.significance(1.0, dt, cwt1[1], 0, data1['alpha'],
    wavelet=mother)
cwt2 = wavelet.cwt(s2 / std2, dt, wavelet=mother)
sig2 = wavelet.significance(1.0, dt, cwt2[1], 0, data1['alpha'],
    wavelet=mother)

# Calculate the cross wavelet transform (XWT). The XWT finds regions in time
# frequency space where the time series show high common power. Torrence and
# Compo (1998) state that the percent point function -- PPF (inverse of the
# cumulative distribution function) of a chi-square distribution at 95%
# confidence and two degrees of freedom is Z2(95%)=3.999. However, calculating
# the PPF using chi2.ppf gives Z2(95%)=5.991. To ensure similar significance
# intervals as in Grinsted et al. (2004), one has to use confidence of 86.46%.
xwt = wavelet.xwt(t1, s1, t2, s2, wavelet=mother, significance_level=0.8646, normalize=True)

# Calculate the wavelet coherence (WTC). The WTC finds regions in time
# frequency space where the two time seris co-vary, but do not necessarily have
# high power.
wct = wavelet.wct(t1, s1, t2, s2, wavelet=mother, significance_level=0.8646, normalize=True)
# Do the plotting!
pylab.close('all')

wavplot = wavplot()
fig = wavplot.figure(ap=dict(left=0.07, bottom=0.06, right=0.95,
    top=0.95, wspace=0.05, hspace=0.10))
ax = fig.add_subplot(2, 1, 1)
fig, ax = wavplot.cwt(t1, s1, cwt1, sig1, fig=fig, ax=ax, extend='both')
bx = fig.add_subplot(2, 1, 2, sharex=ax)
fig, bx = wavplot.cwt(t2, s2, cwt2, sig2, fig=fig, ax=bx, extend='both')
ax.set_xlim = ([t2.min(), t1.max()])
if save:
    fig.savefig('sample_ao-bmi_cwt.png')

fig = wavplot.figure(fp=dict())
ax = fig.add_subplot(1, 1, 1)
fig, ax = wavplot.xwt(*xwt, fig=fig, ax=ax, extend='both')
ax.set_xlim = ([xwt[1].min(), xwt[1].max()])
if save:
    fig.savefig('sample_ao-bmi_xwt.png')


fig = wavplot.figure(fp=dict())
ax = fig.add_subplot(1, 1, 1)
fig, ax = wavplot.xwt(*wct, fig=fig, ax=ax, extend='neither',
    crange=numpy.arange(0, 1.1, 0.1), scale='linear', angle=wct[5])
ax.set_xlim = ([wct[1].min(), wct[1].max()])
if save:
    fig.savefig('sample_ao-bmi_wct.png')