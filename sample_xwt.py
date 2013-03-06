# -*- coding: iso-8859-1 -*-
"""
Python sample script for wavelet analysis, cross wavelet transform (XWT)
and wavelet coherence (WTC) using the wavelet module. To run this 
script successfully, the matplotlib module has to be installed.

DISCLAIMER
    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/, on
    routines provided by Aslak Grinsted, John Moore and Svetlana 
    Jevrejeva and available at 
    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
    on routines provided by A. Brazhe available at 
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com

REVISION
    1 (2013-02-15 15:21)

REFERENCES
    [1] Torrence, C. and Compo, G. P. (1998). A Practical Guide to 
        Wavelet Analysis. Bulletin of the American Meteorological 
        Society, American Meteorological Society, 1998, 79, 61-78.
    [2] Grinsted, A.; Moore, J. C. & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to 
        geophysical time series. Nonlinear Processes in Geophysics, 
        2004, 11, 561-566.

"""

__version__ = '$Revision: 1 $'
# $Source$

import numpy
import pylab
try:
    reload(wavelet)
    reload(wavelet.wav)
    reload(wavelet.plot)
except:
    import wavelet

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
mother = wavelet.Morlet()
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
xwt = wavelet.xwt(t1, s1, t2, s2, significance_level=0.8646, normalize=True)

# Calculate the wavelet coherence (WTC). The WTC finds regions in time 
# frequency space where the two time seris co-vary, but do not necessarily have
# high power.
wct = wavelet.wct(t1, s1, t2, s2, significance_level=0.8646, normalize=True)

# Do the plotting!
pylab.close('all')

fig = wavelet.plot.figure(ap=dict(left=0.07, bottom=0.06, right=0.95, 
    top=0.95, wspace=0.05, hspace=0.10))
ax = fig.add_subplot(2, 1, 1)
fig, ax = wavelet.plot.cwt(t1, s1, cwt1, sig1, fig=fig, ax=ax, extend='both')
bx = fig.add_subplot(2, 1, 2, sharex=ax)
fig, bx = wavelet.plot.cwt(t2, s2, cwt2, sig2, fig=fig, ax=bx, extend='both')
ax.set_xlim = ([t2.min(), t1.max()])
if save:
    fig.savefig('sample_ao-bmi_cwt.png')

fig = wavelet.plot.figure(fp=dict())
ax = fig.add_subplot(1, 1, 1)
fig, ax = wavelet.plot.xwt(*xwt, fig=fig, ax=ax, extend='both')
ax.set_xlim = ([xwt[1].min(), xwt[1].max()])
if save:
    fig.savefig('sample_ao-bmi_xwt.png')
    

fig = wavelet.plot.figure(fp=dict())
ax = fig.add_subplot(1, 1, 1)
fig, ax = wavelet.plot.xwt(*wct, fig=fig, ax=ax, extend='neither',
    crange=numpy.arange(0, 1.1, 0.1), scale='linear', angle=wct[5])
ax.set_xlim = ([wct[1].min(), wct[1].max()])
if save:
    fig.savefig('sample_ao-bmi_wct.png')

# That's all folks!
