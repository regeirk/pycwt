"""
Python sample script for wavelet analysis, cross wavelet transform (XWT)
and wavelet coherence (WTC) using the wavelet module. To run this 
script successfully, the matplotlib module has to be installed.
"""

import numpy as np
import matplotlib.pyplot as plt

import kpywavelet as wavelet

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
    file = 'jbaltic.dat',
    alpha = 0.0
)
mother = wavelet.Morlet()
save = True

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
    x = np.asarray(x)
    n = x.size
    
    # Kind of 'unique'
    i = np.argsort(x)
    d = (np.diff(x[i]) != 0)
    I = np.where(np.concatenate([d, [True]])) # pylab.find
    X = x[i][I]
    
    I = np.concatenate([[0], I[0]+1])
    Y = 0.5 * (I[0:-1] + I[1:]) / n
    bX = np.interp(x, X, Y)
    
    return bX, X, Y


# Loads the data to be analysed.
t1, s1 = np.loadtxt(data1['file'], unpack=True)
t2, s2 = np.loadtxt(data2['file'], unpack=True)
dt = np.diff(t1)[0]
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
plt.close('all')

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
    crange=np.arange(0, 1.1, 0.1), scale='linear', angle=wct[5])
ax.set_xlim = ([wct[1].min(), wct[1].max()])
if save:
    fig.savefig('sample_ao-bmi_wct.png')

# That's all folks!
