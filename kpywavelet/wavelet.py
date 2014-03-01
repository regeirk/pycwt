"""
Continuous wavelet transform module for Python. Includes a collection
of routines for wavelet transform and statistical analysis via FFT
algorithm. This module references to the numpy, scipy and pylab Python
packages.

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
    3 (2013-03-06 19:38 -0300)
    2 (2011-04-28 17:57 -0300)
    1 (2010-12-24 21:59 -0300)

REFERENCES
    [1] Mallat, S. (2008). A wavelet tour of signal processing: The
        sparse way. Academic Press, 2008, 805.
    [2] Addison, P. S. (2002). The illustrated wavelet transform
        handbook: introductory theory and applications in science,
        engineering, medicine and finance. IOP Publishing.
    [3] Torrence, C. and Compo, G. P. (1998). A Practical Guide to
        Wavelet Analysis. Bulletin of the American Meteorological
        Society, American Meteorological Society, 1998, 79, 61-78.
    [4] Torrence, C. and Webster, P. J. (1999). Interdecadal changes in
        the ENSO-Monsoon system, Journal of Climate, 12(8), 2679-2690.
    [5] Grinsted, A.; Moore, J. C. & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to
        geophysical time series. Nonlinear Processes in Geophysics, 11,
        561-566.
    [6] Liu, Y.; Liang, X. S. and Weisberg, R. H. (2007). Rectification
        of the bias in the wavelet power spectrum. Journal of
        Atmospheric and Oceanic Technology, 24(12), 2093-2102.

"""
from __future__ import division, absolute_import

from os import makedirs
from os.path import expanduser
from sys import stdout
from time import time

import numpy as np
import scipy.fftpack as fft
from scipy.stats import chi2

from .helpers import find, ar1, ar1_spectrum, rednoise
from .mothers import Morlet, Paul, DOG, MexicanHat

mothers = {'morlet': Morlet,
           'paul': Paul,
           'dog': DOG,
           'mexicanhat': MexicanHat
           }

def cwt(signal, dt=1, dj=1/12, s0=-1, J=-1, wavelet='morlet'):
    """
    Continuous wavelet transform of the signal at specified scales.

    Parameters
    ----------
    signal : numpy.ndarray, list
    Input signal array
    dt : float
    Sample spacing.
    dj : float, optional
    Spacing between discrete scales. Default value is 0.25.
    Smaller values will result in better scale resolution, but
    slower calculation and plot.
    s0 : float, optional
    Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
    Number of scales less one. Scales range from s0 up to
    s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    Default is J = (log2(N*dt/so))/dj.
    wavelet : instance of a wavelet class, or string
    Mother wavelet class. Default is Morlet wavelet.

    Returns
    -------
    W : numpy.ndarray
    Wavelet transform according to the selected mother wavelet.
    Has (J+1) x N dimensions.
    sj : numpy.ndarray
    Vector of scale indices given by sj = s0 * 2**(j * dj),
    j={0, 1, ..., J}.
    freqs : array like
    Vector of Fourier frequencies (in 1 / time units) that
    corresponds to the wavelet scales.
    coi : numpy.ndarray
    Returns the cone of influence, which is a vector of N
    points containing the maximum Fourier period of useful
    information at that particular time. Periods greater than
    those are subject to edge effects.
    fft : numpy.ndarray
    Normalized fast Fourier transform of the input signal.
    fft_freqs : numpy.ndarray
    Fourier frequencies (in 1/time units) for the calculated
    FFT spectrum.

    Example
    -------
    mother = wavelet.Morlet(6.)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
    0.25, 0.25, 0.5, 28, mother)

    """
    if isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    n0 = len(signal)                              # Original signal length.
    if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = np.int(np.round(np.log2(n0 * dt / s0) / dj))  # Number of scales
    N = 2 ** (np.int(np.round(np.log2(n0)) + 1))                  # Next higher power of 2.
    signal_ft = fft.fft(signal, N)                    # Signal Fourier transform
    ftfreqs = 2 * np.pi * fft.fftfreq(N, dt)             # Fourier angular frequencies

    sj = s0 * 2 ** (np.arange(0, J+1) * dj)         # The scales
    freqs = 1 / (wavelet.flambda() * sj)         # As of Mallat 1999

    # Creates an empty wavlet transform matrix and fills it for every discrete
    # scale using the convolution theorem.
    W = np.zeros((len(sj), N), 'complex')
    for n, s in enumerate(sj):
        psi_ft_bar = ((s * ftfreqs[1] * N) ** .5 *
            np.conjugate(wavelet.psi_ft(s * ftfreqs)))
        W[n, :] = fft.ifft(signal_ft * psi_ft_bar, N)

    # Checks for NaN in transform results and removes them from the scales,
    # frequencies and wavelet transform.
    sel = np.invert(np.isnan(W).all(axis=1))
    sj = sj[sel]
    freqs = freqs[sel]
    W = W[sel, :]

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi
    #
    return (W[:, :n0], sj, freqs, coi, signal_ft[1:N/2] / N ** 0.5,
                ftfreqs[1:N/2] / (2 * np.pi))


def icwt(W, sj, dt, dj=1/12, wavelet='morlet'):
    """
    Inverse continuous wavelet transform.

    Parameters
    ----------
    W : numpy.ndarray
    Wavelet transform, the result of the cwt function.
    sj : numpy.ndarray
    Vector of scale indices as returned by the cwt function.
    dt : float
    Sample spacing.
    dj : float, optional
    Spacing between discrete scales as used in the cwt
    function. Default value is 0.25.
    wavelet : instance of wavelet class, or string
    Mother wavelet class. Default is Morlet

    Returns
    -------
    iW : numpy.ndarray
    Inverse wavelet transform.

    Example
    -------
    mother = wavelet.Morlet()
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
    0.25, 0.25, 0.5, 28, mother)
    iwave = wavelet.icwt(wave, scales, 0.25, 0.25, mother)

    """
    if isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = np.ones([a, 1]) * sj
    else:
        raise Warning, 'Input array dimensions do not match.'

    # As of Torrence and Compo (1998), eq. (11)
    iW = dj * np.sqrt(dt) / wavelet.cdelta * wavelet.psi(0) * (np.real(W) / sj).sum(axis=0)
    return iW


def significance(signal, dt, scales, sigma_test=0, alpha=None,
                 significance_level=0.95, dof=-1, wavelet='morlet'):
    """
    Significance testing for the one dimensional wavelet transform.

    Parameters
    ----------
    signal : array like, float
    Input signal array. If a float number is given, then the
    variance is assumed to have this value. If an array is
    given, then its variance is automatically computed.
    dt : float, optional
    Sample spacing. Default is 1.0.
    scales : array like
    Vector of scale indices given returned by cwt function.
    sigma_test : int, optional
    Sets the type of significance test to be performed.
    Accepted values are 0, 1 or 2. If omitted assume 0.

    If set to 0, performs a regular chi-square test, according
    to Torrence and Compo (1998) equation 18.

    If set to 1, performs a time-average test (equation 23). In
    this case, dof should be set to the number of local wavelet
    spectra that where averaged together. For the global
    wavelet spectra it would be dof=N, the number of points in
    the time-series.

    If set to 2, performs a scale-average test (equations 25 to
    28). In this case dof should be set to a two element vector
    [s1, s2], which gives the scale range that were averaged
    together. If, for example, the average between scales 2 and
    8 was taken, then dof=[2, 8].
    alpha : float, optional
    Lag-1 autocorrelation, used for the significance levels.
    Default is 0.0.
    significance_level :float, optional
    Significance level to use. Default is 0.95.
    dof : variant, optional
    Degrees of freedom for significance test to be set
    according to the type set in sigma_test.
    wavelet : instance of a wavelet class, optional
    Mother wavelet class. Default is Morlet().

    Returns
    -------
    signif : array like
    Significance levels as a function of scale.
    fft_theor (array like):
    Theoretical red-noise spectrum as a function of period.

    """
    if isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    try:
      n0 = len(signal)
    except:
      n0 = 1
    J = len(scales) - 1
    s0 = min(scales)
    dj = np.log2(scales[1] / scales[0])

    if n0 == 1:
        variance = signal
    else:
        variance = signal.std() ** 2

    if alpha == None:
        alpha, _, _ = ar1(signal)

    period = scales * wavelet.flambda()  # Fourier equivalent periods
    freq = dt / period                   # Normalized frequency
    dofmin = wavelet.dofmin              # Degrees of freedom with no smoothing
    Cdelta = wavelet.cdelta              # Reconstruction factor
    gamma_fac = wavelet.gamma            # Time-decorrelation factor
    dj0 = wavelet.deltaj0                # Scale-decorrelation factor

    # Theoretical discrete Fourier power spectrum of the noise signal following
    # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
    pk = lambda k, a, N: (1 - a ** 2) / (1 + a ** 2 - 2 * a *
        np.cos(2 * np.pi * k / N))
    fft_theor = pk(freq, alpha, n0)
    fft_theor = variance * fft_theor     # Including time-series variance
    signif = fft_theor

    try:
        if dof == -1:
            dof = dofmin
    except:
        pass

    if sigma_test == 0:  # No smoothing, dof=dofmin, TC98 sec. 4
        dof = dofmin
        # As in Torrence and Compo (1998), equation 18
        chisquare = chi2.ppf(significance_level, dof) / dof
        signif = fft_theor * chisquare
    elif sigma_test == 1:  # Time-averaged significance
        if len(dof) == 1:
            dof = np.zeros(1, J+1) + dof
        sel = find(dof < 1)
        dof[sel] = 1
        # As in Torrence and Compo (1998), equation 23:
        dof = dofmin * (1 + (dof * dt / gamma_fac / scales) ** 2 ) ** 0.5
        sel = find(dof < dofmin)
        dof[sel] = dofmin  # Minimum dof is dofmin
        for n, d in enumerate(dof):
            chisquare = chi2.ppf(significance_level, d) / d;
            signif[n] = fft_theor[n] * chisquare
    elif sigma_test == 2:  # Time-averaged significance
        if len(dof) != 2:
            raise Exception, ('DOF must be set to [s1, s2], '
                              'the range of scale-averages')
        if Cdelta == -1:
            raise Exception, ('Cdelta and dj0 not defined for %s with f0=%f' %
                             (wavelet.name, wavelet.f0))

        s1, s2 = dof
        sel = find((scales >= s1) & (scales <= s2));
        navg = sel.size
        if navg == 0:
            raise Exception, 'No valid scales between %d and %d.' % (s1, s2)

        # As in Torrence and Compo (1998), equation 25
        Savg = 1 / sum(1. / scales[sel])
        # Power-of-two mid point:
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)
        # As in Torrence and Compo (1998), equation 28
        dof = (dofmin * navg * Savg / Smid) * (
            (1 + (navg * dj / dj0) ** 2) ** 0.5)
        # As in Torrence and Compo (1998), equation 27
        fft_theor = Savg * sum(fft_theor[sel] / scales[sel])
        chisquare = chi2.ppf(significance_level, dof) / dof;
        # As in Torrence and Compo (1998), equation 26
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare
    else:
        raise Exception, 'sigma_test must be either 0, 1, or 2.'

    return [signif, fft_theor]


def xwt(x1, y1, x2, y2, significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    """
    Calculate the cross wavelet transform (XWT). The XWT finds regions in time
    frequency space where the time series show high common power. Torrence and
    Compo (1998) state that the percent point function -- PPF (inverse of the
    cumulative distribution function) of a chi-square distribution at 95%
    confidence and two degrees of freedom is Z2(95%)=3.999. However, calculating
    the PPF using chi2.ppf gives Z2(95%)=5.991. To ensure similar significance
    intervals as in Grinsted et al. (2004), one has to use confidence of 86.46%.

    Parameters
    ----------
    signal, signal2 : numpy.ndarray, list
    Input signal array to calculate cross wavelet transform.
    dt : float
    Sample spacing.
    dj : float, optional
    Spacing between discrete scales. Default value is 0.25.
    Smaller values will result in better scale resolution, but
    slower calculation and plot.
    s0 : float, optional
    Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
    Number of scales less one. Scales range from s0 up to
    s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    Default is J = (log2(N*dt/so))/dj.
    wavelet : instance of a wavelet class, optional
    Mother wavelet class. Default is Morlet wavelet.
    significance_level : float, optional
    Significance level to use. Default is 0.95.
    normalize : bool, optional
    If set to true, normalizes CWT by the standard deviation of
    the signals.

    Returns
    -------
    xwt (array like) :
    Cross wavelet transform according to the selected mother
    wavelet.
    x (array like) :
    Intersected independent variable.
    coi (array like) :
    Cone of influence, which is a vector of N points containing
    the maximum Fourier period of useful information at that
    particular time. Periods greater than those are subject to
    edge effects.
    freqs (array like) :
    Vector of Fourier equivalent frequencies (in 1 / time units)
    that correspond to the wavelet scales.
    signif (array like) :
    Significance levels as a function of scale.

    """
    if isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    # Precision error
    e = 1e-5
    # Defines some parameters like length of both time-series, time step
    # and calculates the standard deviation for normalization and statistical
    # significance tests.
    n1 = x1.size
    n2 = x2.size
    n = min(n1, n2)
    if 'dt' not in kwargs.keys():
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]
        if abs(dx1 - dx2) < e:
            kwargs['dt'] = dx1
        else:
            raise Warning, 'Time step of both series do not match.'
    if normalize:
        std1 = y1.std()
        std2 = y2.std()
    else:
        std1 = std2 = 1.

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    W1 = cwt(y1 / std1, **kwargs)
    kwargs['dt'] = W1['dt']
    kwargs['dj'] = W1['dj']
    kwargs['s0'] = W1['s0']
    kwargs['J'] = W1['J']
    kwargs['wavelet'] = W1['wavelet']
    W2 = cwt(y2 / std2, **kwargs)

    # If both time series are different, determines the intersection of both
    # to ensure same data length.
    x = np.intersect1d(x1, x2)
    idx = dict((k, i) for i, k in enumerate(x1))
    sel1 = [idx[i] for i in x]
    idx = dict((k, i) for i, k in enumerate(x2))
    sel2 = [idx[i] for i in x]
    #
    y1 = y1[sel1[0]:sel1[-1]+1]
    W1['W'] = W1['W'][:, sel1[0]:sel1[-1]+1]
    W1['coi'] = W1['coi'][sel1[0]:sel1[-1]+1]
    y2 = y2[sel2[0]:sel2[-1]+1]
    W2['W'] = W2['W'][:, sel2[0]:sel2[-1]+1]
    W2['coi'] = W2['coi'][sel2[0]:sel2[-1]+1]

    # Now the cross correlation of y1 and y2
    W12 = W1['W'] * W2['W'].conj()
    if n1 < n2:
        coi = W1['coi']
        freqs = W1['freqs']
    else:
        coi = W2['coi']
        freqs = W2['freqs']

    # And the significance tests. Note that the confidence level is calculated
    # using the percent point function (PPF) of the chi-squared cumulative
    # distribution function (CDF) instead of using Z1(95%) = 2.182 and
    # Z2(95%)=3.999 as suggested by Torrence & Compo (1998) and Grinsted et
    # al. (2004). If the CWT has been normalized, then std1 and std2 should
    # be reset to unity, otherwise the standard deviation of both series have
    # to be calculated.
    if not normalize:
        std1 = y1.std()
        std2 = y2.std()
    else:
        std1 = std2 = 1.
    a1, _, _ = ar1(y1)
    a2, _, _ = ar1(y2)
    Pk1 = ar1_spectrum(W1['freqs'] * dx1, a1)
    Pk2 = ar1_spectrum(W2['freqs'] * dx2, a2)
    dof = kwargs['wavelet'].dofmin
    PPF = chi2.ppf(significance_level, dof)
    signif = (std1 * std2 * (Pk1 * Pk2) ** 0.5 * PPF / dof)

    # The resuts:
    return W12, x, coi, freqs, signif


def wct(x1, y1, x2, y2, significance_level=0.95, normalize=True, result=None,
    **kwargs):
    """
    Calculate the wavelet coherence (WTC). The WTC finds regions in time
    frequency space where the two time seris co-vary, but do not necessarily have
    high power.

    Parameters
    ----------
    signal, signal2 : numpy.ndarray, list
    Input signal array to calculate cross wavelet transform.
    dt : float
    Sample spacing.
    dj : float, optional
    Spacing between discrete scales. Default value is 0.25.
    Smaller values will result in better scale resolution, but
    slower calculation and plot.
    s0 : float, optional
    Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
    Number of scales less one. Scales range from s0 up to
    s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    Default is J = (log2(N*dt/so))/dj.
    significance_level (float, optional) :
    Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
    If set to true, normalizes CWT by the standard deviation of
    the signals.

    Returns
    -------
    Something : TBA and TBC

    See also
    --------
    wavelet.cwt, wavelet.xwt

    """
    # Precision error
    e = 1e-5
    # Defines some parameters like length of both time-series, time step
    # and calculates the standard deviation for normalization and statistical
    # significance tests.
    n1 = x1.size
    n2 = x2.size
    n = min(n1, n2)
    if 'dt' not in kwargs.keys():
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]
        if abs(dx1 - dx2) < e:
            kwargs['dt'] = dx1
        else:
            raise Warning, 'Time step of both series do not match.'
    if normalize:
        std1 = y1.std()
        std2 = y2.std()
    else:
        std1 = std2 = 1.

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    kwargs['result'] = 'dictionary'
    W1 = cwt(y1 / std1, **kwargs)
    kwargs['dt'] = W1['dt']
    kwargs['dj'] = W1['dj']
    kwargs['s0'] = W1['s0']
    kwargs['J'] = W1['J']
    kwargs['wavelet'] = W1['wavelet']
    W2 = cwt(y2 / std2, **kwargs)
    scales1 = np.ones([1, n1]) * W1['sj'][:, None]
    scales2 = np.ones([1, n2]) * W1['sj'][:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = kwargs['wavelet'].smooth(abs(W1['W']) ** 2 / scales1, dx1, W1['dj'],
        W1['sj'])
    S2 = kwargs['wavelet'].smooth(abs(W2['W']) ** 2 / scales2, dx2, W2['dj'],
        W1['sj'])

    # If both time series are different, determines the intersection of both
    # to ensure same data length.
    x = np.intersect1d(x1, x2)
    idx = dict((k, i) for i, k in enumerate(x1))
    sel1 = [idx[i] for i in x]
    idx = dict((k, i) for i, k in enumerate(x2))
    sel2 = [idx[i] for i in x]
    #
    y1 = y1[sel1[0]:sel1[-1]+1]
    W1['W'] = W1['W'][:, sel1[0]:sel1[-1]+1]
    W1['coi'] = W1['coi'][sel1[0]:sel1[-1]+1]
    S1 = S1[:, sel1[0]:sel1[-1]+1]
    y2 = y2[sel2[0]:sel2[-1]+1]
    W2['W'] = W2['W'][:, sel2[0]:sel2[-1]+1]
    W2['coi'] = W2['coi'][sel2[0]:sel2[-1]+1]
    S2 = S2[:, sel2[0]:sel2[-1]+1]

    # Now the wavelet transform coherence
    W12 = W1['W'] * W2['W'].conj()
    scales = np.ones([1, n]) * W1['sj'][:, None]
    S12 = kwargs['wavelet'].smooth(W12 / scales, dx1, W1['dj'], W1['sj'])
    WCT = abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)
    #
    if n1 < n2:
        coi = W1['coi']
        freqs = W1['freqs']
    else:
        coi = W2['coi']
        freqs = W2['freqs']
    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.
    a1, _, _ = ar1(y1)
    a2, _, _ = ar1(y2)
    sig = wct_significance(a1, a2, significance_level=0.95, **kwargs)

    if result == 'dictionary':
        result = dict(
            WCT = WCT,
            angle = aWCT,
            coi = coi,
            freqs = freqs,
            signif = sig[0],
            t = x,
            y1 = y1,
            y2 = y2
        )
        return result
    elif result == 'full' :
        return WCT, x, coi, freqs, sig[0], aWCT, y1, y2
    else:
        return WCT, x, coi, freqs, sig[0], aWCT


def wct_significance(a1, a2, significance_level=0.95, mc_count=300,
    verbose=False, **kwargs):
    """
    Calculates wavelet coherence significance using Monte Carlo
    simulations with 95% confidence.

    TODO: Make it work

    Parameters
    ----------
    a1, a2 (float) :
    Lag-1 autoregressive coeficients of both time series.
    significance_level (float, optional) :
    Significance level to use. Default is 0.95.
    count (integer, optional) :
    Number of Monte Carlo simulations. Default is 300.

    Returns
    -------
    """
    # Load cache if previously calculated. It is assumed that wavelet analysis
    # is performed using the wavelet's default parameters.
    aa = np.round(np.arctanh(np.array([a1, a2]) * 4))
    aa = np.abs(aa) + 0.5 * (aa < 0)
    cache = 'cache_%0.5f_%0.5f_%0.5f_%0.5f_%d_%s' % (aa[0], aa[1], kwargs['dj'],
        kwargs['s0']/kwargs['dt'], kwargs['J'], kwargs['wavelet'].name)
    cached = '%s/.klib/wavelet' % (expanduser("~"))
    try:
        dat = np.loadtxt('%s/%s.gz' % (cached, cache), unpack=True)
        stdout.write ("\n\n NOTE: Loading from cache\n\n")
        return dat[:, 0], dat[:, 1]
    except:
        pass
    # Some output to the screen
    if not verbose:
        vS = 'Calculating wavelet coherence significance'
        vs = '%s...' % (vS)
        stdout.write(vs)
        stdout.flush()
    # Choose N so that largest scale has at least some part outside the COI
    ms = kwargs['s0'] * (2 ** (kwargs['J'] * kwargs['dj'])) / kwargs['dt']
    N = np.ceil(ms * 6)
    noise1 = rednoise(N, a1, 1)
    nW1 = cwt(noise1, **kwargs)
    #
    period = np.ones([1, N]) / nW1['freqs'][:, None]
    coi = np.ones([kwargs['J']+1, 1]) * nW1['coi'][None, :]
    outsidecoi = (period <= coi)
    scales = np.ones([1, N]) * nW1['sj'][:, None]
    #
    sig95 = np.zeros(kwargs['J'] + 1)
    maxscale = find(outsidecoi.any(axis=1))[-1]
    sig95[outsidecoi.any(axis=1)] = np.nan
    #
    nbins = 1000
    wlc = np.ma.zeros([kwargs['J']+1, nbins])
    t1 = time()
    for i in range(mc_count):
        t2 = time()
        # Generates two red-noise signals with lag-1 autoregressive
        # coefficients given by a1 and a2
        noise1 = rednoise(N, a1, 1)
        noise2 = rednoise(N, a2, 1)
        # Calculate the cross wavelet transform of both red-noise signals
        nW1 = cwt(noise1, **kwargs)
        nW2 = cwt(noise2, **kwargs)
        nW12 = nW1['W'] * nW2['W'].conj()
        # Smooth wavelet wavelet transforms and calculate wavelet coherence
        # between both signals.
        S1 = kwargs['wavelet'].smooth(abs(nW1['W']) ** 2 / scales,
            kwargs['dt'], nW1['dj'], nW1['sj'])
        S2 = kwargs['wavelet'].smooth(abs(nW2['W']) ** 2 / scales,
            kwargs['dt'], nW2['dj'], nW2['sj'])
        S12 = kwargs['wavelet'].smooth(nW12 / scales, kwargs['dt'], nW1['dj'],
            nW1['sj'])
        R2 = np.ma.array(abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
        # Walks through each scale outside the cone of influence and builds a
        # coherence coefficient counter.
        for s in range(maxscale):
            cd = np.floor(R2[s, :] * nbins)
            for j, t in enumerate(cd[~cd.mask]):
                wlc[s, t] += 1
        # Outputs some text to screen if desired
        if not verbose:
            stdout.write(len(vs) * '\b')
            vs = '%s... %s ' % (vS, profiler(mc_count, i + 1, 0, t1, t2))
            stdout.write(vs)
            stdout.flush()

    # After many, many, many Monte Carlo simulations, determine the
    # significance using the coherence coefficient counter percentile.
    wlc.mask = (wlc.data == 0.)
    R2y = (np.arange(nbins) + 0.5) / nbins
    for s in range(maxscale):
        sel = ~wlc[s, :].mask
        P = wlc[s, sel].data.cumsum()
        P = (P - 0.5) / P[-1]
        sig95[s] = np.interp(significance_level, P, R2y[sel])

    # Save the results on cache to avoid to many computations in the future
    try:
        makedirs(cached)
    except:
        pass
    np.savetxt('%s/%s.gz' % (cached, cache), [sig95, nW1['sj']])

    # And returns the results
    return sig95, nW1['sj']


def profiler(N, n, t0, t1, t2):
    """Profiles the module usage.

    PARAMETERS
        N, n (int) :
            Number of total elements (N) and number of overall elements
            completed (n).
        t0, t1, t2 (float) :
            Time since the Epoch in seconds for the current module
            (t0), subroutine (t1) and step (t2).
    RETURNS
        s (string) :
            String containing the analysis result.

    EXAMPLE

    """
    n, N = float(n), float(N)
    perc = n / N * 100.
    elap0 = s2hms(time() - t0)[3]
    elap1 = s2hms(time() - t1)[3]
    elap2 = s2hms(time() - t2)[3]
    try:
        togo = s2hms(-(N - n) / n * (time()-t1))[3]
    except:
        togo = '?h??m??s'

    if t0 == 0:
        s = '%.1f%%, %s (%s, %s)' % (perc, elap1, togo, elap2)
    elif (t1 == 0) and (t2 == 0):
        s = '%.1f%%, %s' % (perc, elap0)
    else:
        s = '%.1f%%, %s (%s, %s, %s)' % (perc, elap1, togo, elap0, elap2)
    return s


def s2hms(t) :
    """Converts seconds to hour, minutes and seconds.

    PARAMETERS
        t (float) :
            Seconds value to convert

    RETURNS
        hh, mm, ss (float) :
            Calculated hour, minute and seconds
        s (string) :
            Formated output string.

    EXAMPLE
        hh, mm, ss, s = s2hms(123.45)

    """
    if t < 0:
        sign = -1
        t = -t
    else:
        sign = 1
    hh = int(t / 3600.)
    t -= hh * 3600.
    mm = int(t / 60)
    ss = t - (mm * 60.)
    dd = int(hh / 24.)
    HH = hh - dd * 24.

    if (hh > 0) | (mm > 0):
        s = '%04.1fs' % (ss)
        if hh > 0:
            s = '%dh%02dm%s' % (HH, mm, s)
            if dd > 0:
                s = '%dd%s' % (dd, s)
        else:
            s = '%dm%s' % (mm, s)
    else:
        s = '%.1fs' % (ss)
    if sign == -1:
        s = '-%s' % (s)
    #
    return (hh, mm, ss, s)
