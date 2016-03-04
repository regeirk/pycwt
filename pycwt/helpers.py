from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
try:                            # pyfftw wrapper for FFTW
    import pyfftw.interfaces.scipy_fftpack as fftmod
    from multiprocessing import cpu_count
    _FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE', # fast planning
                            'threads': cpu_count(), # use all available threads
                            'n': None, # do not pad
                            }
    def fft_kwargs(signal, **kwargs):
        '''Return optimized keyword arguments for FFTW'''
        kwargs.update(_FFTW_KWARGS_DEFAULT)
        return kwargs
except ImportError:             # fall back to 2^n padded scipy FFTPACK
    import scipy.fftpack as fftmod
    _FFT_NEXT_POW2 = True       # can be turned off, e.g. for MKL optimizations
    def fft_kwargs(signal, **kwargs):
        '''Return next higher power of 2 for given signal to speed up FFT'''
        if _FFT_NEXT_POW2:
            return {'n': np.int(2**np.ceil(np.log2(len(signal))))}
from scipy.signal import lfilter


def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res


def ar1(x):
    """
    Allen and Smith autoregressive lag-1 autocorrelation alpha. In a AR(1) model

    x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,

    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Parameters
    ----------
    x : numpy.ndarray, list
    Univariate time series

    Return
    ------
    g : float
    Estimate of the lag-one autocorrelation.
    a : float
    Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
    mu2 : float
    Estimated square on the mean of a finite segment of AR(1)
    noise, mormalized by the process variance.

    References
    ----------
    [1] Allen, M. R. and Smith, L. A. (1996). Monte Carlo SSA:
    detecting irregular oscillations in the presence of colored
    noise. Journal of Climate, 9(12), 3373-3404.
    http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html
    """
    x = np.asarray(x)
    N = x.size
    xm = x.mean()
    x = x - xm

    # Estimates the lag zero and one covariance
    c0 = x.transpose().dot(x) / N
    c1 = x[0:N-1].transpose().dot(x[1:N]) / (N - 1)

    # According to A. Grinsteds' substitutions
    B = -c1 * N - c0 * N**2 - 2 * c0 + 2 * c1 - c1 * N**2 + c0 * N
    A = c0 * N**2
    C = N * (c0 + c1 * N - c1)
    D = B**2 - 4 * A * C

    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning ('Cannot place an upperbound on the unbiased AR(1). '
            'Series is too short or trend is to large.')

    # According to Allen & Smith (1996), footnote 4
    mu2 = -1 / N + (2 / N**2) * ((N - g**N) / (1 - g) -
        g * (1 - g**(N - 1)) / (1 - g)**2)
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2


def ar1_spectrum(freqs, ar1=0.) :
    """
    Lag-1 autoregressive theoretical power spectrum.


    Parameters
    ----------
    freqs : numpy.ndarray, list
    Frequencies at which to calculate the theoretical power
    spectrum.
    ar1 : float
    Lag-1 autoregressive correlation coefficient.

    Returns
    -------
    Pk : numpy.ndarray
    Theoretical discrete Fourier power spectrum of noise signal.

    References
    ----------
    [1] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html

    """
    # According to a post from the MadSci Network available at
    # http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html,
    # the time-series spectrum for an auto-regressive model can be
    # represented as
    #
    # P_k = \frac{E}{\left|1- \sum\limits_{k=1}^{K} a_k \, e^{2 i \pi
    #   \frac{k f}{f_s} } \right|^2}
    #
    # which for an AR1 model reduces to
    #
    freqs = np.asarray(freqs)
    Pk = (1 - ar1 ** 2) / np.abs(1 - ar1 * np.exp(-2 * np.pi * 1j * freqs)) ** 2

    # Theoretical discrete Fourier power spectrum of the noise signal following
    # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
    #N = len(freqs)
    #Pk = (1 - ar1 ** 2) / (1 + ar1 ** 2 - 2 * ar1 * cos(2 * pi * freqs / N))

    return Pk


def rednoise(N, g, a=1.) :
    """
    Red noise generator using filter.

    Parameters
    ----------
    N : int
    Length of the desired time series.
    g : float
    Lag-1 autocorrelation coefficient.
    a : float, optional
    Noise innovation variance parameter.

    Returns
    -------
    y : numpy.ndarray
    Red noise time series.

    """
    if g == 0:
        yr = np.randn(N, 1) * a;
    else:
        # Twice the decorrelation time.
        tau = np.ceil(-2 / np.log(np.abs(g)))
        yr = lfilter([1, 0], [1, -g], np.random.randn(N + tau, 1) * a)
        yr = yr[tau:]

    return yr.flatten()


def rect(x, normalize=False) :
    if type(x) in [int, float]:
        shape = [x, ]
    elif type(x) in [list, dict]:
        shape = x
    elif type(x) in [np.ndarray, np.ma.core.MaskedArray]:
        shape = x.shape
    X = np.zeros(shape)
    X[0] = X[-1] = 0.5
    X[1:-1] = 1

    if normalize:
        X /= X.sum()

    return X

def boxpdf(x):
    """
    Forces the probability density function of the input data to have
    a boxed distribution.

    Parameters
    ----------
    x (array like) :
    Input data

    Returns
    -------
    X (array like) :
    Boxed data varying between zero and one.
    Bx, By (array like) :
    Data lookup table

    #TODO: THE FUCK?
    """
    x = np.asarray(x)
    n = x.size

    # Kind of 'unique'
    i = np.argsort(x)
    d = (np.diff(x[i]) != 0)
    I = find(np.concatenate([d, [True]]))
    X = x[i][I]

    I = np.concatenate([[0], I+1])
    Y = 0.5 * (I[0:-1] + I[1:]) / n
    bX = np.interp(x, X, Y)

    return bX, X, Y
