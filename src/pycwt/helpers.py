"""PyCWT helper functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy

# Try to import the Python wrapper for FFTW.
try:
    from multiprocessing import cpu_count

    import pyfftw.interfaces.scipy_fftpack as fft

    # Fast planning, use all available threads.
    _FFTW_KWARGS_DEFAULT = {"planner_effort": "FFTW_ESTIMATE", "threads": cpu_count()}

    def fft_kwargs(signal, **kwargs):
        """Return optimized keyword arguments for FFTW"""
        kwargs.update(_FFTW_KWARGS_DEFAULT)
        kwargs["n"] = len(signal)  # do not pad
        return kwargs


# Otherwise, fall back to 2 ** n padded scipy FFTPACK
except ImportError:
    import scipy.fftpack as fft

    # Can be turned off, e.g. for MKL optimizations
    _FFT_NEXT_POW2 = True

    def fft_kwargs(signal, **kwargs):
        """Return next higher power of 2 for given signal to speed up FFT"""
        if _FFT_NEXT_POW2:
            return {"n": int(2 ** numpy.ceil(numpy.log2(len(signal))))}


from os import makedirs
from os.path import exists, expanduser

from scipy.signal import lfilter


def find(condition):
    """Returns the indices where ravel(condition) is true."""
    (res,) = numpy.nonzero(numpy.ravel(condition))
    return res


def ar1(x):
    """
    Allen and Smith autoregressive lag-1 autocorrelation coefficient.
    In an AR(1) model

        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,

    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Parameters
    ----------
    x : numpy.ndarray, list
        Univariate time series

    Returns
    -------
    g : float
        Estimate of the lag-one autocorrelation.
    a : float
        Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
    mu2 : float
        Estimated square on the mean of a finite segment of AR(1)
        noise, mormalized by the process variance.

    References
    ----------
    [1] Allen, M. R. and Smith, L. A. Monte Carlo SSA: detecting
        irregular oscillations in the presence of colored noise.
        *Journal of Climate*, **1996**, 9(12), 3373-3404.
        <http://dx.doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2>
    [2] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html

    """
    x = numpy.asarray(x)
    N = x.size
    xm = x.mean()
    x = x - xm

    # Estimates the lag zero and one covariance
    c0 = x.transpose().dot(x) / N
    c1 = x[0 : N - 1].transpose().dot(x[1:N]) / (N - 1)

    # According to A. Grinsteds' substitutions
    B = -c1 * N - c0 * N**2 - 2 * c0 + 2 * c1 - c1 * N**2 + c0 * N
    A = c0 * N**2
    C = N * (c0 + c1 * N - c1)
    D = B**2 - 4 * A * C

    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning(
            "Cannot place an upperbound on the unbiased AR(1). "
            "Series is too short or trend is to large."
        )

    # According to Allen & Smith (1996), footnote 4
    mu2 = -1 / N + (2 / N**2) * (
        (N - g**N) / (1 - g) - g * (1 - g ** (N - 1)) / (1 - g) ** 2
    )
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2


def ar1_spectrum(freqs, ar1=0.0):
    """
    Lag-1 autoregressive theoretical power spectrum.

    Parameters
    ----------
    freqs : numpy.ndarray, list
        Frequencies at which to calculate the theoretical power
        spectrum.
    ar1 : float
        Autoregressive lag-1 correlation coefficient.

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
    freqs = numpy.asarray(freqs)
    Pk = (1 - ar1**2) / numpy.abs(
        1 - ar1 * numpy.exp(-2 * numpy.pi * 1j * freqs)
    ) ** 2

    return Pk


def rednoise(N, g, a=1.0):
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
        yr = numpy.randn(N, 1) * a
    else:
        # Twice the decorrelation time.
        tau = int(numpy.ceil(-2 / numpy.log(numpy.abs(g))))
        yr = lfilter([1, 0], [1, -g], numpy.random.randn(N + tau, 1) * a)
        yr = yr[tau:]

    return yr.flatten()


def rect(x, normalize=False):
    """TODO: describe what I do."""
    if type(x) in [int, float]:
        shape = [
            x,
        ]
    elif type(x) in [list, dict]:
        shape = x
    elif type(x) in [numpy.ndarray, numpy.ma.core.MaskedArray]:
        shape = x.shape
    X = numpy.zeros(shape)
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
        Data lookup table.

    """
    x = numpy.asarray(x)
    n = x.size

    # Kind of 'unique'
    i = numpy.argsort(x)
    d = numpy.diff(x[i]) != 0
    j = find(numpy.concatenate([d, [True]]))
    X = x[i][j]

    j = numpy.concatenate([[0], j + 1])
    Y = 0.5 * (j[0:-1] + j[1:]) / n
    bX = numpy.interp(x, X, Y)

    return bX, X, Y


def get_cache_dir():
    """Returns the location of the cache directory."""
    # Sets cache directory according to user home path.
    cache_dir = "{}/.cache/pycwt/".format(expanduser("~"))
    # Creates cache directory if not existant.
    if not exists(cache_dir):
        makedirs(cache_dir)
    # Returns cache directory.
    return cache_dir
