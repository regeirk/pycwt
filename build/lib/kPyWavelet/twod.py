# -*- coding: iso-8859-1 -*-
"""
Bi-dimensional continuous wavelet transform module for Python. Includes a 
collection of routines for wavelet transform and statistical analysis via
FFT algorithm. This module references to the numpy, scipy and pylab Python
packages.

DISCLAIMER
    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com

REVISION
    1 (2011-04-30 19:48 -3000)

REFERENCES
    [1] Wang, Ning and Lu, Chungu (2010). Two-dimensional continuous
        wavelet analysis and its application to meteorological data

"""

__version__ = '$Revision: 1 $'
# $Source$

from numpy import (arange, ceil, concatenate, conjugate, cos, exp, floor, 
                   isnan, log, log2, meshgrid, ones, pi, prod, real, sqrt,
                   zeros, polyval)
from numpy.fft import fft2, ifft2, fftfreq
from pylab import find


class Mexican_hat():
    """Implements the Mexican hat wavelet class."""

    name = 'Mexican hat'
    
    def __init__(self):
        # Reconstruction factor $C_{\psi, \delta}$
        self.cpsi = 1. # pi

    def psi_ft(self, k, l):
        """
        Fourier transform of the Mexican hat wavelet as in Wang and
        Lu (2010), equation [15].
 
        """
        K, L = meshgrid(k, l)
        return (K ** 2. + L ** 2.) * exp(-0.5 * (K ** 2. + L ** 2.))

    def psi(self, x, y):
        """Mexican hat wavelet as in Wang and Lu (2010), equation [14]."""
        X, Y = meshgrid(x, y)
        return (2. - (X ** 2. + Y ** 2.)) * exp(-0.5 * (X ** 2. + Y ** 2.))


def cwt2d(f, dx, dy, a=None, wavelet=Mexican_hat()):
    """
    Bi-dimensional continuous wavelet transform of the signal at 
    specified scale a.

    PARAMETERS
        f (array like):
            Input signal array.
        dx, dy (float):
            Sample spacing for each dimension.
        a (array like, optional):
            Scale parameter array.
        wavelet (class, optional) :
            Mother wavelet class. Default is Mexican_hat()

    RETURNS

    EXAMPLE

    """
    # Determines the shape of the arrays and the discrete scales.
    n0, m0 = f.shape
    N, M = 2 ** int(ceil(log2(n0))), 2 ** int(ceil(log2(m0)))
    if a == None:
        a = 2 ** arange(int(floor(log2(min(n0, m0)))))
    A = len(a)
    # Calculates the zonal and meridional wave numbers.
    l, k = fftfreq(N, dy), fftfreq(M, dx)
    # Calculates the Fourier transform of the input signal.
    f_ft = fft2(f, s=(N, M))
    # Creates empty wavelet transform array and fills it for every discrete
    # scale using the convolution theorem.
    Wf = zeros((A, N, M), 'complex')
    for i, an in enumerate(a):
        psi_ft_bar = an * wavelet.psi_ft(an * k, an * l)
        Wf[i, :, :] = ifft2(f_ft * psi_ft_bar, s=(N, M))

    return Wf[:, :n0, :m0], a


def icwt2d(W, a, dx=0.25, dy=0.25, da=0.25, wavelet=Mexican_hat()):
    """
    Inverse bi-dimensional continuous wavelet transform as in Wang and
    Lu (2010), equation [5].

    PARAMETERS
        W (array like):
            Wavelet transform, the result of the cwt2d function.
        a (array like, optional):
            Scale parameter array.
        w (class, optional) :
            Mother wavelet class. Default is Mexican_hat()

    RETURNS
        iW (array like) :
            Inverse wavelet transform.

    EXAMPLE

    """
    m0, l0, k0 = W.shape
    if m0 != a.size:
        raise Warning, 'Scale parameter array shape does not match wavelet' \
                       ' transform array shape.'
    # Calculates the zonal and meridional wave numters.
    L, K = 2 ** int(ceil(log2(l0))), 2 ** int(ceil(log2(k0)))
    # Calculates the zonal and meridional wave numbers.
    l, k = fftfreq(L, dy), fftfreq(K, dx)
    # Creates empty inverse wavelet transform array and fills it for every 
    # discrete scale using the convolution theorem.
    iW = zeros((m0, L, K), 'complex')
    for i, an in enumerate(a):
        psi_ft_bar = an * wavelet.psi_ft(an * k, an * l)
        W_ft = fft2(W[i, :, :], s=(L, K))
        iW[i, :, :] = ifft2(W_ft * psi_ft_bar, s=(L, K)) * da / an ** 2.

    return iW[:, :l0, :k0].real.sum(axis=0) / wavelet.cpsi
