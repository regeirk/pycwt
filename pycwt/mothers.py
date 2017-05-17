"""PyCWT mother wavelet classes."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.special import gamma
from scipy.signal import convolve2d
from scipy.special.orthogonal import hermitenorm

from .helpers import rect, fft, fft_kwargs


class Morlet(object):
    """Implements the Morlet wavelet class.

    Note that the input parameters f and f0 are angular frequencies.
    f0 should be more than 0.8 for this function to be correct, its
    default value is f0 = 6.

    """

    def __init__(self, f0=6):
        self._set_f0(f0)
        self.name = 'Morlet'

    def psi_ft(self, f):
        """Fourier transform of the approximate Morlet wavelet."""
        return (np.pi ** -0.25) * np.exp(-0.5 * (f - self.f0) ** 2)

    def psi(self, t):
        """Morlet wavelet as described in Torrence and Compo (1998)."""
        return (np.pi ** -0.25) * np.exp(1j * self.f0 * t - t ** 2 / 2)

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (4 * np.pi) / (self.f0 + np.sqrt(2 + self.f0 ** 2))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / np.sqrt(2)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / self.coi

    def _set_f0(self, f0):
        # Sets the Morlet wave number, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta},
        # \gamma, \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.f0 = f0             # Wave number
        self.dofmin = 2          # Minimum degrees of freedom
        if self.f0 == 6:
            self.cdelta = 0.776  # Reconstruction factor
            self.gamma = 2.32    # Decorrelation factor for time averaging
            self.deltaj0 = 0.60  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1

    def smooth(self, W, dt, dj, scales):
        """Smoothing function used in coherence analysis.

        Parameters
        ----------
        W :
        dt :
        dj :
        scales :

        Returns
        -------
        T :

        """
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a
        # total weight of unity, according to suggestions by Torrence &
        # Webster (1999) and by Grinsted et al. (2004).
        m, n = W.shape

        # Filter in time.
        k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
        k2 = k ** 2
        snorm = scales / dt
        # Smoothing by Gaussian window (absolute value of wavelet function)
        # using the convolution theorem: multiplication by Gaussian curve in
        # Fourier domain for each scale, outer product of scale and frequency
        F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
        smooth = fft.ifft(F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
                          axis=1,  # Along Fourier frequencies
                          **fft_kwargs(W[0, :], overwrite_x=True))
        T = smooth[:, :n]  # Remove possibly padded region due to FFT

        if np.isreal(W).all():
            T = T.real

        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = self.deltaj0 / dj * 2
        win = rect(np.int(np.round(wsize)), normalize=True)
        T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

        return T


class Paul(object):
    """Implements the Paul wavelet class.

    Note that the input parameter f is the angular frequency and that
    the default order for this wavelet is m=4.

    """
    def __init__(self, m=4):
        self._set_m(m)
        self.name = 'Paul'

    def psi_ft(self, f):
        """Fourier transform of the Paul wavelet."""
        return (2 ** self.m /
                np.sqrt(self.m * np.prod(range(2, 2 * self.m))) *
                f ** self.m * np.exp(-f) * (f > 0))

    def psi(self, t):
        """Paul wavelet as described in Torrence and Compo (1998)."""
        return (2 ** self.m * 1j ** self.m * np.prod(range(2, self.m - 1)) /
                np.sqrt(np.pi * np.prod(range(2, 2 * self.m + 1))) *
                (1 - 1j * t) ** (-(self.m + 1)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return 4 * np.pi / (2 * self.m + 1)

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return np.sqrt(2)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1 / self.coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta},
        # \gamma, \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.m = m               # Wavelet order
        self.dofmin = 2          # Minimum degrees of freedom
        if self.m == 4:
            self.cdelta = 1.132  # Reconstruction factor
            self.gamma = 1.17    # Decorrelation factor for time averaging
            self.deltaj0 = 1.50  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1


class DOG(object):
    """Implements the derivative of a Guassian wavelet class.

    Note that the input parameter f is the angular frequency and that
    for m=2 the DOG becomes the Mexican hat wavelet, which is then
    default.

    """
    def __init__(self, m=2):
        self._set_m(m)
        self.name = 'DOG'

    def psi_ft(self, f):
        """Fourier transform of the DOG wavelet."""
        return (- 1j ** self.m / np.sqrt(gamma(self.m + 0.5)) * f ** self.m *
                np.exp(- 0.5 * f ** 2))

    def psi(self, t):
        """DOG wavelet as described in Torrence and Compo (1998).

        The derivative of a Gaussian of order `n` can be determined using
        the probabilistic Hermite polynomials. They are explicitly
        written as:
            Hn(x) = 2 ** (-n / s) * n! * sum ((-1) ** m) *
                    (2 ** 0.5 * x) ** (n - 2 * m) / (m! * (n - 2*m)!)
        or in the recursive form:
            Hn(x) = x * Hn(x) - nHn-1(x)

        Source: http://www.ask.com/wiki/Hermite_polynomials

        """
        p = hermitenorm(self.m)
        return ((-1) ** (self.m + 1) * np.polyval(p, t) *
                np.exp(-t ** 2 / 2) / np.sqrt(gamma(self.m + 0.5)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (2 * np.pi / np.sqrt(self.m + 0.5))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1 / np.sqrt(2)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1 / self.coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta},
        # \gamma, \delta j_0 (Torrence and Compo, 1998, Table 2).
        self.m = m               # m-derivative
        self.dofmin = 1          # Minimum degrees of freedom
        if self.m == 2:
            self.cdelta = 3.541  # Reconstruction factor
            self.gamma = 1.43    # Decorrelation factor for time averaging
            self.deltaj0 = 1.40  # Factor for scale averaging
        elif self.m == 6:
            self.cdelta = 1.966
            self.gamma = 1.37
            self.deltaj0 = 0.97
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1


class MexicanHat(DOG):
    """Implements the Mexican hat wavelet class.

    This class inherits the DOG class using m=2.

    """
    def __init__(self):
        self.name = 'Mexican Hat'
        self._set_m(2)
