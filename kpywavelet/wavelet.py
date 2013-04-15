"""
Continuous wavelet transform module for Python. Includes a collection
of routines for wavelet transform and statistical analysis via FFT
algorithm. This module references to the numpy, scipy and pylab Python
packages.
"""
from __future__ import division

import numpy as np
import numpy.fft as fft
from numpy.random import randn
from numpy.lib.polynomial import polyval
from pylab import find
from scipy.stats import chi2
from scipy.special import gamma
from scipy.signal import convolve2d, lfilter
from scipy.special.orthogonal import hermitenorm
from os import makedirs
from os.path import expanduser
from sys import stdout
import time

class Morlet:
    """
    Implements the Morlet wavelet class.

    Note that the input parameters f and f0 are angular frequencies.
    f0 should be more than 0.8 for this function to be correct, its
    default value is f0=6.

    #TODO: Implenment arbitarty order
    """

    def __init__(self, f0=6.0):
        self._set_f0(f0)

    def psi_ft(self, f):
        """Fourier transform of the approximate Morlet wavelet."""
        return (np.pi ** -.25) * np.exp(-0.5 * (f - self.f0) ** 2.)

    def psi(self, t):
        """Morlet wavelet as described in Torrence and Compo (1998)."""
        return (np.pi ** -.25) * np.exp(1j * self.f0 * t - t ** 2. / 2.)

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (4 * np.pi) / (self.f0 + np.sqrt(2 + self.f0 ** 2))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / np.sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / self.coi

    def _set_f0(self, f0):
        # Sets the Morlet wave number, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.f0 = f0             # Wave number
        self.dofmin = 2          # Minimum degrees of freedom
        if self.f0 == 6.:
            self.cdelta = 0.776  # Reconstruction factor
            self.gamma = 2.32    # Decorrelation factor for time averaging
            self.deltaj0 = 0.60  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1

    def rect(x, normalize=False) :
        """
        TODO: THE FUCK IS THIS?
        """
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
    
    def smooth(self, W, dt, dj, scales):
        """
        Smoothing function used in coherence analysis.
        
        The smoothing is performed by using a filter given by the absolute
        value of the wavelet function at each scale, normalized to have a 
        total weight of unity, according to suggestions by Torrence & 
        Webster (1999) and by Grinsted et al. (2004).

        #TODO: Implenment arbitarty order 
        """

        T = np.zeros([W.shape[0], W.shape[1]])
        
        # Filter in time. 
        npad = int(2 ** np.ceil(np.log2(W.shape[1])))
        k = 2 * np.pi * fft.fftfreq(npad)
        k2 = k ** 2
        snorm = scales / dt
        
        for i in range(W.shape[0]):
            F = np.exp(-0.5 * (snorm[i] ** 2) * k2)
            smooth = fft.ifft(F * fft.fft(W[i, :], npad))
            T[i, :] = smooth[0:W.shape[1]]
        
        if np.isreal(W).all():
            T = T.real
        
        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = self.deltaj0 / dj * 2
        win = self.rect(int(round(wsize)), normalize=True)
        T = convolve2d(T, win[:, None], 'same')
        
        return T        

class Paul:
    """
    Implements the Paul wavelet class.

    Note that the input parameter f is the angular frequency and that
    the default order for this wavelet is m=4.

    #TODO: Implenment arbitarty order
    """

    def __init__(self, m=4):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the Paul wavelet."""
        return (2 ** self.m / np.sqrt(self.m * np.prod(range(2, 2 * self.m))) *
                f ** self.m * np.exp(-f) * (f > 0))

    def psi(self, t):
        """Paul wavelet as described in Torrence and Compo (1998)."""
        return (2 ** self.m * 1j ** self.m * np.prod(range(2, self.m - 1)) /
                np.sqrt(np.pi * np.prod(range(2, 2 * self.m + 1))) * (1 - 1j * t) **
                (-(self.m + 1)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return 4 * np.pi / (2 * self.m + 1)

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return np.sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / self.coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.m = m               # Wavelet order
        self.dofmin =  2         # Minimum degrees of freedom
        if self.m == 4:
            self.cdelta = 1.132  # Reconstruction factor
            self.gamma = 1.17    # Decorrelation factor for time averaging
            self.deltaj0 = 1.50  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1


class DOG:
    """
    Implements the derivative of a Guassian wavelet class.

    Note that the input parameter f is the angular frequency and that
    for m=2 the DOG becomes the Mexican hat wavelet and that
    the default order for this wavelet is m=6.

    #TODO: Implenment arbitarty order
    """
    def __init__(self, m=6):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the DOG wavelet."""
        return (- 1j ** self.m / np.sqrt(gamma(self.m + 0.5)) * f ** self.m *
                np.exp(- 0.5 * f ** 2))

    def psi(self, t):
        """DOG wavelet as described in Torrence and Compo (1998)

        The derivative of a Gaussian of order n can be determined using
        the probabilistic Hermite polynomials. They are explicitly
        written as:
            Hn(x) = 2 ** (-n / s) * n! * sum ((-1) ** m) * (2 ** 0.5 *
                x) ** (n - 2 * m) / (m! * (n - 2*m)!)
        or in the recursive form:
            Hn(x) = x * Hn(x) - nHn-1(x)

        Source: http://www.ask.com/wiki/Hermite_polynomials

        """
        p = hermitenorm(self.m)
        return ((-1) ** (self.m + 1) * polyval(p, t) * np.exp(-t ** 2 / 2) /
                np.sqrt(gamma(self.m + 0.5)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (2 * np.pi / np.sqrt(self.m + 0.5))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / np.sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / self.coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
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


class Mexican_hat(DOG):
    """
    Implements the Mexican hat wavelet class.

    This class inherits the DOG class using m=2.

    """
    def __init__(self):
        self._set_m(2)


def fftconv(x, y):
    """ Convolution of x and y using the FFT convolution theorem. """
    N = len(x)
    n = int(2 ** np.ceil(np.log2(N))) + 1
    X, Y, x_y = fft(x, n), fft(y, n), []
    for i in range(n):
        x_y.append(X[i] * Y[i])

    # Returns the inverse Fourier transform with padding correction
    return fft.ifft(x_y)[4:N+4]


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
    x -= x.mean()
    
    # Estimates the lag zero and one covariance
    c0 = x.transpose().dot(x) / x.size
    c1 = x[0:x.size-1].transpose().dot(x[1:x.size]) / (x.size - 1)
    
    # According to A. Grinsteds' substitutions
    B = -c1 * x.size - c0 * x.size**2 - 2 * c0 + 2 * c1 - c1 * x.size**2 + c0 * x.size
    A = c0 * x.size**2
    C = x.size * (c0 + c1 *x.size - c1)
    D = B**2 - 4 * A * C
    
    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning ('Cannot place an upperbound on the unbiased AR(1). '
            'Series is too short or trend is to large.')
    
    # According to Allen & Smith (1996), footnote 4    
    mu2 = -1 / x.size + (2 / x.size**2) * ((x.size - g**x.size) / (1 - g) - 
        g * (1 - g**(x.size - 1)) / (1 - g)**2)
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2


def ar1_spectrum(freqs, ar1=0., fourier=False) :
    """
    Lag-1 autoregressive theoretical power spectrum.
    
    According to a post from the MadSci Network, the time-series spectrum for 
    an auto-regressive model can be represented as
     
    P_k = \frac{E}{\left|1- \sum\limits_{k=1}^{K} a_k \, e^{2 i \pi 
       \frac{k f}{f_s} } \right|^2}
    
    which for an AR1 model can be reduced and is used here.
    
    The theoretical discrete fourier power spectrum of the noise signal
    following Gilman et al. (1963) and Torrence and Compo (1998), equation 16
    is available.

    Parameters
    ----------
        freqs : numpy.ndarray, list
            Frequencies at which to calculate the theoretical power 
            spectrum.
        ar1 : float
            Lag-1 autoregressive correlation coefficient.
        fourier : bool
            Returns the theoretical power spectrum for FFT
    Returns
    -------
        Pk : numpy.ndarray
            Theoretical discrete Fourier power spectrum of noise signal.
    
    References
    ----------
        [1] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html
        
    """
    
    freqs = np.asarray(freqs)
    
    if fourier:
        Pk = (1 - ar1 ** 2) / (1 + ar1 ** 2 - 2 * ar1 * np.cos(2 * np.pi * freqs / len(freqs)))
    else:
        Pk = (1 - ar1 ** 2) / abs(1 - ar1 * np.exp(-2 * np.pi * 1j * freqs)) ** 2
    
    return Pk


def rednoise(N, g, a=1.):
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
        yr = randn(N, 1) * a;
    else:
        # Twice the decorrelation time.
        tau = np.ceil(-2 / np.log(abs(g)))
        yr = lfilter([1, 0], [1, -g], randn(N + tau, 1) * a)
        yr = yr[tau:]
    
    return yr.flatten()

def cwt(signal, dt, dj=1./12, s0=-1, J=-1, wavelet=Morlet()):
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
        wavelet : instance of a wavelet class, optional 
            Mother wavelet class. Default is Morlet wavelet.

    Returns
    -------
        W  : numpy.ndarray
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
    n0 = len(signal)                              # Original signal length.
    if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = int(np.log2(n0 * dt / s0) / dj)  # Number of scales
    N = 2 ** (int(np.log2(n0)) + 1)                  # Next higher power of 2.
    signal_ft = fft.fft(signal, N)                    # Signal Fourier transform
    ftfreqs = 2 * np.pi * fft.fftfreq(N, dt)             # Fourier angular frequencies

    sj = s0 * 2. ** (np.arange(0, J+1.) * dj)         # The scales
    freqs = 1. / (wavelet.flambda() * sj)         # As of Mallat 1999

    # Creates an empty wavlet transform matrix and fills it for every discrete
    # scale using the convolution theorem.
    W = np.zeros((len(sj), N), 'complex')
    for n, s in enumerate(sj):
        psi_ft_bar = ((s * ftfreqs[1] * N) ** .5 * 
            np.conjugate(wavelet.psi_ft(s * ftfreqs)))
        W[n, :] = fft.ifft(signal_ft * psi_ft_bar, N)

    # Checks for NaN in transform results and removes them from the scales,
    # frequencies and wavelet transform.
    sel = np.logical_not(np.isnan(W).all(axis=1))
    sj = sj[sel]
    freqs = freqs[sel]
    W = W[sel, :]

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2. - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi
    #
    return (W[:, :n0], sj, freqs, coi, signal_ft[1:N/2] / N ** 0.5,
                ftfreqs[1:N/2] / (2. * np.pi))


def icwt(W, sj, dt, dj=0.25, w=Morlet()):
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
        w : instance of wavelet class, optional
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
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = np.ones([a, 1]) * sj
    else:
        raise Warning, 'Input array dimensions do not match.'

    # As of Torrence and Compo (1998), eq. (11)
    iW = dj * np.sqrt(dt) / w.cdelta * w.psi(0) * (np.real(W) / sj).sum(axis=0)
    return iW


def significance(signal, dt, scales, sigma_test=0, alpha=None,
                 significance_level=0.95, dof=-1, wavelet=Morlet()):
    """
    Significance testing for the onde dimensional wavelet transform.

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
    try:
      n0 = len(signal)
    except:
      n0 = 1
    J = len(scales) - 1
#    s0 = min(scales) # This is unused, not sure if thats a good thing or not.
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

    return (signif, fft_theor)


def xwt(signal, signal2, dt, significance_level=0.95, dj=1./12, s0=-1, J=-1, wavelet=Morlet(), normalize=True):
    """
    Cross wavelet transform. Both signals need to have the same length and the same dt
    
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
    y1 = np.asarray(signal) 
    y2 = np.asarray(signal2) 
      
    if normalize:
        std1 = y1.std()
        std2 = y2.std()
    else:
        std1 = std2 = 1.
    
    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    W1, sj, freqs, coi, signal_ft, ftfreqs = cwt(y1/std1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)

    W2, sj2, freqs2, coi2, signal_ft2, ftfreqs2 = cwt(y2/std2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    
    # Now the cross correlation of y1 and y2
    W12 = W1*W2.conj()
    
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
    Pk1 = ar1_spectrum(W1['freqs'] * dt, a1)
    Pk2 = ar1_spectrum(W2['freqs'] * dt, a2)
    dof = wavelet.dofmin
    PPF = chi2.ppf(significance_level, dof)
    signif = (std1 * std2 * (Pk1 * Pk2) ** 0.5 * PPF / dof)
    
    return W12, coi, freqs, signif

def wct(signal, signal2, dt, significance_level=0.95, dj=1./12, s0=-1, J=-1, wavelet=Morlet(), normalize=True):
    """
    Wavelet transform coherence.
    
    Parameters
    ----------
        x[1, 2], y[1, 2] (array like) :
            Input data arrays to calculate cross wavelet transform.
        significance_level (float, optional) :
            Significance level to use. Default is 0.95.
        normalize (boolean, optional) :
            If set to true, normalizes CWT by the standard deviation of
            the signals.
        result (string, optional) :
            If 'full' also returns intersected time-series. If set to
            'dictionary' returns the result arrays as itens of a 
            dictionary.
        kwargs (dictionary) :
            List of parameters like dt, dj, s0, J=-1 and wavelet.
            Please refer to the wavelet.cwt function documentation for
            further details.
    
    Returns
    -------
        Something : TBA and TBC
    
    See also
    --------
        wavelet.cwt, wavelet.xwt
    
    """ 
    y1 = np.asarray(signal) 
    y2 = np.asarray(signal2) 
      
    if normalize:
        std1 = y1.std()
        std2 = y2.std()
    else:
        std1 = std2 = 1.
    
    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    W1, sj, freqs, coi, signal_ft, ftfreqs = cwt(y1/std1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)

    W2, sj2, freqs2, coi2, signal_ft2, ftfreqs2 = cwt(y2/std2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)


    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj2[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(abs(W2) ** 2 / scales2, dt, dj, sj2)
      
    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.
    a1, _, _ = ar1(y1)
    a2, _, _ = ar1(y2)
    sig = wct_significance(a1, a2, significance_level=0.95,dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)

    return WCT, coi, freqs, sig[0], aWCT
    
    
def wct_significance(a1, a2, dt, dj, s0, J, wavelet, significance_level=0.95, mc_count=300, verbose=False):
    """
    Calculates wavelet coherence significance using Monte Carlo
    simulations with 95% confidence.
    
    PARAMETERS
        a1, a2 (float) :
            Lag-1 autoregressive coeficients of both time series.
        significance_level (float, optional) :
            Significance level to use. Default is 0.95.
        count (integer, optional) :
            Number of Monte Carlo simulations. Default is 300.
        verbose (boolean, optional) :
            If set to true, does not print anything on screen.
        kwargs (dictionary) :
            List of parameters like dt, dj, s0, J=-1 and wavelet.
            Please refer to the wavelet.cwt function documentation for
            further details.
    
    RETURNS
    
    """
    # Load cache if previously calculated. It is assumed that wavelet analysis
    # is performed using the wavelet's default parameters.
    aa = np.round(np.arctanh(np.array([a1, a2]) * 4))
    aa = np.abs(aa) + 0.5 * (aa < 0)
    cache = 'cache_%0.5f_%0.5f_%0.5f_%0.5f_%d_%s' % (aa[0], aa[1], dj,
        s0/dt, J, wavelet.name)
    cached = '%s/.klib/wavelet' % (expanduser("~"))
    try:
        dat = np.loadtxt('%s/%s.gz' % (cached, cache), unpack=True)
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
    ms = s0 * (2 ** (J * dj)) / dt
    N = np.ceil(ms * 6)
    noise1 = rednoise(N, a1, 1)
    nW1, sj, freqs, coi, signal_ft, ftfreqs = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    #
    period = np.ones([1, N]) / freqs[:, None]
    coi = np.ones([J+1, 1]) * coi[None, :]
    outsidecoi = (period <= coi)
    scales = np.ones([1, N]) * sj[:, None]
    #
    sig95 = np.zeros(J+1)
    maxscale = find(outsidecoi.any(axis=1))[-1]
    sig95[outsidecoi.any(axis=1)] = np.nan
    #
    nbins = 1000
    wlc = np.ma.zeros([J+1, nbins])
    t1 = time()
    for i in range(mc_count):
        t2 = time()
        # Generates two red-noise signals with lag-1 autoregressive 
        # coefficients given by a1 and a2
        noise1 = rednoise(N, a1, 1)
        noise2 = rednoise(N, a2, 1)
        # Calculate the cross wavelet transform of both red-noise signals
        nW1, sj, freqs, coi, signal_ft, ftfreqs = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
        nW2, sj2, freqs2, coi2, signal_ft2, ftfreqs2 = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
        nW12 = nW1 * nW2.conj()
        # Smooth wavelet wavelet transforms and calculate wavelet coherence
        # between both signals.
        S1 =wavelet.smooth(np.abs(nW1) ** 2 / scales, 
            dt, dj, sj)
        S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, 
            dt, dj, sj2)
        S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)
        R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
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
        sig95[s] = np.interp(significance_level, P, R2y[s:, sel])
    
    # Save the results on cache to avoid to many computations in the future
    try:
        makedirs(cached)
    except:
        pass
    np.savetxt('%s/%s.gz' % (cached, cache), [sig95, nW1['sj']])
    
    # And returns the results
    return sig95, sj


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
        s = '%.1f%%, %s (%s, %s)\n' % (perc, elap1, togo, elap2)
    elif (t1 == 0) and (t2 == 0):
        s = '%.1f%%, %s\n' % (perc, elap0)
    else:
        s = '%.1f%%, %s (%s, %s, %s)\n' % (perc, elap1, togo, elap0, elap2)
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
        
    return (hh, mm, ss, s)