# -*- coding: iso-8859-1 -*-
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

__version__ = '$Revision: 3 $'
# $Source$

from numpy import (asarray, arange, array, argsort, arctanh, ceil, concatenate, conjugate, cos, diff, exp, intersect1d, isnan, isreal, log, log2, mod, ones, pi, prod, real, round, sort, sqrt, unique, zeros, polyval, nan, ma, floor, interp, loadtxt, savetxt, angle)
from numpy.fft import fft, ifft, fftfreq
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
from time import time

class Morlet:
    """Implements the Morlet wavelet class.

    Note that the input parameters f and f0 are angular frequencies.
    f0 should be more than 0.8 for this function to be correct, its
    default value is f0=6.

    """

    name = 'Morlet'

    def __init__(self, f0=6.0):
        self._set_f0(f0)

    def psi_ft(self, f):
        """Fourier transform of the approximate Morlet wavelet."""
        return (pi ** -.25) * exp(-0.5 * (f - self.f0) ** 2.)

    def psi(self, t):
        """Morlet wavelet as described in Torrence and Compo (1998)."""
        return (pi ** -.25) * exp(1j * self.f0 * t - t ** 2. / 2.)

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (4 * pi) / (self.f0 + sqrt(2 + self.f0 ** 2))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / coi

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
    
    def smooth(self, W, dt, dj, scales):
        """Smoothing function used in coherence analysis."""
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a 
        # total weight of unity, according to suggestions by Torrence & 
        # Webster (1999) and bry Grinsted et al. (2004).
        
        m, n = W.shape
        T = zeros([m, n])
        
        # Filter in time. 
        npad = int(2 ** ceil(log2(n)))
        k = 2 * pi * fftfreq(npad)
        k2 = k ** 2
        snorm = scales / dt
        
        for i in range(m):
            F = exp(-0.5 * (snorm[i] ** 2) * k2)
            smooth = ifft(F * fft(W[i, :], npad))
            T[i, :] = smooth[0:n]
        
        if isreal(W).all():
            T = T.real
        
        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = self.deltaj0 / dj * 2
        win = rect(int(round(wsize)), normalize=True)
        T = convolve2d(T, win[:, None], 'same')
        
        return T
        

class Paul:
    """Implements the Paul wavelet class.

    Note that the input parameter f is the angular frequency and that
    the default order for this wavelet is m=4.

    """

    name = 'Paul'

    def __init__(self, m=4):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the Paul wavelet."""
        return (2 ** self.m / sqrt(self.m * prod(range(2, 2 * self.m))) *
                f ** self.m * exp(-f) * (f > 0))

    def psi(self, t):
        """Paul wavelet as described in Torrence and Compo (1998)."""
        return (2 ** self.m * 1j ** self.m * prod(range(2, self.m - 1)) /
                sqrt(pi * prod(range(2, 2 * self.m + 1))) * (1 - 1j * t) **
                (-(self.m + 1)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return 4 * pi / (2 * self.m + 1)

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / coi

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
    """Implements the derivative of a Guassian wavelet class.

    Note that the input parameter f is the angular frequency and that
    for m=2 the DOG becomes the Mexican hat wavelet, which is then
    default.

    """

    name = 'DOG'

    def __init__(self, m=2):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the DOG wavelet."""
        return (- 1j ** self.m / sqrt(gamma(self.m + 0.5)) * f ** self.m *
                exp(- 0.5 * f ** 2))

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
        return ((-1) ** (self.m + 1) * polyval(p, t) * exp(-t ** 2 / 2) /
                sqrt(gamma(self.m + 0.5)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (2 * pi / sqrt(self.m + 0.5))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / coi

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
    """Implements the Mexican hat wavelet class.

    This class inherits the DOG class using m=2.

    """

    name = 'Mexican hat'

    def __init__(self):
        self._set_m(2)


def fftconv(x, y):
    """ Convolution of x and y using the FFT convolution theorem. """
    N = len(x)
    n = int(2 ** ceil(log2(N))) + 1
    X, Y, x_y = fft(x, n), fft(y, n), []
    for i in range(n):
        x_y.append(X[i] * Y[i])

    # Returns the inverse Fourier transform with padding correction
    return ifft(x_y)[4:N+4]


def ar1(x):
    r"""Allen and Smith autoregressive lag-1 autocorrelation alpha. In a
    AR(1) model
    
        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,
    
    where <x> is the process mean, \gamma and \alpha are process 
    parameters and z(t) is a Gaussian unit-variance white noise.
        
    PARAMETERS
        x (array like) :
            Univariate time series
    
    RETURNS
        g (float) :
            Estimate of the lag-one autocorrelation.
        a (float) :
            Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
        mu2 (foat) :
            Estimated square on the mean of a finite segment of AR(1) 
            noise, mormalized by the process variance.
    
    REFERENCES
        [1] Allen, M. R. and Smith, L. A. (1996). Monte Carlo SSA: 
            detecting irregular oscillations in the presence of colored 
            noise. Journal of Climate, 9(12), 3373-3404.

    """
    x = asarray(x)
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
    """Lag-1 autoregressive theoretical power spectrum
    
    PARAMETERS
        ar1 (float) :
            Lag-1 autoregressive correlation coefficient.
        freqs (array like) :
            Frequencies at which to calculate the theoretical power 
            spectrum.
    
    RETURNS
        Pk (array like) :
            Theoretical discrete Fourier power spectrum of noise signal.
    
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
    freqs = asarray(freqs)
    Pk = (1 - ar1 ** 2) / abs(1 - ar1 * exp(-2 * pi * 1j * freqs)) ** 2

    # Theoretical discrete Fourier power spectrum of the noise signal following
    # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
    #N = len(freqs)
    #Pk = (1 - ar1 ** 2) / (1 + ar1 ** 2 - 2 * ar1 * cos(2 * pi * freqs / N))
    
    return Pk


def rednoise(N, g, a=1.) :
    """Red noise generator using filter.
    
    PARAMETERS
        N (integer) :
            Length of the desired time series.
        g (float) :
            Lag-1 autocorrelation coefficient.
        a (float, optional) :
            Noise innovation variance parameter.
    
    RETURNS
        y (array like) :
            Red noise time series.
    
    """
    if g == 0:
        yr = randn(N, 1) * a;
    else:
        # Twice the decorrelation time.
        tau = ceil(-2 / log(abs(g)))
        yr = lfilter([1, 0], [1, -g], randn(N + tau, 1) * a)
        yr = yr[tau:]
    
    return yr.flatten()


def rect(x, normalize=False) :
    if type(x) in [int, float]:
        shape = [x, ]
    elif type(x) in [list, dict]:
        shape = x
    elif type(x) in [numpy.ndarray, numpy.ma.core.MaskedArray]:
        shape = x.shape
    X = zeros(shape)
    X[0] = X[-1] = 0.5
    X[1:-1] = 1
    
    if normalize:
        X /= X.sum()
    
    return X


def cwt(signal, dt=1., dj=1./12, s0=-1, J=-1, wavelet=Morlet(), result=None):
    """Continuous wavelet transform of the signal at specified scales.

    PARAMETERS
        signal (array like) :
            Input signal array
        dt (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales. Default value is 0.25.
            Smaller values will result in better scale resolution, but
            slower calculation and plot.
        s0 (float, optional) :
            Smallest scale of the wavelet. Default value is 2*dt.
        J (float, optional) :
            Number of scales less one. Scales range from s0 up to
            s0 * 2**(J * dj), which gives a total of (J + 1) scales.
            Default is J = (log2(N*dt/so))/dj.
        wavelet (class, optional) :
            Mother wavelet class. Default is Morlet()
        result (string, optional) :
            If set to 'dictionary' returns the result arrays as itens
            of a dictionary.

    RETURNS
        W (array like) :
            Wavelet transform according to the selected mother wavelet.
            Has (J+1) x N dimensions.
        sj (array like) :
            Vector of scale indices given by sj = s0 * 2**(j * dj),
            j={0, 1, ..., J}.
        freqs (array like) :
            Vector of Fourier frequencies (in 1 / time units) that
            corresponds to the wavelet scales.
        coi (array like) :
            Returns the cone of influence, which is a vector of N
            points containing the maximum Fourier period of useful
            information at that particular time. Periods greater than
            those are subject to edge effects.
        fft (array like) :
            Normalized fast Fourier transform of the input signal.
        fft_freqs (array like):
            Fourier frequencies (in 1/time units) for the calculated
            FFT spectrum.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)

    """
    n0 = len(signal)                              # Original signal length.
    if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = int(log2(n0 * dt / s0) / dj)  # Number of scales
    N = 2 ** (int(log2(n0)) + 1)                  # Next higher power of 2.
    signal_ft = fft(signal, N)                    # Signal Fourier transform
    ftfreqs = 2 * pi * fftfreq(N, dt)             # Fourier angular frequencies

    sj = s0 * 2. ** (arange(0, J+1) * dj)         # The scales
    freqs = 1. / (wavelet.flambda() * sj)         # As of Mallat 1999

    # Creates an empty wavlet transform matrix and fills it for every discrete
    # scale using the convolution theorem.
    W = zeros((len(sj), N), 'complex')
    for n, s in enumerate(sj):
        psi_ft_bar = ((s * ftfreqs[1] * N) ** .5 * 
            conjugate(wavelet.psi_ft(s * ftfreqs)))
        W[n, :] = ifft(signal_ft * psi_ft_bar, N)

    # Checks for NaN in transform results and removes them from the scales,
    # frequencies and wavelet transform.
    sel = ~isnan(W).all(axis=1)
    sj = sj[sel]
    freqs = freqs[sel]
    W = W[sel, :]

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2. - abs(arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi
    #
    if result == 'dictionary':
        result = dict(
            W = W[:, :n0],
            sj = sj,
            freqs = freqs,
            #period = 1. / freqs,
            coi = coi,
            signal_ft = signal_ft[1:N/2] / N ** 0.5,
            ftfreqs = ftfreqs[1:N/2] / (2. * pi),
            dt = dt,
            dj = dj,
            s0 = s0,
            J = J,
            wavelet = wavelet
        )
        return result
    else:
        return (W[:, :n0], sj, freqs, coi, signal_ft[1:N/2] / N ** 0.5,
                ftfreqs[1:N/2] / (2. * pi))


def icwt(W, sj, dt, dj=0.25, w=Morlet()):
    """Inverse continuous wavelet transform.

    PARAMETERS
        W (array like):
            Wavelet transform, the result of the cwt function.
        sj (array like):
            Vector of scale indices as returned by the cwt function.
        dt (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales as used in the cwt
            function. Default value is 0.25.
        w (class, optional) :
            Mother wavelet class. Default is Morlet()

    RETURNS
        iW (array like) :
            Inverse wavelet transform.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)
        iwave = wavelet.icwt(wave, scales, 0.25, 0.25, mother)

    """
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = ones([a, 1]) * sj
    else:
        raise Warning, 'Input array dimensions do not match.'

    # As of Torrence and Compo (1998), eq. (11)
    iW = dj * sqrt(dt) / w.cdelta * w.psi(0) * (real(W) / sj).sum(axis=0)
    return iW


def significance(signal, dt, scales, sigma_test=0, alpha=None,
                 significance_level=0.95, dof=-1, wavelet=Morlet()):
    """
    Significance testing for the onde dimensional wavelet transform.

    PARAMETERS
        signal (array like or float) :
            Input signal array. If a float number is given, then the
            variance is assumed to have this value. If an array is
            given, then its variance is automatically computed.
        dt (float, optional) :
            Sample spacing. Default is 1.0.
        scales (array like) :
            Vector of scale indices given returned by cwt function.
        sigma_test (int, optional) :
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
        alpha (float, optional) :
            Lag-1 autocorrelation, used for the significance levels.
            Default is 0.0.
        significance_level (float, optional) :
            Significance level to use. Default is 0.95.
        dof (variant, optional) :
            Degrees of freedom for significance test to be set
            according to the type set in sigma_test.
        wavelet (class, optional) :
            Mother wavelet class. Default is Morlet().

    RETURNS
        signif (array like) :
            Significance levels as a function of scale.
        fft_theor (array like):
            Theoretical red-noise spectrum as a function of period.

    """
    try:
      n0 = len(signal)
    except:
      n0 = 1
    J = len(scales) - 1
    s0 = min(scales)
    dj = log2(scales[1] / scales[0])

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
        cos(2 * pi * k / N))
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
            dof = zeros(1, J+1) + dof
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
        Smid = exp((log(s1) + log(s2)) / 2.)
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


def xwt(x1, y1, x2, y2, significance_level=0.95, normalize=True, result=None,
    **kwargs):
    """Cross wavelet transform.
    
    PARAMETERS
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
        kwargs (list) :
            List of parameters like dt, dj, s0, J=-1 and wavelet.
            Please refer to the wavelet.cwt function documentation for
            further details.
    
    RETURNS
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
    
    SEE ALSO
        wavelet.cwt, wavelet.wct
    
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
    
    # If both time series are different, determines the intersection of both
    # to ensure same data length.
    x = intersect1d(x1, x2)
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
    if result == 'dictionary':
        result = dict(
            XWT = W12,
            coi = coi,
            freqs = freqs,
            signif = signif,
            t = x,
            y1 = y1,
            y2 = y2
        )
        return result
    elif result == 'full' :
        return W12, x, coi, freqs, signif, y1, y2
    else:
        return W12, x, coi, freqs, signif


def wct(x1, y1, x2, y2, significance_level=0.95, normalize=True, result=None,
    **kwargs):
    """Wavelet transform coherence.
    
    PARAMETERS
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
    
    RETURNS
    
    SEE ALSO
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
    scales1 = ones([1, n1]) * W1['sj'][:, None]
    scales2 = ones([1, n2]) * W1['sj'][:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = kwargs['wavelet'].smooth(abs(W1['W']) ** 2 / scales1, dx1, W1['dj'], 
        W1['sj'])
    S2 = kwargs['wavelet'].smooth(abs(W2['W']) ** 2 / scales2, dx2, W2['dj'], 
        W1['sj'])
    
    # If both time series are different, determines the intersection of both
    # to ensure same data length.
    x = intersect1d(x1, x2)
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
    scales = ones([1, n]) * W1['sj'][:, None]
    S12 = kwargs['wavelet'].smooth(W12 / scales, dx1, W1['dj'], W1['sj'])
    WCT = abs(S12) ** 2 / (S1 * S2)
    aWCT = angle(W12)
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
    aa = round(arctanh(array([a1, a2]) * 4))
    aa = abs(aa) + 0.5 * (aa < 0)
    cache = 'cache_%0.5f_%0.5f_%0.5f_%0.5f_%d_%s' % (aa[0], aa[1], kwargs['dj'],
        kwargs['s0']/kwargs['dt'], kwargs['J'], kwargs['wavelet'].name)
    cached = '%s/.klib/wavelet' % (expanduser("~"))
    try:
        dat = loadtxt('%s/%s.gz' % (cached, cache), unpack=True)
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
    N = ceil(ms * 6)
    noise1 = rednoise(N, a1, 1)
    nW1 = cwt(noise1, **kwargs)
    #
    period = ones([1, N]) / nW1['freqs'][:, None]
    coi = ones([kwargs['J']+1, 1]) * nW1['coi'][None, :]
    outsidecoi = (period <= coi)
    scales = ones([1, N]) * nW1['sj'][:, None]
    #
    sig95 = zeros(kwargs['J'] + 1)
    maxscale = find(outsidecoi.any(axis=1))[-1]
    sig95[outsidecoi.any(axis=1)] = nan
    #
    nbins = 1000
    wlc = ma.zeros([kwargs['J']+1, nbins])
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
        R2 = ma.array(abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
        # Walks through each scale outside the cone of influence and builds a
        # coherence coefficient counter.
        for s in range(maxscale):
            cd = floor(R2[s, :] * nbins)
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
    R2y = (arange(nbins) + 0.5) / nbins
    for s in range(maxscale):
        sel = ~wlc[s, :].mask
        P = wlc[s, sel].data.cumsum()
        P = (P - 0.5) / P[-1]
        sig95[s] = interp(significance_level, P, R2y[sel])
    
    # Save the results on cache to avoid to many computations in the future
    try:
        makedirs(cached)
    except:
        pass
    savetxt('%s/%s.gz' % (cached, cache), [sig95, nW1['sj']])
    
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
