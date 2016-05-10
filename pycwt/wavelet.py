from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

try:
    import progressbar as pg
except ImportError:
    pg = None

import numpy as np
from scipy.stats import chi2
from .helpers import find, ar1, ar1_spectrum, rednoise, fftmod, fft_kwargs
from .mothers import Morlet, Paul, DOG, MexicanHat

mothers = {'morlet': Morlet,
           'paul': Paul,
           'dog': DOG,
           'mexicanhat': MexicanHat
           }

def cwt(signal, dt, dj=1/12, s0=-1, J=-1, wavelet='morlet'):
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
    >> mother = wavelet.Morlet(6.)
    >> wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
           0.25, 0.25, 0.5, 28, mother)

    """
    if isinstance(wavelet, unicode) or isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    # Original signal length.
    n0 = len(signal)
    # Smallest resolvable scale
    if s0 == -1: 
        s0 = 2 * dt / wavelet.flambda()
    # Number of scales
    if J == -1: 
        J = np.int(np.round(np.log2(n0 * dt / s0) / dj))
    ## CALLS TO THE FFT ARE ALSO SLOW.
    # Signal Fourier transform
    signal_ft = fftmod.fft(signal, **fft_kwargs(signal))
    N = len(signal_ft)
    # Fourier angular frequencies
    ftfreqs = 2 * np.pi * fftmod.fftfreq(N, dt)

    # The scales as of Mallat 1999
    sj = s0 * 2 ** (np.arange(0, J+1) * dj)
    freqs = 1 / (wavelet.flambda() * sj)

    # Creates wavelet transform matrix as outer product of scaled transformed
    # wavelets and transformed signal according to the convolution theorem.
    sj_col = sj[:,np.newaxis]   # transform to column vector for outer product
    # 2D matrix [s, f] for each scale s and Fourier angular frequency f
    psi_ft_bar = ((sj_col * ftfreqs[1] * N) ** .5 *
                  np.conjugate(wavelet.psi_ft(sj_col * ftfreqs)))
    W = fftmod.ifft(signal_ft * psi_ft_bar,
                    axis=1,  # transform along Fourier frequencies axis
                    # input not needed later, can be destroyed by FFTW
                    **fft_kwargs(signal_ft, overwrite_x=True))

    # Checks for NaN in transform results and removes them from the scales,
    # frequencies and wavelet transform.
    sel = np.invert(np.isnan(W).all(axis=1))
    if np.any(sel):             # attempt removal only if needed
        sj = sj[sel]
        freqs = freqs[sel]
        W = W[sel, :] ##SLOW

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi

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
    >> mother = wavelet.Morlet()
    >> wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
           0.25, 0.25, 0.5, 28, mother)
    >> iwave = wavelet.icwt(wave, scales, 0.25, 0.25, mother)

    """
    if isinstance(wavelet, unicode) or isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = np.ones([a, 1]) * sj
    else:
        raise Warning('Input array dimensions do not match.')

    # As of Torrence and Compo (1998), eq. (11)
    iW = (dj * np.sqrt(dt) / wavelet.cdelta * wavelet.psi(0) * 
          (np.real(W) / sj).sum(axis=0))
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

    Notes
    -----
    If sigma_test is set to 0, performs a regular chi-square test, according
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
    significance_level : float, optional
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
    if isinstance(wavelet, unicode) or isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    try:
      n0 = len(signal)
    except:
      n0 = 1
    J = len(scales) - 1
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
            raise Exception('DOF must be set to [s1, s2], '
                              'the range of scale-averages')
        if Cdelta == -1:
            raise Exception('Cdelta and dj0 not defined for %s with f0=%f' %
                             (wavelet.name, wavelet.f0))

        s1, s2 = dof
        sel = find((scales >= s1) & (scales <= s2));
        navg = sel.size
        if navg == 0:
            raise Exception('No valid scales between %d and %d.' % (s1, s2))

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
        raise Exception('sigma_test must be either 0, 1, or 2.')

    return signif, fft_theor


def xwt(signal, signal2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.95,
        wavelet='morlet', normalize=True):
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
    xwt (array like):
        Cross wavelet transform according to the selected mother wavelet.
    x (array like):
        Intersected independent variable.
    coi (array like):
        Cone of influence, which is a vector of N points containing
        the maximum Fourier period of useful information at that
        particular time. Periods greater than those are subject to
        edge effects.
    freqs (array like):
        Vector of Fourier equivalent frequencies (in 1 / time units)
        that correspond to the wavelet scales.
    signif (array like):
        Significance levels as a function of scale.

    """
    if isinstance(wavelet, unicode) or isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    # Defines some parameters like length of both time-series, time step
    # and calculates the standard deviation for normalization and statistical
    # significance tests.
    signal = np.asarray(signal)
    signal2 = np.asarray(signal2)

    if normalize:
        std1 = signal.std()
        std2 = signal2.std()
    else:
        std1 = std2 = 1

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.

    cwt_kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = cwt(signal/std1, **cwt_kwargs)
    W2, sj, freq, coi, _, _ = cwt(signal2/std2, **cwt_kwargs)

    # Now the cross correlation of y1 and y2
    W12 = W1 * W2.conj()

    # And the significance tests. Note that the confidence level is calculated
    # using the percent point function (PPF) of the chi-squared cumulative
    # distribution function (CDF) instead of using Z1(95%) = 2.182 and
    # Z2(95%)=3.999 as suggested by Torrence & Compo (1998) and Grinsted et
    # al. (2004). If the CWT has been normalized, then std1 and std2 should
    # be reset to unity, otherwise the standard deviation of both series have
    # to be calculated.
    if not normalize:
        std1 = signal.std()
        std2 = signal2.std()
    else:
        std1 = std2 = 1.
    a1, _, _ = ar1(signal)
    a2, _, _ = ar1(signal2)
    Pk1 = ar1_spectrum(freq * dt, a1)
    Pk2 = ar1_spectrum(freq * dt, a2)
    dof = wavelet.dofmin
    PPF = chi2.ppf(significance_level, dof)
    signif = (std1 * std2 * (Pk1 * Pk2) ** 0.5 * PPF / dof)

    # The resuts:
    return W12, coi, freq, signif


def wct(signal, signal2, dt, dj=1/12, s0=-1, J=-1, sig=True, significance_level=0.95,
        wavelet='morlet', normalize=True, **kwargs):
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
    if isinstance(wavelet, unicode) or isinstance(wavelet, str):
        wavelet = mothers[wavelet]()

    if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = np.int(np.round(np.log2(signal.size * dt / s0) / dj))  # Number of scales

   # Defines some parameters like length of both time-series, time step
    # and calculates the standard deviation for normalization and statistical
    # significance tests.

    signal = np.asarray(signal)
    signal2 = np.asarray(signal2)

    if normalize:
        std1 = signal.std()
        std2 = signal2.std()
    else:
        std1 = std2 = 1.

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    cwt_kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = cwt(signal/std1, **cwt_kwargs)
    W2, sj, freq, coi, _, _ = cwt(signal2/std2, **cwt_kwargs)

    scales1 = np.ones([1, signal.size]) * sj[:, None]
    scales2 = np.ones([1, signal2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, signal.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.
    a1, b1, c1 = ar1(signal)
    a2, b2, c2 = ar1(signal)
    if sig:
        kwargs.update(cwt_kwargs)
        sig = wct_significance(a1, a2, significance_level=significance_level,
                               **kwargs)
    else:
        sig = np.asarray([0])

    return WCT, aWCT, coi, freq, sig

def wct_significance(al1, al2, dt, dj, s0, J, significance_level, wavelet,
                     mc_count=300, progress=True, cache=True,
                     cache_dir='~/.pycwt/'):
    """
    Calculates wavelet coherence significance using Monte Carlo
    simulations with 95% confidence.


    Parameters
    ----------
    a1, a2: float
        Lag-1 autoregressive coeficients of both time series.
    significance_level (optional): float
        Significance level to use. Default is 0.95.
    count (optional): integer
        Number of Monte Carlo simulations. Default is 300.

    Returns
    -------
    """
    if progress and pg is None:
        raise ImportError('Progressbar is not installed')

    if cache:
        # Load cache if previously calculated. It is assumed that wavelet analysis
        # is performed using the wavelet's default parameters.
        aa = np.round(np.arctanh(np.array([al1, al2]) * 4))
        aa = np.abs(aa) + 0.5 * (aa < 0)
        cache = 'cache_{:0.5f}_{:0.5f}_{:0.5f}_{:0.5f}_{:d}_{}'.format(aa[0],
                                                 aa[1], dj, s0/dt, J, wavelet.name)
        cached = os.path.expanduser(os.path.join(cache_dir,'wct_sig'))
#        try:
#            dat = np.loadtxt('{}/{}.gz'.format(cached, cache), unpack=True)
#            print("\n\nNOTE: Loading from cache\n\n")
#            return dat
#        except IOError:
#            pass

    # Some output to the screen
    if progress:
        print('Calculating wavelet coherence significance')
        widgets = [pg.Percentage(), ' ', pg.Bar(), ' ', pg.ETA()]
        pbar = pg.ProgressBar(widgets=widgets, maxval=mc_count)
        pbar.start()

    # Choose N so that largest scale has at least some part outside the COI
    ms = s0 * (2 ** (J * dj)) / dt
    N = np.ceil(ms * 6)
    noise1 = rednoise(N, al1, 1)
    cwt_kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    nW1, sj, freq, coi, _, _ = cwt(noise1, **cwt_kwargs)

    period = np.ones([1, N]) / freq[:, None]
    coi = np.ones([J+1, 1]) * coi[None, :]
    outsidecoi = (period <= coi)
    scales = np.ones([1, N]) * sj[:, None]

    sig95 = np.zeros(J + 1)
    maxscale = find(outsidecoi.any(axis=1))[-1]
    sig95[outsidecoi.any(axis=1)] = np.nan

    nbins = 1000
    wlc = np.ma.zeros([J+1, nbins])
    for i in range(mc_count):
        # Generates two red-noise signals with lag-1 autoregressive
        # coefficients given by a1 and a2
        noise1 = rednoise(N, al1, 1)
        noise2 = rednoise(N, al2, 1)
        # Calculate the cross wavelet transform of both red-noise signals
        nW1, sj, freq, coi, _, _ = cwt(noise1, **cwt_kwargs)
        nW2, sj, freq, coi, _, _ = cwt(noise2, **cwt_kwargs)
        nW12 = nW1 * nW2.conj()
        # Smooth wavelet wavelet transforms and calculate wavelet coherence
        # between both signals.
        S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
        S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
        S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)
        R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
        # Walks through each scale outside the cone of influence and builds a
        # coherence coefficient counter.
        ## THIS LOOP IS THE SLOWEST PART OF THIS CODE!
        for s in range(maxscale):
            cd = np.floor(R2[s, :] * nbins)
            for j, t in enumerate(cd[~cd.mask]):
                wlc[s, t] += 1
        # Outputs some text to screen if desired
        if progress:
            pbar.update(i + 1)

    # After many, many, many Monte Carlo simulations, determine the
    # significance using the coherence coefficient counter percentile.
    wlc.mask = (wlc.data == 0.)
    R2y = (np.arange(nbins) + 0.5) / nbins
    for s in range(maxscale):
        sel = ~wlc[s, :].mask
        P = wlc[s, sel].data.cumsum()
        P = (P - 0.5) / P[-1]
        sig95[s] = np.interp(significance_level, P, R2y[sel])

    if cache:
        # Save the results on cache to avoid to many computations in the future
        try:
            os.makedirs(cached)
        except OSError:
            pass
        np.savetxt('{}/{}.gz'.format(cached, cache), sig95)

    # And returns the results
    return sig95
