# -*- coding: iso-8859-1 -*-
"""
Continuous wavelet transform plot module for Python.

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
    1 (2013-02-15 17:51 -0300)

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

__version__ = '$Revision: 1 $'
# $Source$

import numpy
from matplotlib import pyplot

def __init__(show=False):
    fontsize = 'medium'
    params = {'font.family': 'serif',
              'font.sans-serif': ['Helvetica'],
              'font.size': 18,
              'font.stretch': 'ultra-condensed',
              'text.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'text.usetex': True,
              'text.latex.unicode': True,
              'timezone': 'UTC'
             }
    pyplot.rcParams.update(params)
    pyplot.ion()


def figure(fp=dict(), ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95, 
    wspace=0.10, hspace=0.10), orientation='landscape'):
    """Creates a standard figure.
    
    PARAMETERS
        fp (dictionary, optional) :
            Figure properties.
        ap (dictionary, optional) :
            Adjustment properties.
    
    RETURNS
        fig : Figure object
    
    """

    __init__()
    
    if 'figsize' not in fp.keys():
        if orientation == 'landscape':
            fp['figsize'] = [11, 8]
        elif orientation == 'portrait':
            fp['figsize'] = [8, 11]
        elif orientation == 'squared':
            fp['figsize'] = [8, 8]
        elif orientation == 'worldmap':
            fp['figsize'] = [9, 5.0625] # Widescreen aspect ratio 16:9
        else:
            raise Warning, 'Orientation \'%s\' not allowed.' % (orientation, )
    
    fig = pyplot.figure(**fp)
    fig.subplots_adjust(**ap)
    
    return fig


def cwt(t, f, cwt, sig, rectify=False, **kwargs):
    """Plots the wavelet power spectrum.
    
    It rectifies the bias in the wavelet power spectrum as noted by
    Liu et al. (2007) dividing the power by the wavelet scales.
    
    PARAMETERS
        t (array like) :
            Time array.
        f (array like) :
            Function array.
        cwt (list) :
            List containing the results from wavelet.cwt function (e.g.
            wave, scales, freqs, coi, fft, fftfreqs)
        sig (list) :
            List containig the results from wavelet.significance funciton
            (e.g. signif, fft_theor)
        rectify (boolean, optional) :
            Sets wether to rectify the wavelet power by dividing by the 
            wavelet scales, according to Liu et al. (2007).
    
    RETURNS
        A list with the figure and axis objects for the plot.
    
    """
    # Sets some parameters and renames some of the input variables.
    N = len(t)
    dt = numpy.diff(t)[0]
    wave, scales, freqs, coi, fft, fftfreqs = cwt
    signif, fft_theor = sig
    #
    if 'std' in kwargs.keys():
        std = kwargs['std']
    else:
        std = f.std() # calculates standard deviation ...
    std2 = std ** 2   # ... and variance
    #
    period = 1. / freqs
    power = (abs(wave)) ** 2 # normalized wavelet power spectrum
    fft_power = std2 * abs(fft) ** 2 # FFT power spectrum
    sig95 = numpy.ones([1, N]) * signif[:, None]
    sig95 = power / sig95 # power is significant where ratio > 1
    if rectify:
        scales = numpy.ones([1, N]) * scales[:, None]
        levels = numpy.arange(0, 2.1, 0.1)
        labels = levels
    else:
        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']

    result = []
    
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
    else:
        fig = figure()
    result.append(fig)
    
    # Plots the normalized wavelet power spectrum and significance level 
    # contour lines and cone of influece hatched area.
    if 'fig' in kwargs.keys():
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(1,1,1)
    cmin, cmax = power.min(), power.max()
    rmin, rmax = min(levels), max(levels)
    if 'extend' in kwargs.keys():
        extend = kwargs['extend']
    elif (cmin < rmin) & (cmax > rmax):
        extend = 'both'
    elif (cmin < rmin) & (cmax <= rmax):
        extend = 'min'
    elif (cmin >= rmin) & (cmax > rmax):
        extend = 'max'
    elif (cmin >= rmin) & (cmax <= rmax):
        extend = 'neither'
    
    if rectify:
        cf = ax.contourf(t, numpy.log2(period), power / scales, levels, 
            extend=extend)
    else:
        cf = ax.contourf(t, numpy.log2(period), numpy.log2(power), 
        numpy.log2(levels), extend=extend)
    ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', 
        linewidths=2.)
    ax.fill(numpy.concatenate([t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt, 
        t[:1]-dt]), numpy.log2(numpy.concatenate([coi, [1e-9], 
        period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
    Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())), 
        numpy.ceil(numpy.log2(period.max())))
    ax.set_yticks(numpy.log2(Yticks))
    ax.set_yticklabels(Yticks)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
    ax.invert_yaxis()
    if rectify:
        cbar = fig.colorbar(cf)
    if not rectify:
        cbar = fig.colorbar(cf, ticks=numpy.log2(levels))
        cbar.ax.set_yticklabels(labels)
    result.append(ax)
    
    return result

def xwt(*args, **kwargs):
    """Plots the cross wavelet power spectrum and phase arrows.
    function.
    
    The relative phase relationship convention is the same as adopted
    by Torrence and Webster (1999), where in phase signals point 
    upwards (N), anti-phase signals point downwards (S). If X leads Y,
    arrows point to the right (E) and if X lags Y, arrow points to the
    left (W).
    
    PARAMETERS
        xwt (array like) : 
            Cross wavelet transform.
        coi (array like) :
            Cone of influence, which is a vector of N points containing
            the maximum Fourier period of useful information at that
            particular time. Periods greater than those are subject to
            edge effects.
        freqs (array like) :
            Vector of Fourier equivalent frequencies (in 1 / time units)
            that correspond to the wavelet scales.
        signif (array like) :
            Significance levels as a function of Fourier equivalent 
            frequencies.
        da (list, optional) :
            Pair of integers that the define frequency of arrows in 
            frequency and time, default is da = [3, 3].
    
    RETURNS
        A list with the figure and axis objects for the plot.
    
    SEE ALSO
        wavelet.xwt
    
    """
    # Sets some parameters and renames some of the input variables.
    xwt, t, coi, freqs, signif = args[:5]
    if 'scale' in kwargs.keys():
        scale = kwargs['scale']
    else:
        scale = 'log2'
    
    N = len(t)
    dt = t[1] - t[0]
    period = 1. / freqs
    power = abs(xwt)
    sig95 = numpy.ones([1, N]) * signif[:, None]
    sig95 = power / sig95 # power is significant where ratio > 1
    
    # Calculates the phase between both time series. The phase arrows in the 
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    if 'angle' in kwargs.keys():
        angle = 0.5 * numpy.pi - kwargs['angle']
    else:
        angle = 0.5 * numpy.pi - numpy.angle(xwt)
    u, v = numpy.cos(angle), numpy.sin(angle)
    
    result = []
    
    if 'da' in kwargs.keys():
        da = kwargs['da']
    else:
        da = [3, 3]
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
    else:
        fig = figure()
    result.append(fig)

    if 'fig' in kwargs.keys():
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(1, 1, 1)
    
    # Plots the cross wavelet power spectrum and significance level 
    # contour lines and cone of influece hatched area.
    if 'crange' in kwargs.keys():
        levels = labels = kwargs['crange']
    else:
        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']
    cmin, cmax = power.min(), power.max()
    rmin, rmax = min(levels), max(levels)
    if 'extend' in kwargs.keys():
        extend = kwargs['extend']
    elif (cmin < rmin) & (cmax > rmax):
        extend = 'both'
    elif (cmin < rmin) & (cmax <= rmax):
        extend = 'min'
    elif (cmin >= rmin) & (cmax > rmax):
        extend = 'max'
    elif (cmin >= rmin) & (cmax <= rmax):
        extend = 'neither'
    
    if scale == 'log2':
        Power = numpy.log2(power)
        Levels = numpy.log2(levels)
    else:
        Power = power
        Levels = levels
    
    cf = ax.contourf(t, numpy.log2(period), Power, Levels, extend=extend)
    ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', 
        linewidths=2.)
    q = ax.quiver(t[::da[1]], numpy.log2(period)[::da[0]], u[::da[0], ::da[1]],
        v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid',
        linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
        headaxislength=5, minshaft=2, minlength=5)
    ax.fill(numpy.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt, 
        t[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi, [1e-9], 
        period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
    Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())), 
        numpy.ceil(numpy.log2(period.max())))
    ax.set_yticks(numpy.log2(Yticks))
    ax.set_yticklabels(Yticks)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
    ax.invert_yaxis()
    cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
    cbar.ax.set_yticklabels(labels)
    result.append(ax)
    
    return result
