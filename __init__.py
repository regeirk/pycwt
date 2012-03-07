# -*- coding: iso-8859-1 -*-
"""
Continuous wavelet transform module for Python. Includes a collection
of routines for wavelet transform and statistical analysis via FFT
algorithm. This module references to the numpy, scipy and pylab Python
packages.

DISCLAIMER
    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/
    and on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com

REVISION
    3 (2011-09-04 20:28 -0300)
    2 (2011-04-28 17:57 -0300)
    1 (2010-12-24 21:59 -0300)

REFERENCES
    [1] Mallat, Stephane G. (1999). A wavelet tour of signal processing
    [2] Addison, Paul S. The illustrated wavelet transform handbook
    [3] Torrence, Christopher and Compo, Gilbert P. (1998). A Practical
        Guide to Wavelet Analysis

"""

__version__ = '$Revision: 3 $'
# $Source$

from wavelet import cwt, icwt, significance, Morlet, Paul, DOG, Mexican_hat

__all__ = ['cwt', 'icwt', 'significance', 'Morlet', 'Paul', 'DOG',
           'Mexican_hat']
