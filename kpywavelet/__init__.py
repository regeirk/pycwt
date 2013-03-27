"""
kPyWavelet
==========

Continuous wavelet transform module for Python. 

Web Links
---------
Source Code:  https://github.com/regeirk/kPyWavelet

Available subpackages
---------------------

2d
    TBA
plot
    TBA
sample
    TBA
sample_xwt
    TBA
twod
    TBA
wavelet
    TBA

"""

__version__ = '0.4'

from wavelet import *
#from wavelet import (ar1, ar1_spectrum, cwt, icwt, significance, xwt, wct, Morlet,
#    Paul, DOG, Mexican_hat)
import plot
from . Examples import *

__all__ = ['ar1', 'ar1_spectrum', 'cwt', 'icwt', 'significance', 'xwt', 'wct',
    'Morlet', 'Paul', 'DOG', 'Mexican_hat', ]