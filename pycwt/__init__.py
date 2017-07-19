"""Python module for continuous wavelet spectral analysis.

This module includes a collection of routines for wavelet transform and
statistical analysis via FFT algorithm. In addition, the module also
includes cross-wavelet transforms, wavelet coherence tests and sample
scripts.


Functions
---------
* cwt : Continuous wavelet transform.
* icwt: Inverse continuous wavelet transform.
* significance : Significance test for the one dimensional wavelet
                 transform.
* xwt : cross-wavelet transform.
* wct : Wavelet coherence transform.
* wct_significance : Wavelet coherence significance using Monte Carlo
                     simulations.


Classes
-------
* Morlet : Morlet wavelet
* Paul : Paul wavelet
* DOG : Derivative of a Guassian wavelet family
* MexicanHat : Mexican hat wavelet


Disclaimer
----------
This module is based on routines provided by C. Torrence and G. P. Compo
available at http://paos.colorado.edu/research/wavelets/, on routines
provided by A. Grinsted, J. Moore and S. Jevrejeva available at
http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
on routines provided by A. Brazhe available at
http://cell.biophys.msu.ru/static/swan/.

This software is released under a BSD-style open source license. Please read
the license file for further information. This routine is provided as is
without any express or implied warranties whatsoever.


Acknowledgements
----------------
We would like to thank Christopher Torrence, Gilbert P. Compo, Aslak Grinsted,
John Moore, Svetlana Jevrejevaand and Alexey Brazhe for their code and also
Jack Ireland and Renaud Dussurget for their attentive eyes, feedback and
debugging.


Authors
-------
Sebastian Krieger, Nabil Freij, Alexey Brazhe, Christopher Torrence,
Gilbert P. Compo and contributors.


References
----------
1. Torrence, C. and Compo, G. P.. A Practical Guide to Wavelet
   Analysis. Bulletin of the American Meteorological Society, *American
   Meteorological Society*, **1998**, 79, 61-78.
   http://dx.doi.org/10.1175/1520-0477(1998)079\<0061:APGTWA\>2.0.CO;2>
2. Torrence, C. and Webster, P. J.. Interdecadal changes in the
   ENSO-Monsoon system, *Journal of Climate*, **1999**, 12(8),
   2679-2690. http://dx.doi.org/10.1175/1520-0442(1999)012\<2679:ICITEM\>2.0.CO;2>
3. Grinsted, A.; Moore, J. C. & Jevrejeva, S. Application of the cross
   wavelet transform and wavelet coherence to geophysical time series.
   *Nonlinear Processes in Geophysics*, **2004**, 11, 561-566.
   http://dx.doi.org/10.5194/npg-11-561-2004
4. Mallat, S.. A wavelet tour of signal processing: The sparse way.
   *Academic Press*, **2008**, 805.
5. Addison, P. S. The illustrated wavelet transform handbook:
   introductory theory and applications in science, engineering,
   medicine and finance. *IOP Publishing*, **2002**.
   http://dx.doi.org/10.1201/9781420033397
6. Liu, Y., Liang, X. S. and Weisberg, R. H. Rectification of the bias
   in the wavelet power spectrum. *Journal of Atmospheric and Oceanic
   Technology*, **2007**, 24, 2093-2102.
   http://dx.doi.org/10.1175/2007JTECHO511.1

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from . wavelet import *

__all__ = ['cwt', 'icwt', 'significance', 'xwt', 'wct', 'wct_significance',
           'mothers', 'Morlet', 'Paul', 'DOG', 'MexicanHat']
__name__ = 'PyCWT'
__version__ = '0.3.0a8'
