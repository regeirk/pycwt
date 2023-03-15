|ReadTheDocs| |PyPi| |Travis|

PyCWT
=====

A Python module for continuous wavelet spectral analysis. It includes a
collection of routines for wavelet transform and statistical analysis via FFT
algorithm. In addition, the module also includes cross-wavelet transforms,
wavelet coherence tests and sample scripts.

Please read the documentation `here <http://pycwt.readthedocs.io/en/latest/>`__\.

This module requires ``NumPy``, ``SciPy``, ``tqdm``. In addition, you will 
also need ``matplotlib`` to run the examples.

The sample scripts (``sample.py``, ``sample_xwt.py``) illustrate the use of
the wavelet and inverse wavelet transforms, cross-wavelet transform and
wavelet transform coherence. Results are plotted in figures similar to the
sample images.


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


Installation
------------

We recommend using PyPI to install this package.

.. code-block:: sh

    $ pip install pycwt

However, if you want to install directly from GitHub, use:

.. code-block:: sh

    $ pip install git+https://github.com/regeirk/pycwt


Acknowledgements
----------------

We would like to thank Christopher Torrence, Gilbert P. Compo, Aslak Grinsted,
John Moore, Svetlana Jevrejevaand and Alexey Brazhe for their code and also
Jack Ireland and Renaud Dussurget for their attentive eyes, feedback and
debugging.


Contributors
------------

- Sebastian Krieger
- Nabil Freij
- Ken Mankoff
- Aaron Nielsen
- Rodrigo Nemmen
- Ondrej Grover
- Joscelin Rocha Hidalgo
- Stuart Mumford
- ymarcon1
- Tariq Hassan


References
----------

1. Torrence, C. and Compo, G. P.. A Practical Guide to Wavelet
   Analysis. Bulletin of the American Meteorological Society, *American
   Meteorological Society*, **1998**, 79, 61-78.
2. Torrence, C. and Webster, P. J.. Interdecadal changes in the
   ENSO-Monsoon system, *Journal of Climate*, **1999**, 12(8),
   2679-2690.
3. Grinsted, A.; Moore, J. C. & Jevrejeva, S. Application of the cross
   wavelet transform and wavelet coherence to geophysical time series.
   *Nonlinear Processes in Geophysics*, **2004**, 11, 561-566.
4. Mallat, S.. A wavelet tour of signal processing: The sparse way.
   *Academic Press*, **2008**, 805.
5. Addison, P. S. The illustrated wavelet transform handbook:
   introductory theory and applications in science, engineering,
   medicine and finance. *IOP Publishing*, **2002**.
6. Liu, Y., Liang, X. S. and Weisberg, R. H. Rectification of the bias
   in the wavelet power spectrum. *Journal of Atmospheric and Oceanic
   Technology*, **2007**, 24, 2093-2102.


.. |ReadTheDocs| image:: https://readthedocs.org/projects/pycwt/badge/?version=latest
   :target: http://pycwt.readthedocs.io/en/latest/?badge=latest

.. |PyPi| image:: https://badge.fury.io/py/pycwt.svg
   :target: https://badge.fury.io/py/pycwt

.. |Travis| image:: https://travis-ci.org/regeirk/pycwt.svg?branch=master
   :target: https://travis-ci.org/regeirk/pycwt
