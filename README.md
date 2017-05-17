PyCWT
=====

A Python module for continuous wavelet spectral analysis. It includes a
collection of routines for wavelet transform and statistical analysis via FFT
algorithm. In addition, the module also includes cross-wavelet transforms,
wavelet coherence tests and sample scripts.

Please read the documentation on http://regeirk.github.io/pycwt/.

This module requires ``NumPy`` and ``SciPy``. In addition, you also need
``matplotlib`` and ``progressbar`` to run the samples.

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
You can use PyPI to install this package.

    >> pip install pycwt

Or, you can download the code and run the below line within the top level
folder.

    >> python setup.py install


Comments
--------
There is an errata page at the wavelet website maintained at the Program
in Atmospheric and Oceanic Sciences, University of Colorado, Boulder,
Colorado, which is accessible at
http://paos.colorado.edu/research/wavelets/errata.html


> ## A Practical Guide to Wavelet Analysis
> **Christopher Torrence and Gilbert P. Compo**
>
> _Program in Atmospheric and Oceanic Sciences, University of Colorado,
> Boulder, Colorado_
>
>
> ### Errata
>
> - Figure 3: N/(2 sigma^2) should just be N/sigma^2.
> - Equation (17), left-hand side: Factor of 1/2 should be removed.
> - Table 1, DOG, Psi-hat (third column, bottom row): Should be a minus sign
>   in front of the equation.
> - Sec 3f, last paragraph: Plugging N=506, dt=1/4 yr, s0=2dt, and dj=0.125
>   into Eqn (10) actually gives J=64, not J=56 as stated in the text.
>   However, in Figure 1b, the scales are only plotted out to J=56 since the
>   power is so low at larger scales.
>
> ### Additional information
>
> Table 3: Cross-wavelet significance levels, from Eqn.(30)-(31). (DOF =
> degrees of freedom)
>
> Significance level | Real wavelet (1 DOF) | Complex wavelet (2 DOF)
> -------------------|----------------------|-------------------------
>        0.10        |        1.595         |          3.214
>        0.05        |        2.182         |          3.999
>        0.01        |        3.604         |          5.767


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
