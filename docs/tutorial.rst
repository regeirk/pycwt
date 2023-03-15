Tutorial
========

.. toctree::
   :maxdepth: 4

In the following example we will walk you through each step in order to use
PyCWT to perform the wavelet analysis of a given time-series.


Time-series spectral analysis using wavelets
--------------------------------------------

In this example we will follow the approach suggested by Torrence and Compo
(1998)\ [#f1]_, using the NINO3 sea surface temperature anomaly dataset
between 1871 and 1996. This and other sample data files are kindly provided by
C. Torrence and G. Compo
`here <http://paos.colorado.edu/research/wavelets/software.html>`__\ .

We begin by importing the relevant libraries. Please make sure that PyCWT is
properly installed in your system.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 10-15

Then, we load the dataset and define some data related parameters. In this
case, the first 19 lines of the data file contain meta-data, that we ignore,
since we set them manually (*i.e.* title, units).

.. literalinclude:: ./sample/simple_sample.py
   :lines: 20-26

We also create a time array in years.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 29-30

We write the following code to detrend and normalize the input data by its
standard deviation. Sometimes detrending is not necessary and simply
removing the mean value is good enough. However, if your dataset has a well
defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available in the
above mentioned website, it is strongly advised to perform detrending.
Here, we fit a one-degree polynomial function and then subtract it from the
original data.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 39-43

The next step is to define some parameters of our wavelet analysis. We
select the mother wavelet, in this case the Morlet wavelet with
:math:`\omega_0=6`.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 48-52

The following routines perform the wavelet transform and inverse wavelet
transform using the parameters defined above. Since we have normalized our
input time-series, we multiply the inverse transform by the standard
deviation.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 58-60

We calculate the normalized wavelet and Fourier power spectra, as well as
the Fourier equivalent periods for each wavelet scale.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 64-66

Optionally, we could also rectify the power spectrum according to the
suggestions proposed by Liu et al. (2007)\ [#f2]_

.. code-block:: python

   power /= scales[:, None]

We could stop at this point and plot our results. However we are also
interested in the power spectra significance test. The power is significant
where the ratio ``power / sig95 > 1``.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 71-75

Then, we calculate the global wavelet spectrum and determine its
significance level.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 79-83

We also calculate the scale average between 2 years and 8 years, and its
significance level.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 87-96

Finally, we plot our results in four different subplots containing the
(i) original series anomaly and the inverse wavelet transform; (ii) the
wavelet power spectrum (iii) the global wavelet and Fourier spectra ; and
(iv) the range averaged wavelet spectrum. In all sub-plots the significance
levels are either included as dotted lines or as filled contour lines.

.. literalinclude:: ./sample/simple_sample.py
   :lines: 104-166

Running this sequence of commands you should be able to generate the following
figure. If you don't want to type all the code manually, please download the
script source code using the link below.

.. plot:: ./sample/simple_sample.py

Wavelet analysis of the NINO3 Sea Surface Temperature record: (a) Time-
series (solid black line) and inverse wavelet transform (solid grey line),
(b) Normalized wavelet power spectrum of the NINO3 SST using the Morlet
wavelet (:math:`\omega_0=6`) as a function of time and of Fourier
equivalent wave period (in years). The black solid contour lines enclose
regions of more than 95% condence relative to a red-noise random process
(:math:`\alpha=0.77`). The cross-hatched and shaded area indicates the
affected by the cone of inuence of the mother wavelet. (iii) Global wavelet
power spectrum (solid black line) and Fourier power spectrum (solid grey
line). The dotted line indicates the 95% condence level.
(iv) Scale-averaged wavelet power over the 2--8 year band (solid black
line), power trend (solid grey line) and the 95% condence level (black
dotted line).

References
^^^^^^^^^^
.. [#f1] Torrence, C. and Compo, G. P.. A Practical Guide to Wavelet
         Analysis. Bulletin of the American Meteorological Society, *American
         Meteorological Society*, **1998**, 79, 61-78.
         `DOI <http://dx.doi.org/10.1175/1520-0477(1998)079\<0061:APGTWA\>2.0.CO;2>`__\ .

.. [#f2] Liu, Y., Liang, X. S. and Weisberg, R. H. Rectification of the bias
         in the wavelet power spectrum. *Journal of Atmospheric and Oceanic
         Technology*, **2007**, 24, 2093-2102.
         `DOI <http://dx.doi.org/10.1175/2007JTECHO511.1>`__\ .
