Comments
========

There is an errata page at the wavelet website maintained at the Program
in Atmospheric and Oceanic Sciences, University of Colorado, Boulder,
Colorado, which is accessible at
http://paos.colorado.edu/research/wavelets/errata.html


A Practical Guide to Wavelet Analysis
-------------------------------------

**Christopher Torrence and Gilbert P. Compo** (*Program in Atmospheric and 
Oceanic Sciences, University of Colorado, Boulder, Colorado*)


Errata
^^^^^^

- Figure 3: N/(2 sigma^2) should just be N/sigma^2.
- Equation (17), left-hand side: Factor of 1/2 should be removed.
- Table 1, DOG, Psi-hat (third column, bottom row): Should be a minus sign
in front of the equation.
- Sec 3f, last paragraph: Plugging N=506, dt=1/4 yr, s0=2dt, and dj=0.125
into Eqn (10) actually gives J=64, not J=56 as stated in the text.
However, in Figure 1b, the scales are only plotted out to J=56 since the
power is so low at larger scales.


Additional information
^^^^^^^^^^^^^^^^^^^^^^

Table 3: Cross-wavelet significance levels, from Eqn.(30)-(31). (DOF =
degrees of freedom)

==================  ====================  =======================
Significance level  Real wavelet (1 DOF)  Complex wavelet (2 DOF)
==================  ====================  =======================
0.10                1.595                 3.214
0.05                2.182                 3.999
0.01                3.604                 5.767
==================  ====================  =======================
