"""
Dataset manager for the wavelet analysis sample script.


Disclaimer
----------
This software is released under a BSD-style open source license. Please read
the license file for furter information. This routine is provided as is without
any express or implied warranties whatsoever.

Authors
-------
Sebastian Krieger, Nabil Freij

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import loadtxt


class Dataset:
    """
    A simple dataset class.

    Example
    -------
    ds = Dataset('NINO3', usetex=True)
    dat = ds.load()

    """

    def __init__(self, name, usetex=True):
        """
        Initializes the dataset class.

        Parameters
        ----------
        name : string
            Valid options are either 'NINO3', 'MAUNA', 'MONSOON',
            'SUNSPOTS' or 'SOI'.
        usetext : boolean, optional
            If 'True' (default) uses TeX string formating.

        """
        if name == 'NINO3':
            (self.fname, self.title, self.label, self.t0, self.dt, self.units,
             self.units2) = self._sample_nino3(usetex)
        elif name == 'MAUNA':
            (self.fname, self.title, self.label, self.t0, self.dt, self.units,
             self.units2) = self._sample_mauna(usetex)
        elif name == 'MONSOON':
            (self.fname, self.title, self.label, self.t0, self.dt, self.units,
             self.units2) = self._sample_monsoon(usetex)
        elif name == 'SUNSPOTS':
            (self.fname, self.title, self.label, self.t0, self.dt, self.units,
             self.units2) = self._sample_sunspots(usetex)
        elif name == 'SOI':
            (self.fname, self.title, self.label, self.t0, self.dt, self.units,
             self.units2) = self._sample_soi(usetex)
        else:
            raise ValueError('No valid dataset chosen.')

    def load(self):
        """Return sample data."""
        return loadtxt(self.fname)

    def _sample_nino3(self, usetex=True):
        """Return NINO3 attributes."""
        fname = 'sst_nino3.dat'
        title = 'NINO3 Sea Surface Temperature (seasonal)'
        t0 = 1871
        dt = 0.25
        label = 'NINO3 SST'
        if usetex:
            units = r'$^{\circ}\textnormal{C}$'
            units2 = r'$(^{\circ} \textnormal{C})^2$'
        else:
            units = 'degC'
            units2 = 'degC^2'
        return fname, title, label, t0, dt, units, units2

    def _sample_mauna(self, usetex=True):
        """Return Mauna Loa attributes."""
        fname = 'mauna.dat'
        title = 'Mauna Loa Carbon Dioxide'
        t0 = 1958.0
        dt = 0.08333333
        units = 'ppm'
        if usetex:
            label = 'Mauna Loa CO$_{2}$'
            units2 = '{}$^2$'.format(units)
        else:
            label = 'Mauna Loa CO2'
            units2 = '{}^2'.format(units)
        return fname, title, label, t0, dt, units, units2

    def _sample_monsoon(self, usetex=True):
        """Return All-India Monsoon attributes."""
        fname = 'monsoon.dat'
        title = 'All-India Monsoon Rainfall'
        t0 = 1871.0
        dt = 0.25
        label = 'Rainfall'
        units = 'mm'
        if usetex:
            units2 = '{}$^2$'.format(units)
        else:
            units2 = '{}^2'.format(units)
        return fname, title, label, t0, dt, units, units2

    def _sample_sunspots(self, usetex=True):
        """Return Wolf\'s Sunspot attributes."""
        fname = 'sunspot.dat'
        title = 'Wolf\'s Sunspot Number'
        label = 'Sunspots'
        t0 = 1748
        dt = 0.25
        units = ''
        units2 = ''
        return fname, title, label, t0, dt, units, units2

    def _sample_soi(self, usetex=True):
        """Return SOI attributes."""
        fname = 'soi.dat'
        title = 'Southern Oscillation Index'
        label = 'SOI'
        t0 = 1896
        dt = 0.25
        units = 'mb'
        if usetex:
            units2 = '{}$^2$'.format(units)
        else:
            units2 = '{}^2'.format(units)
        return fname, title, label, t0, dt, units, units2
