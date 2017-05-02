import unittest

from helpers import (find, fftconv, ar1, ar1_spectrum, rednoise, rect,
                     boxpdf, get_cache_dir)


class HelpersTest(unittest.TestCase):
    def test_get_cache_dir(self):
        cache_dir = get_cache_dir()
        self.assertEqual(cache_dir, '/home/sebastian/.cache/pycwt/')
