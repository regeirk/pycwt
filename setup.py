import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

AUTHORS = "Nabil Freij" # TODO: Add the rest.

setup(
    name = "pycwt",
    version = "0.2",
    author = AUTHORS,
    description = ("Continuous wavelet transform module for Python."),
    license = "BSD",
    url = "https://github.com/nabobalis/pycwt",
    packages=['pycwt'],
    install_requires=['numpy','matplotlib','scipy'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
