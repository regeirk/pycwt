import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


AUTHORS = ('Sebastian Krieger, Nabil Freij, Alexey Brazhe, '
           'Christopher Torrence, Gilbert P. Compo and contributors')

setup(
    name='pycwt',
    version='0.3.0a8',
    author=AUTHORS,
    author_email='sebastian@nublia.com',
    description=('Continuous wavelet transform module for Python.'),
    license='BSD',
    url='https://github.com/regeirk/pycwt',
    packages=['pycwt'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'tqdm'],
    long_description=read('README.txt'),
    keywords=['wavelet', 'spectral analysis', 'signal processing',
              'data science'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Intended Audience :: Science/Research'
    ],
)
