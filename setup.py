import os
import sys
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('SEQC requires Python 3')
if sys.version_info.minor < 5:
    warn('Multiprocessing analysis methods may not function on Python versions < 3.5')


# get version
with open('src/rut/version.py') as f:
    exec(f.read())

setup(name='rut',
      version=__version__,  # read in from the exec of version.py; ignore error
      description=('Resampled non-parametric tests for distributions with unequal sampling'
                   'and variance'),
      author='Ambrose J. Carr',
      author_email='mail@ambrosejcarr.com',
      package_dir={'': 'src'},
      packages=['Rut'],
      install_requires=[
          'numpy',
          'pandas>=0.18.1',
          'scipy>=0.14.0',
          'statsmodels'],
      )

