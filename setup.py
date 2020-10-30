# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de


import io
import os
import os.path as osp

from setuptools import find_packages, setup

import torch

# Package meta-data.
NAME = 'torch_trust_ncg'
DESCRIPTION = 'PyTorch Trust Region Newton Conjugate gradient method'
URL = ''
EMAIL = 'vassilis.choutas@tuebingen.mpg.de'
AUTHOR = 'Vassilis Choutas'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      packages=find_packages(),
      classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Environment :: Console",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"],
      install_requires=[
          'torch>=1.0.1',
      ],
      )
