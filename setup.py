#!/usr/bin/env python
import os

from setuptools import setup, find_packages

VERSION = 0.0.1
version = os.path.join('rh_aligner', '__init__.py')
execfile(version)

README = open('README.md').read()

setup(
    name='rh_aligner',
    version=VERSION,
    packages=find_packages(),
    author='Adi Suissa-Peleg',
    author_email='adisuis@seas.harvard.edu',
    url="https://github.com/Rhoana/rh_aligner",
    description="Rhoana's 2D and 3D alignment tool",
    long_description=README,
    include_package_data=True,
    install_requires=[
        "numpy>=1.9.3",
        "scipy>=0.16.0",
    ],
    dependency_links = ['http://github.com/Rhoana/rh_renderer/tarball/master#egg=rh_renderer-0.0.1']
    zip_safe=False
)
