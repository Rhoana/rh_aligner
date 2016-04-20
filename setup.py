#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import subprocess
import numpy as np

VERSION = '0.0.1'

README = open('README.md').read()

flags = subprocess.check_output(['pkg-config', '--cflags-only-I', 'opencv'])
include_dirs_list = [flag[2:] for flag in flags.split()]
include_dirs_list.append('.')
flags = subprocess.check_output(['pkg-config', '--libs-only-L', 'opencv'])
library_dirs_list = flags
flags = subprocess.check_output(['pkg-config', '--libs', 'opencv'])
libraries_list = []
for flag in flags.split():
    libraries_list.append(flag)

EXTENSIONS = [
        Extension(
                  "rh_aligner/common/cv_wrap_module",
                  ["rh_aligner/common/cv_wrap_module.pyx", "rh_aligner/common/cv_wrap.cpp"],
                  language="c++",
                  include_dirs=include_dirs_list,
                  extra_compile_args=['-O3', '--verbose'],
                  extra_objects=libraries_list
                 ),
        Extension(
                  "rh_aligner/alignment/mesh_derivs_multibeam",
                  ["rh_aligner/alignment/mesh_derivs_multibeam.pyx"],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-fopenmp', '-O3', '--verbose'],
                  extra_link_args=['-fopenmp']
                 )
]

setup(
    name='rh_aligner',
    version=VERSION,
    packages=find_packages(exclude=['*.common']),
    ext_modules = cythonize(EXTENSIONS),
    author='Adi Suissa-Peleg',
    author_email='adisuis@seas.harvard.edu',
    url="https://github.com/Rhoana/rh_aligner",
    description="Rhoana's 2D and 3D alignment tool",
    long_description=README,
    include_package_data=True,
    install_requires=[
        "numpy>=1.9.3",
        "scipy>=0.16.0",
        "argparse>=1.4.0",
        "h5py>=2.5.0",
        "progressbar>=2.3",
        "Cython>=0.23.3",
    ],
    dependency_links = ['http://github.com/Rhoana/rh_renderer/tarball/master#egg=rh_renderer-0.0.1'],
    zip_safe=False
)
