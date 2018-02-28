
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

from setuptools import Extension

if os.name == 'nt':
    fopenmp_arg= "/openmp"
else:
    fopenmp_arg = "-fopenmp"


setup(ext_modules = cythonize(Extension(
           "hellocython",                                         
           sources=["hellocython.pyx", "SinglePredict.cpp", "SingleUpdate.cpp"],   
           include_dirs=[numpy.get_include(), '.'],                                                               
           language="c++",
           extra_compile_args = [fopenmp_arg],
           extra_link_args=[])
      ))


