
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

from setuptools import Extension

if os.name == 'nt':
    fopenmp_arg= "/openmp"
else:
    fopenmp_arg = "-fopenmp"


setup(ext_modules = [
        Extension(
            "hello9",
            sources = ["hellocython2.pyx", "SingleUpdate.c", "SinglePredict.c"],
            libraries= [],
            include_dirs=[numpy.get_include(), '.'],
            extra_compile_args = [fopenmp_arg]
            )]
      )


