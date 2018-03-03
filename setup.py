
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

from setuptools import Extension

if os.name == 'nt':
    compile_args_OS = "/openmp"
else:
    compile_args_OS = ["-O3", "-fopenmp", "-ffast-math", "-mavx"]



setup(ext_modules = [
        Extension(
            "hello9",
            sources = ["hellocython2.pyx", "SingleUpdate.c", "SinglePredict.c"],
            libraries= [],
            include_dirs=[numpy.get_include(), '.'],
            extra_compile_args = compile_args_OS
            )]
      )


