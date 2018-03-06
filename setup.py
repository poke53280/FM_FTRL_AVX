
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

from setuptools import Extension

if os.name == 'nt':
    compile_args_OS = ["/openmp", "/Ox"]
    link_args_OS = []
else:
    compile_args_OS = ["-O3", "-fopenmp", "-ffast-math", "-mavx"]
    link_args_OS = ["-fopenmp"] 


setup(ext_modules = [
        Extension(
            "hello9",
            sources = ["hellocython2.pyx", "avx_ext.c"],
            libraries= [],
            include_dirs=[numpy.get_include(), '.'],
            extra_compile_args = compile_args_OS,
            extra_link_args= link_args_OS
            )]
      )


