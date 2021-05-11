# from distutils.core import setup, Extension

from setuptools import Extension, setup
from Cython.Build import cythonize

NVCOMP_INCLUDE_DIR = './nvcomp/include'
NVCOMP_LIB_NAME = 'nvcomp'
NVCOMP_LIB_DIR = './nvcomp/lib'

CUDA_INCLUDE_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include'
CUDA_LIB_NAME = 'cudart'
CUDA_LIB_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64'

extensions = [Extension(
    'nvcomp',
    ['./src/nvcomp.pyx'],
    language='c++',
    include_dirs=[NVCOMP_INCLUDE_DIR, CUDA_INCLUDE_DIR],
    libraries=[NVCOMP_LIB_NAME, CUDA_LIB_NAME],
    library_dirs=[NVCOMP_LIB_DIR, CUDA_LIB_DIR],
)]

setup(name='nvcomp', ext_modules=cythonize(extensions, compiler_directives={'language_level' : '3'}))
