from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('CR3BP',
        ['CR3BP.pyx', 'c_CR3BP.c'],
        include_dirs=['.', numpy.get_include()])
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level='3',
        compiler_directives={'embedsignature': True}
    )
)