import numpy as np
import mpi4py as mpi
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("tree",["tree.pyx"],
            #language="c++",
            include_dirs=[np.get_include()])])
#setup(
#        cmdclass = {'build_ext': build_ext},
#        ext_modules = [Extension("test_none",["test_none.pyx"],
#            include_dirs=[np.get_include()])])
