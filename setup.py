import os
import glob

from distutils.core import setup
from distutils.extension import Extension
#from setuptools.extension import Extension
#from setuptools import find_packages

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

#packages = find_packages()
#package_data = dict((pkg, ['*.pxd']) for pkg in packages)
#
#extensions = []
#for package in packages:
#    subdir = package.replace(".", os.path.sep)
#    if len(glob.glob(os.path.join(subdir, '*.pyx'))) == 0: continue
#    extensions.append(
#            Extension(package + ".*",
#                [os.path.join(subdir, "*.pyx")],
#                include_dirs=[np.get_include()]
#                )


#print os.getcwd(), type(os.getcwd())

extensions = []
utils = Extension("phd.utils.*",
        ["phd/utils/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(utils)

containers = Extension("phd.containers.*",
        ["phd/containers/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(containers)

domain = Extension("phd.domain.*",
        ["phd/domain/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(domain)

hilbert = Extension("phd.hilbert.*",
        ["phd/hilbert/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(hilbert)

mesh = Extension("phd.mesh.*",
        ["phd/mesh/*.pyx", "phd/mesh/tess.cpp", "phd/mesh/tess3.cpp"],
        define_macros=[('CGAL_NDEBUG',1)],
        include_dirs=[np.get_include(), "/Users/Ricky/repo/moving-mesh/phd/boundary/"],
        libraries=["CGAL", "gmp"],
        language="c++"
        )
extensions.append(mesh)

boundary = Extension("phd.boundary.*",
        ["phd/boundary/*.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
        )
extensions.append(boundary)

load_balance = Extension("phd.load_balance.*",
        ["phd/load_balance/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(load_balance)

reconstruction = Extension("phd.reconstruction.*",
        ["phd/reconstruction/*.pyx"],
        include_dirs=[np.get_include(), "m", "/Users/Ricky/repo/moving-mesh/phd/mesh/", "/Users/Ricky/repo/moving-mesh/phd/boundary/"],
        language="c++"
        )
extensions.append(reconstruction)

riemann = Extension("phd.riemann.*",
        ["phd/riemann/*.pyx"],
        include_dirs=[np.get_include(), "m", "/Users/Ricky/repo/moving-mesh/phd/mesh/", "/Users/Ricky/repo/moving-mesh/phd/boundary/"],
        language="c++"
        )
extensions.append(riemann)

integrate = Extension("phd.integrate.*",
        ["phd/integrate/*.pyx"],
        include_dirs=[np.get_include(), "m", "/Users/Ricky/repo/moving-mesh/phd/mesh/", "/Users/Ricky/repo/moving-mesh/phd/boundary"],
        language="c++"
        )
extensions.append(integrate)

setup(
        name="phd",
        version="0.1",
        author="Ricardo Fernandez",
        license="MIT",
        cmdclass={'build_ext':build_ext},
        ext_modules=cythonize(extensions),
        packages=["phd", "phd.utils", "phd.containers", "phd.domain", "phd.reconstruction", "phd.riemann",
            "phd.hilbert", "phd.load_balance", "phd.boundary", "phd.mesh", "phd.integrate"],
        package_data={'phd.utils':['*.pxd'], 'phd.containers':['*.pxd'], 'phd.integrate':['*.pxd'],
            'phd.domain':['*.pxd'], 'phd.hilbert':['*.pxd'], 'phd.load_balance':['*.pxd'], 'phd.boundary':['*.pxd'],
            'phd.mesh':['*.pxd'], 'phd.load_balance':['*.pxd'],
            },
        )
