
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

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
        ["phd/mesh/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(mesh)

boundary = Extension("phd.boundary.*",
        ["phd/boundary/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(boundary)

load_balance = Extension("phd.load_balance.*",
        ["phd/load_balance/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(load_balance)

reconstruction = Extension("phd.reconstruction.*",
        ["phd/reconstruction/*.pyx"],
        include_dirs=[np.get_include(), "m"]
        )
extensions.append(reconstruction)

riemann = Extension("phd.riemann.*",
        ["phd/riemann/*.pyx"],
        include_dirs=[np.get_include(), "m"]
        )
extensions.append(riemann)

integrate = Extension("phd.integrate.*",
        ["phd/integrate/*.pyx"],
        include_dirs=[np.get_include(), "m"]
        )
extensions.append(integrate)

#ngb = Extension("phd.ngb.*",
#        ["phd/ngb/*.pyx"],
#        include_dirs=[np.get_include()]
#        )
#extensions.append(ngb)

setup(
        name="phd",
        version="0.1",
        author="Ricardo Fernandez",
        license="MIT",
        cmdclass={'build_ext':build_ext},
        ext_modules=cythonize(extensions),
        packages=["phd", "phd.utils", "phd.containers", "phd.domain", "phd.reconstruction",
            "phd.riemann", "phd.integrate", "phd.hilbert", "phd.mesh", "phd.load_balance", "phd.boundary"],
            #"phd.riemann", "phd.integrate", "phd.hilbert", "phd.mesh", "phd.load_balance", "phd.boundary", "phd.ngb"],
        package_data={'phd.utils':['*.pxd'], 'phd.containers':['*.pxd'],
            'phd.domain':['*.pxd'], 'phd.reconstruction':['*.pxd'], 'phd.riemann':['*.pxd'],
            'phd.integrate':['*.pxd'], 'phd.hilbert':['*.pxd'], 'phd.mesh':['*.pxd'],
            'phd.load_balance':['*.pxd'], 'phd.boundary':['*.pxd'],
            #'phd.load_balance':['*.pxd'], 'phd.boundary':['*.pxd'], 'phd.ngb':['*.pxd'],
            },
        )
