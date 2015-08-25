
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

particles = Extension("phd.particles.*",
        ["phd/particles/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(particles)

domain = Extension("phd.domain.*",
        ["phd/domain/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(domain)

load_balance = Extension("phd.load_balance.*",
        ["phd/load_balance/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(load_balance)

reconstruction = Extension("phd.reconstruction.*",
        ["phd/reconstruction/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(reconstruction)

fluxes = Extension("phd.fluxes.*",
        ["phd/fluxes/*.pyx"],
        include_dirs=[np.get_include()]
        )
extensions.append(fluxes)

setup(
        name="phd",
        version="0.1",
        author="Ricardo Fernandez",
        license="MIT",
        cmdclass={'build_ext':build_ext},
        ext_modules=cythonize(extensions),
        packages=["phd", "phd.utils", "phd.particles", "phd.doamin", "phd.load_balance"],
        package_data={'phd.utils':['*.pxd'], 'phd.particles':['*.pxd'],
            'phd.domain':['*.pxd'], 'phd.load_balance':['*.pxd'],
            'phd.reconstruction':['*.pxd'], 'phd.fluxes':['*.pxd']
            },
        )
#setup(
#        #cmdclass={'build_ext':build_ext},
#        ext_modules=cythonize([
#            'phd/utils/carray.pyx'#,
#            #'phd/particles/particle_array.pyx'
#            ],
#            include_path=[np.get_include()]
#            )
#        )
