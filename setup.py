import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

subdirs = [
        "phd/utils/",
        "phd/containers/",
        "phd/domain/",
        "phd/hilbert/",
        "phd/mesh/",
        "phd/boundary/",
        "phd/load_balance/",
        "phd/reconstruction/",
        "phd/riemann/",
        "phd/integrate/",
]

cpp = ("mesh", "boundary", "reconstruction", "riemann", "integrate")

extensions = []
for subdir in subdirs:
    extensions.append(
            Extension(subdir.replace("/", ".") + ".*",
                [os.path.join(subdir, "*.pyx")],
                include_dirs = [np.get_include()] + subdirs,
            )
    )
    if any(_ in subdir for _ in cpp):
        extensions[-1].language = "c++"

setup(
        name="phd",
        version="0.1",
        author="Ricardo Fernandez",
        license="MIT",
        cmdclass={'build_ext':build_ext},
        ext_modules=cythonize(extensions),
        packages=["phd", "phd.utils", "phd.containers", "phd.domain", "phd.hilbert", "phd.load_balance", "phd.boundary",
             "phd.mesh"],
        package_data={'':['*.pxd']},
)
