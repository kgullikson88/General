from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("FittingUtilities2", ["FittingUtilities.pyx"], include_dirs=[numpy.get_include()]), Extension("RotBroad_Fast", ["RotBroad2.pyx"], include_dirs=[numpy.get_include()]), Extension("MakeModel_Fast", ["MakeModel.pyx"], include_dirs=[numpy.get_include()])]
    )