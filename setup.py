from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("FittingUtilities", ["FittingUtilities.pyx"], include_dirs=[numpy.get_include()]),
                   Extension("RotBroad_Fast", ["RotBroad2.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
                   Extension("MakeModel_Fast", ["MakeModel.pyx"], include_dirs=[numpy.get_include()]),
                   Extension("fast_cubic_spline", ["fast_cubic_spline.pyx"], include_dirs=[numpy.get_include()])]
    )
