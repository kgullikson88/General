from setuptools import setup, Extension
import numpy as np
from Cython.Distutils import build_ext

requires = ['TelFit',
            'h5py',
            'pandas',
            'george',
            'emcee',
            'scikit-learn',
            'lmfit',
            'scikit-monaco', 
            'statsmodels',
            'triangle-plot',
            'pymultinest',
            'seaborn',
            'astroquery',
            ]

optional_requires = ['bokeh', 'astropysics',
                     'json', 'pyraf', 'mlpy',
                     'anfft']

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("RotBroad_Fast", ["RotBroad2.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-O3"])],
    install_requires=requires,
    extras_require={'Extra stuff': optional_requires},

)
