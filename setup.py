from setuptools import setup, find_packages

from vaenas.__version__ import __version__

setup(
    name='VAENAS',
    version=__version__,

    url='https://github.com/fpavesi11/NASVAE',
    author='Federico Pavesi',
    author_email='f.pavesi11@campus.unimib.it',

    packages=find_packages(include=['vaenas', 
                                    'vaenas.*',
                                    'vaenas.decoders',
                                    'vaenas.flowVAE',
                                    'vaenas.IAF',
                                    'vaenas.VanillaVAE']),
)