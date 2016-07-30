"""aura library
See: https://github.com/sschaetz/aura

File modified from: https://github.com/pypa/sampleproject/blob/master/setup.py
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path, pathsep

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aura',

    version='0.1.0',

    description='C++ and Python accelerator programming',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/sschaetz/aura',

    # Author details
    author='Sebastian Schaetz',
    author_email='seb.schaetz@gmail.com',

    license='Boost',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',
        'Intended Audience:: Science / Research',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Hardware',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Code Generators',

        'License :: OSI Approved :: Boost',

        'Programming Language :: Python :: 3.4',
    ],

    keywords='gpgpu',

    packages=['python' + pathsep + 'aura'],

    install_requires=[],

    extras_require={},

    package_data={},

    data_files=[],

)