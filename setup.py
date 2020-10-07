
from setuptools import find_packages
from setuptools import setup


setup(
    name='fridadrp',
    version='0.1.dev0',
    author='Sergio Pascual',
    author_email='sergiopr@fis.ucm.es',
    url='https://github.com/guaix-ucm/fridadrp',
    license='GPLv3',
    description='FRIDA Data Procesing Pipeline',
    packages=find_packages(),
    package_data={
        'fridadrp': [
            'drp.yaml',
        ],
        'fridadrp.instrument.configs': ['*.json']
    },
    install_requires=[
        'setuptools>=36.2.1',
        'six',
        'numpy',
        'matplotlib',
        'astropy >= 2',
        'scipy',
        'numina >= 0.22',
        'enum34;python_version<"3.4"',
        'contextlib2;python_version<"3.5"',
    ],
    extras_require={
        'test': [
            'pytest<3.7; python_version<"3"',
            'pytest>=6; python_version>="3"',
            'pytest-remotedata'
            ]
    },
    entry_points={
        'numina.pipeline.1': [
            'FRIDA = fridadrp.loader:drp_load',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Development Status :: 3 - Alpha',
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        ],
    long_description=open('README.rst').read()
)
