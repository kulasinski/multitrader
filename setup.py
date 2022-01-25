#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Multitrader - backtesting engine for multiple tickers at once
# https://github.com/kulasinski/multitrader

from setuptools import setup
import io
from os import path

# --- get version ---
version = "unknown"
with open("multitrader/version.py") as f:
    line = f.read().strip()
    version = line.replace("version = ", "").replace('"', '')
# --- /get version ---


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='yfinance',
    version=version,
    description='Backtesting engine for multiple tickers at once',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kulasinski/multitrader',
    author='Karol Kulasinski',
    author_email='physica.solutions@gmail.com',
    license='Apache',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',


        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    platforms=['any'],
    keywords='pandas, yahoo finance',
    # packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
)