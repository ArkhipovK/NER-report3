#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for homeadminkprojectsner_report3.

    This file was generated with PyScaffold 2.5.11, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys
from setuptools import setup
import ner_report3

print(ner_report3.__version__)
def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(name='report3',
          version=ner_report3.__version__,
          setup_requires=['six'] + sphinx,
          use_pyscaffold=True,
          install_requires=[
            'wikiextractor',
            'numpy>=1.15.2',
            'pandas>=0.23.4',
            'nltk==3.3',
            'keras>=2.2.4',
            "tensorflow>=1.11.0",
            'sklearn',
            'keras_contrib'
          ],
          dependency_links=[
            "https://github.com/attardi/wikiextractor.git#egg=wikiextractor",
            "git+https://www.github.com/keras-team/keras-contrib.git#egg=keras_contrib"]
          )


if __name__ == "__main__":
    setup_package()
