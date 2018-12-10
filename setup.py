#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for homeadminkprojectsner_report3.

    This file was generated with PyScaffold 2.5.11, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import os
import sys
import configparser
from setuptools import setup, find_packages
import ner_report3


print(ner_report3.__version__)

def config_setup():
    __config__ = configparser.ConfigParser()
    __config__.read_file(open('config.ini'))
    __config__['GLOBAL']['storepath'] = os.path.dirname(os.path.abspath(__file__))
    with open('config.ini', 'w') as configfile:
        __config__.write(configfile)




def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(name='report3',
          version=ner_report3.__version__,
          packages=find_packages(),
          entry_points = {
              'console_scripts': [
                  'report3 = ner_report3.__main__:main'
              ]
          },
          setup_requires=['six'] + sphinx,
          use_pyscaffold=True,
          include_package_data=True,
          install_requires=[
            'wikiextractor',
            'numpy>=1.15.2',
            'pandas>=0.23.4',
            'nltk==3.3',
            'keras>=2.2.4',
            "tensorflow>=1.11.0",
            'sklearn',
            'keras_contrib',
            'seqeval',
            'pydot',
            'GraphViz'
          ],
          dependency_links=[
            "git+https://github.com/attardi/wikiextractor.git#egg=wikiextractor",
            "git+https://www.github.com/keras-team/keras-contrib.git#egg=keras_contrib"
          ],
          data_files=[('ner_report3',['config.ini'])]
          )


if __name__ == "__main__":
    config_setup()
    setup_package()

