# -*- coding: utf-8 -*-
import pkg_resources
import configparser
import io

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = '1.0.0'
    
__config__ = configparser.ConfigParser(allow_no_value=True)
__config__.read_file(open('config.ini'))