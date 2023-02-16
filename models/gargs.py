# -*- coding: utf-8 -*-
from collections import OrderedDict
def _init_global():
    global _args
    _args = OrderedDict()

def _set_value(key, value):
    _args[key] = value

def _get_value(key):
    try:
        return _args[key]
    except:
        print('error!')
        exit(-1)
