#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:23:54 2019

@author: Darshan
"""

import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)
logging.warning('This is a Warning')
logging.debug('This will get logged')
logging.info('Admin logged in')
name = 'John'

logging.error('%s raised an error', name)
logging.error(f'{name} raised an error')
logging.info(f'{name} raised an error')    
