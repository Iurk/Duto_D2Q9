#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:05:49 2020

@author: iurk
"""

def criar_pasta(name, main_root='./'):
    from shutil import rmtree
    from os import walk, listdir, mkdir, getcwd
    from os.path import isdir, join
    
    onlyfolders = [dirs for root, dirs, files in walk(main_root)][0]
    path = main_root + "/" + name
    
    if name not in onlyfolders:
        mkdir(path)
        return path
    
    else:
        return path