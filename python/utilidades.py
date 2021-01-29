#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:05:49 2020

@author: iurk
"""

def criar_pasta(name, folder="", main_root='./'):
    from shutil import rmtree
    from os import walk, listdir, mkdir, getcwd
    from os.path import isdir, join
    
    onlyfolders = [dirs for root, dirs, files in walk(main_root)][0]
    
    if folder and folder not in onlyfolders:
        path = main_root + "/" + folder
        mkdir(path)
        path = criar_pasta(name, folder=folder, main_root=main_root)
        return path
    
    elif folder:
        onlyfolders = [dirs for root, dirs, files in walk(main_root)][5]    
        path = main_root + "/" + folder + "/" + name
        if name not in onlyfolders:
            mkdir(path)
        
        return path
    
    elif not folder and name not in onlyfolders:
        path = main_root + "/" + name
        mkdir(path)
        return path
    
    else:
        for root, dirs, files in walk(main_root):
            if name in root:
                return root
        
        
                