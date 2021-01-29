#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:27:54 2020

@author: iurk
"""
import yaml
import numpy as np
from os import walk
import utilidades as util
import multiprocessing as mp
import funcoes_graficos as fg
from time import time, sleep

def plotting(args):
    idx_file, rho, ux, uy = args
    u_mod = np.sqrt(ux**2 + uy**2)
    
    fg.grafico(u_mod, idx_file, pasta_img)
    
if __name__ == '__main__':

    ini = time()
    main = "./bin"
    fileyaml = "./bin/dados.yml"
    velocity = "Velocity"
    
    datafile = open(fileyaml)
    data = yaml.load(datafile, Loader=yaml.FullLoader)
    datafile.close()
    
    Nx = data['domain']['Nx']
    Ny = data['domain']['Ny']
    
    Steps = data['simulation']['NSTEPS']
    Saves = data['simulation']['NSAVE']
    digitos = len(str(Steps))
    
    idx_files = ["%0{}d".format(digitos) % i for i in range(0, Steps+Saves, Saves)]
    
    results = "./bin/Results/"
    pasta_img = util.criar_pasta('Images', folder=velocity, main_root=main)
    # pasta_stream = util.criar_pasta('Stream', folder=velocity, main_root=main)
    
    rho_files = []
    ux_files = []
    uy_files = []
    dic = {"rho": rho_files, "ux":ux_files, "uy":uy_files}
    
    for var in dic.keys():
        path = results + var
        
        for root, dirs, files in walk(path):
            for file in sorted(files):
                path_full = path + "/%s" % file
                dic[var].append(path_full)
                
    x = np.arange(1, Nx+1, 1)
    y = np.arange(1, Ny+1, 1)
    
    CPU = mp.cpu_count()
    pool = mp.Pool()
    
    idx = []
    rhos = np.empty((CPU, Ny, Nx))
    uxs = np.empty_like(rhos)
    uys = np.empty_like(rhos)
    
    i = 0
    print("Reading and plotting data...")
    while(i < len(idx_files)):
        for j in range(CPU):
    
            idx.append(idx_files[i])
            rhos[j] = np.fromfile(rho_files[i], dtype='float64').reshape(Ny, Nx)
            uxs[j] = np.fromfile(ux_files[i], dtype='float64').reshape(Ny, Nx)
            uys[j] = np.fromfile(uy_files[i], dtype='float64').reshape(Ny, Nx)
            
            i += 1
            if(i == len(idx_files)):
                break
        
        inputs = zip(idx, rhos, uxs, uys)
        pool.map(plotting, inputs)
        idx = []
        
    print('Animating...')
    fg.animation('Velocidade', './', pasta_img)
    # fg.animation('Stream', './', pasta_stream)
    print('Done!')
    fim = time()
    print("Finish in {} s".format(fim - ini))
