#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:27:54 2020

@author: iurk
"""
import re
import yaml
import numpy as np
from os import walk
import utilidades as util
import multiprocessing as mp
import funcoes_graficos as fg
from time import time

def plotting_image(args):
    idx_file, rho, ux, uy = args
    u_mod = np.sqrt(ux**2 + uy**2)
    
    fg.image(u_mod, idx_file, pasta_img)
    
def plotting_perfil(args):
    idx_file, ux, y = args
    
    fg.grafico(ux[:, Point], y, idx_file, pasta_perfil)
    
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
    Point = int(Nx/2)
    
    Steps = data['simulation']['NSTEPS']
    Saves = data['simulation']['NSAVE']
    digitos = len(str(Steps))
    
    results = "./bin/Results/"
    pasta_img = util.criar_pasta('Images', folder=velocity, main_root=main)
    pasta_perfil = util.criar_pasta('Perfil', folder=velocity, main_root=main)
    
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
                
    # x = np.arange(1, Nx+1, 1)

    CPU = mp.cpu_count()
    pool = mp.Pool()
    
    idx = []
    ys = np.empty((CPU, Ny))
    rhos = np.empty((CPU, Ny, Nx))
    uxs = np.empty_like(rhos)
    uys = np.empty_like(rhos)
    
    i = 0
    pattern = r'\d{%d}' % len(str(Steps))
    
    print("Reading and plotting data...")
    while(i < len(rho_files)):
        for j in range(CPU):
            idx.append(re.search(pattern, rho_files[i]).group(0))
            rhos[j] = np.fromfile(rho_files[i], dtype='float64').reshape(Ny, Nx)
            uxs[j] = np.fromfile(ux_files[i], dtype='float64').reshape(Ny, Nx)
            uys[j] = np.fromfile(uy_files[i], dtype='float64').reshape(Ny, Nx)
            ys[j] = np.arange(1, Ny+1, 1)
            
            i += 1
            if(i == len(rho_files)):
                break
        
        image_inputs = zip(idx, rhos, uxs, uys)
        perfil_inputs = zip(idx, uxs, ys)
        
        pool.map(plotting_image, image_inputs)
        pool.map(plotting_perfil, perfil_inputs)
        idx = []
        
    print('Animating...')
    fg.animation('Velocidade', main, pasta_img)
    fg.animation('Perfil', main, pasta_perfil)
    print('Done!')
    fim = time()
    print("Finish in {} s".format(fim - ini))
