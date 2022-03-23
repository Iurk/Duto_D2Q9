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
from time import time
import multiprocessing as mp
import funcoes_graficos as fg
from functools import partial
from utilidades import criar_pasta

def getting_files():
    
    rho = []
    ux = []
    uy = []
    dic = {"rho": rho, "ux":ux, "uy":uy}
    
    for var in dic.keys():
        path = results + var
        
        for root, dirs, files in walk(path):
            for file in sorted(files):
                path_full = path + "/%s" % file
                dic[var].append(path_full)
                
    return rho, ux, uy

def getting_perfil_folders(Points, directory):
    folders = []
    for i in Points:
        name = 'Nx = {}'.format(i)
        folders.append(criar_pasta(name, main_root=directory))
    return folders

def plotting_module(module_vel, args):
    idx_file, ux, uy = args
    u_mod = np.sqrt(ux**2 + uy**2)
    fg.image(u_mod, idx_file, module_vel)
    
def plotting_perfil(Point, pasta_perfil, args):
    idx_file, x, y = args
    fg.grafico(x[:, Point], y, idx_file, pasta_perfil)
    
def reading_plotting(func, mode, point=None, folder=None):
    CPU = mp.cpu_count()
    
    idx = []
    ys = np.empty((CPU, Ny))
    rhos = np.empty((CPU, Ny, Nx))
    uxs = np.empty_like(rhos)
    uys = np.empty_like(rhos)

    pool = mp.Pool()
    pattern = r'\d{%d}' % len(str(Steps))
    
    i = 0
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
        
        if mode == 'Velocity Module':
            inputs = zip(idx, uxs, uys)
        elif mode == 'Velocity Perfil':
            inputs = zip(idx, uxs, ys)
        elif mode == 'Rho Perfil':
            inputs = zip(idx, rhos, ys)
        
        if(point == None):
            func_partial = partial(func, module_vel)
        else:
            func_partial = partial(func, point, folder)
            
        pool.map(func_partial, inputs)
        idx = []
    
if __name__ == '__main__':

    ini = time()
    main = "./bin"
    results = "./bin/Results/"
    fileyaml = "./bin/dados.yml"
    velocity = "Velocity"
    pressure = "Pressure"
    
    with open(fileyaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    Nx = data['domain']['Nx']
    Ny = data['domain']['Ny']
    Steps = data['simulation']['NSTEPS']
    Saves = data['simulation']['NSAVE']
    plot_module = data['output']['plot_module']
    plot_profile = data['output']['plot_profiles']
    digitos = len(str(Steps))
    
    pressure = criar_pasta(pressure, main_root=main)
    velocity = criar_pasta(velocity, main_root=main)
    module_vel = criar_pasta('Module', main_root=velocity)
    perfil_vel = criar_pasta('Perfil', main_root=velocity)
    perfil_rho = criar_pasta('Perfil', main_root=pressure)
    
    Points_rho = [0, int(Nx-1)]
    Points_vel = [0, int((Nx-1)/4), int((Nx-1)/2), int(3*(Nx-1)/4), int(Nx-1)]
    
    rho_files, ux_files, uy_files = getting_files()
    
    if plot_module:
        Rho_Points_folder = getting_perfil_folders(Points_rho, perfil_rho)
        Velocity_Points_folders = getting_perfil_folders(Points_vel, perfil_vel)
    
        print('Poltting Velocity Module...')
        reading_plotting(plotting_module, 'Velocity Module')
        
    if plot_profile:
        print('Plotting Rho values at inlet and outlet...')
        for i in range(len(Points_rho)):
            Point = Points_rho[i]
            pasta_perfil = Rho_Points_folder[i]
            
            reading_plotting(plotting_perfil, 'Rho Perfil', point=Point, folder=pasta_perfil)
    
        print('Plotting Velocity Perfil at specified points...')
        for i in range(len(Points_vel)):
            Point = Points_vel[i]
            pasta_perfil = Velocity_Points_folders[i]
            
            reading_plotting(plotting_perfil, 'Velocity Perfil', point=Point, folder=pasta_perfil)
    
    
    print('Animating...')
    if plot_module:
        name = 'Velocity_Module'
        fg.animation(name, main, module_vel)
        
    if plot_profile:
        for folder in Rho_Points_folder:
            name = 'Rho_perfil_{}'.format(folder.split('/')[-1])
            fg.animation(name, main, folder)
            
        for folder in Velocity_Points_folders:
            name = 'Velocity_perfil_{}'.format(folder.split('/')[-1])
            fg.animation(name, main, folder)
        
    print('Done!')
    fim = time()
    print("Finish in {} s".format(fim - ini))
