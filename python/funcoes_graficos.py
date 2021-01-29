#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:31:44 2019

@author: iurk
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio

def grafico(u, step, pasta_imagens):
    file = 'vel.' + str(step) + '.png'
    path = pasta_imagens + '/%s' % file
    plt.clf()
    plt.imshow(u, cmap=cm.RdBu, interpolation='nearest')
    plt.colorbar()
    plt.savefig(path, dpi=250)
    
def stream(x, y, u, v, u_mod, step, pasta_imagens):
    file = 'vel.' + str(step) + '.png'
    path = pasta_imagens + '/%s' % file
    
    lw = u_mod/u_mod.max()
    
    plt.clf()
    plt.streamplot(x, y, u, v, density=[0.5, 1], linewidth=lw,
                   color=u_mod, cmap=cm.YlOrRd)
    plt.colorbar(cmap=cm.YlOrRd)
    plt.savefig(path, dpi=250)
    
def animation(nome, pasta, pasta_imagens):
    from os import listdir
    
    file = nome + '.gif'
    file_2 = nome + '.mp4'

    path = pasta + '/%s' % file
    path_2 = pasta + '/%s' % file_2
    
    files_imgs = [im for im in listdir(pasta_imagens) if im.endswith('.png')]
    files_imgs.sort(key=__ordenar)
    
    images = [imageio.imread(pasta_imagens + '/%s' % file) for file in files_imgs]
    imageio.mimsave(path, images)
    imageio.mimsave(path_2, images)
    
def __ordenar(item):
    return int(item.split('.')[1])
    
        
