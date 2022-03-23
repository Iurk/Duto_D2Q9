#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:49:54 2020

@author: iurk
"""
import yaml
import numpy as np
import utilidades as utils

Sim_yaml = "../bin/dados.yml"
path_mesh = "../bin/Mesh/"
with open(Sim_yaml) as file:
    simulation = yaml.load(file, Loader=yaml.FullLoader)

Nx = simulation['domain']['Nx']
Ny = simulation['domain']['Ny']

walls = np.zeros((Ny, Nx), dtype=bool)
inlet = np.zeros_like(walls)
outlet = np.zeros_like(walls)

walls[0, :] = True
walls[Ny-1, :] = True

inlet[:, 0] = True
outlet[:, Nx-1] = True

walls = walls.flatten()
inlet = inlet.flatten()
outlet = outlet.flatten()

pasta = utils.criar_pasta("Mesh", main_root="../bin")

mesh_data = [walls, inlet, outlet]
mesh_files = ['walls.bin', 'inlet.bin', 'outlet.bin']

for data, filem in zip(mesh_data, mesh_files):
    file_path = path_mesh + filem
    with open(file_path, 'wb') as file:
        file.write(bytearray(data))

