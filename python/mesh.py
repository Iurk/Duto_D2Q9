#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:49:54 2020

@author: iurk
"""
import yaml
import numpy as np
import utilidades as utils

Sim_yaml = "./bin/dados.yml"
with open(Sim_yaml) as file:
    simulation = yaml.load(file, Loader=yaml.FullLoader)

Nx = simulation['domain']['Nx']
Ny = simulation['domain']['Ny']

solid = np.zeros((Ny, Nx), dtype=bool)
solid[0,:] = True
solid[Ny-1,:] = True

solid = solid.flatten()

pasta = utils.criar_pasta("Mesh", main_root="./bin")

name_solid = "mesh.bin"

solid_path = pasta + "/%s" % name_solid
with open(solid_path, 'wb') as file:
    file.write(bytearray(solid))


