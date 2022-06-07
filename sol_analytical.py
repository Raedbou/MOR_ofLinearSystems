# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:43:26 2022

@author: raedb
"""
import numpy as np

def analytical_sol(Material, Mesh, load):
    '''
    Calculate analytical solution of maximal displacement (at x=L) 0f 
    a cantilever beam.

    Parameters
    ----------
    Material : MATERIAL()
        Contains material parameters.
    Mesh : MESH()
        Contains mesh parameters.
    load : dtype('float64')
        Point force magnitude.

    Returns
    -------
    sigma : dtype('float64')
        Maximum displacement.

    '''
    sigma=((load*np.power(Mesh.x,2))/(6*Material.I*Material.E))*(3*Material.L-Mesh.x)
    return sigma