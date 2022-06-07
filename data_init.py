# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:29:22 2022

@author: raedb
"""
class MATERIAL:
    '''
    Define material properties
    
    Parameters:
    ----------
    E:              Young's modulus
    Nu:             Poisson ratio
    rho:            Density
    L:              Total length
    b:              Beam depth
    h:              Beam height
    A:              Beam cross sectional area
    I:              Beam moment of inertia
    mass:           Total mass
    Irot:           Rotary inertia
    '''
    E=None 
    Nu=None
    rho=None 
    L=None 
    b=None 
    h=None 
    A=None 
    I=None 
    mass=None 
    Irot=None 
        
    def __init__(self):
        pass
    
class MESH:
    '''
    Define mesh properties
    
    Parameters:
    ----------  
    N:              Number of elements 
    Nnodes:         Number of nodes
    ell:            Length of beam element
    Nmodes:         Number of modes to keep
    L:              Total length
    x:              Grid points/ spatial mesh
    '''
    N=None #number of elements 
    Nnodes=None #number of nodes
    ell=None #length of beam element
    Nmodes=None #number of modes to keep
    L=None #total length
    x=None # np.arange(0,ell+L,ell)
    
    def __init__(self):
        pass