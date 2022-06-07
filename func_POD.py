# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:28:08 2022

@author: raedb
"""
import numpy as np
import time
from func_FEM import *
from sol_analytical import *

def snapshot_matrix(t_int, Material, Mesh, load, analytical=True, display=True):
    '''
    Compute snapshot matrix for t (thickness) in t_int.

    Parameters
    ----------
    t_int : numpy array(1D), dtype('float64')
        Range of beam thicknesses.
    Material : MATERIAL()
        Contains material parameters.
    Mesh : MESH()
        Contains mesh parameters.
    load : dtype('float64')
        Point force magnitude.
    analytical: True/False, optional
        Control variable => compute analytical solution from beam theory
        The default is True.
    display : True/False, optional
        Display error. The default is True.

    Returns
    -------
    X : numpy array, dtype('float64')
        Snapshot matrix.

    '''
    f=load_vector(Mesh,load)
    K, cnst = k_global(Material,Mesh)
    K=K/cnst
    
    dx_norm=np.linalg.solve(K,f)
    
    X=np.zeros((K.shape[0],t_int.shape[0]))
    for i in range(t_int.shape[0]):
        t=t_int[i]
        Material.b=t
        Material.h=t
        Material.A=Material.b*Material.h
        Material.I=(Material.b*Material.h**3)/12 
        factor=Material.E*Material.I/Mesh.ell**3
        dx=dx_norm/factor
        if analytical==True:
            sigma=analytical_sol(Material, Mesh, load)
        X[:,i]=dx
        
        if display==True:
            print("__________________ solving for t = %.3f ________________" %t)
            dx, _=get_disp(dx, Mesh)
            print("Deflection at extremity from FEM:             %.3E"  %dx[-1])
            print("Deflection at extremity from beam theory:     %.3E"  %sigma[-1])
            print("L2 error with respect to analytical solution:  %.3E" %np.linalg.norm(dx-sigma,ord=2))
    return X
    
def dominant_modes(X, delta=0.9):
    '''
    

    Parameters
    ----------
    X : numpy array, dtype('float64')
        Snapshot matrix.
    delta : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    r : int
        Number of leading modes .

    '''
    u, s, vh=np.linalg.svd(X)
    r=np.min(np.where(np.cumsum(s)/np.cumsum(s)[-1]>=delta))
    return r

def PODModesSVD(X,r):
    '''
    

    Parameters
    ----------
    X : numpy array, dtype('float64')
        Snapshot matrix.
    r : int
        Number of leading modes .

    Returns
    -------
    numpy array dtype('float64')
        Orthornormal projection basis.

    '''
    u, s, vh=np.linalg.svd(X)
    return u[:,0:r+1]

def project_system(K,f,u):
    '''
    

    Parameters
    ----------
    K : (n,n)-Numpy array, dtype('float64')
        Global stiffness matrix of clamped beam.
    f : (n,1)-Numpy array, dtype('float64')
        Load vector.
    u : (n,1)-Numpy array, dtype('float64')
        Solutions of system containing displacements and rotations (angles).

    Returns
    -------
    K_red : (r,r)-Numpy array, dtype('float64')
        Reduced stiffness matrix of clamped beam.
    f_red : (r,1)-Numpy array, dtype('float64')
        Reduced vector.

    '''
    #cnst=Material.E*Material.I/Mesh.ell**3
    #K=K/cnst
    
    K_red=u.T@K@u
    f_red=u.T@f
    
    return K_red, f_red

def solve_red_system(Material,Mesh,f,u,t):
    '''
    

    Parameters
    ----------
    Material : TYPE
        DESCRIPTION.
    Mesh : TYPE
        DESCRIPTION.
    f : (n,1)-Numpy array, dtype('float64')
        Load vector.
    u : (n,1)-Numpy array, dtype('float64')
        Solutions of system containing displacements and rotations (angles).
    t : dtype('float64')
        Beam thickness.

    Returns
    -------
    x : numpy array dtype('float64')
        Solution of reduced system (Back projected into original space).

    '''
    Material.b=t
    Material.h=t
    Material.A=Material.b*Material.h
    Material.I=(Material.b*Material.h**3)/12 
    K,_=k_global(Material, Mesh)
    K_red, f_red=project_system(K, f, u)
    x_red=np.linalg.solve(K_red, f_red)
    x=u@x_red
    return x    

def PODModesCorrelation(X,r):
    '''
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    w, v=np.linalg.eigh(X@X.T)
    return v[:,0:r+1]

def PODModesSnapshots(X,r):
    '''
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    w, v=np.linalg.eigh(X.T@X)
    u= np.expand_dims(X@v[:,0:r+1]@np.reciprocal(w[0:r+1]), axis=1)
    return u