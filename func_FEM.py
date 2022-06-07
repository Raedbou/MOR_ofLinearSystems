# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:42:21 2022

@author: raedb
"""
import numpy as np

def k_elem(Material, Mesh):
    '''
    Constrcut beam element stiffnes matrix

    Parameters
    ----------
    Material : MATERIAL()
        Contains material parameters.
    Mesh : MESH()
        Contains mesh parameters.

    Returns
    -------
    K_e : (4,4)-Numpy array, dtype('float64')
        Element stiffness matrix.
    factor : dtype('float64')
        Factor E*I/ell**3, comes in front of stiffness matrix. 
        pre-calculated for computational efficiency
    '''
    E=Material.E
    I=Material.I
    ell=Mesh.ell
    
    factor=E*I/ell**3
    
    K_e=factor*np.array([[12, 6*ell, -12, 6*ell], #x1
                                [6*ell, 4*ell**2, -6*ell, 2*ell**2], #theta1
                                [-12, -6*ell, 12, -6*ell], #x2
                                [6*ell, 2*ell**2, -6*ell, 4*ell**2]]) #theta2
    return K_e, factor

def k_global(Material,Mesh):  
    '''
    Construct global stiffness matrix from beam element stiffness matrices.
    Global stiffnes matrix of cantilever beam, with clamped end at node 0.
    Therefore first and last element removed

    Parameters
    ----------
    Material : MATERIAL()
        Contains material parameters.
    Mesh : MESH()
        Contains mesh parameters.

    Returns
    -------
    K : (n,n)-Numpy array, dtype('float64')
        Global stiffness matrix of clamped beam.
    factor : dtype('float64')
        Factor E*I/ell**3, comes in front of stiffness matrix. 
        pre-calculated for computational efficiency

    '''
    K_e, factor=k_elem(Material, Mesh)
    K=np.zeros((2*(Mesh.N+1),2*(Mesh.N+1))) 
    
    for i in range(1,Mesh.N+1):
        K[(2*(i-1)+1-1):(2*(i-1)+4), 
          (2*(i-1)+1-1):(2*(i-1)+4)]=K[(2*(i-1)+1-1):(2*(i-1)+4)][:,(2*(i-1)+1-1):(2*(i-1)+4)]+K_e
        
    #cantilever beam BCs
    K=np.delete(K,[0,1],0) #K[1:2,:]=[] 
    K=np.delete(K,[0,1],1) #K[:,1:2]=[] 
    
    return K, factor
        
def load_vector(Mesh, load):
    '''
    Construct load vector in the case of the cantilever beam.
    A point force of magnitude 'load' is applied at node N (extremity) of beam

    Parameters
    ----------
    Mesh : MESH()
        Contains mesh parameters.
    load : dtype('float64')
        Point force magnitude.

    Returns
    -------
    f : (n,1)-Numpy array, dtype('float64')
        Load vector.

    '''
    f=np.zeros((2*Mesh.Nnodes,1))
    f[2*Mesh.Nnodes-2]=load
    f=np.delete(f,[0,1]) #f[1:2]=[]     
    return f

def solve_FEM_system(Material, Mesh, load):
    '''
    Solve linear system (Full order) of FEM of cantilever beam.

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
    u : (n,1)-Numpy array, dtype('float64')
        Solutions of system containing displacements and rotations (angles).

    '''
    f=load_vector(Mesh,load)
    K, _=k_global(Material, Mesh)
    u=np.linalg.solve(K,f)
    return u

def get_disp(dx_vec,Mesh):
    '''
    Split solution vector into displacements and rotations.

    Parameters
    ----------
    dx_vec : (n,1)-Numpy array, dtype('float64')
        Solutions of system containing displacements and rotations (angles).
    Mesh : MESH()
        Contains mesh parameters.

    Returns
    -------
    dx : (n/2,1)-Numpy array, dtype('float64')
        Displacement.
    dtheta : (n/2,1)-Numpy array, dtype('float64')
        Rotations.

    '''
    dx=np.hstack([0., dx_vec[0:2*Mesh.Nnodes-2:2]])
    dtheta=np.hstack([0., dx_vec[1:2*Mesh.Nnodes-1:2]])
    return dx, dtheta