# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:44:19 2022

@author: raedb
"""
import numpy as np
import time
from func_FEM import *
from sol_analytical import *
from func_POD import *

def cpu_time(Material, Mesh, N_elem, t_int, load, r, method):
    '''
    

    Parameters
    ----------
    Material : TYPE
        DESCRIPTION.
    Mesh : TYPE
        DESCRIPTION.
    N_elem : TYPE
        DESCRIPTION.
    t_int : TYPE
        DESCRIPTION.
    load : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.

    Returns
    -------
    t : TYPE
        DESCRIPTION.
    e : TYPE
        DESCRIPTION.

    '''
    t=np.zeros(N_elem.shape)
    e=np.zeros(N_elem.shape)
    for idx in range(N_elem.shape[0]):
        Mesh.N=N_elem[idx]
        Mesh.Nnodes=Mesh.N+1 
        Mesh.ell=Mesh.L/Mesh.N 
        Mesh.x=np.arange(0,Mesh.ell+Mesh.L,Mesh.ell)
        f= load_vector(Mesh, load)
        print('MOR using the snapshot method for N=', Mesh.N)
        X=snapshot_matrix(t_int, Material, Mesh, load, 
                          False, False)
        t0=time.time()
        if method=='svd':
            u=PODModesSVD(X,r)            
        elif method=='corr':
            u=PODModesCorrelation(X,r)            
        elif method=='snapshots':
            u=PODModesSnapshots(X,r)
        disp=solve_red_system(Material,Mesh,f,u,0.1)
        dx, dtheta=get_disp(disp, Mesh)
        sigma=analytical_sol(Material, Mesh, load)
        t[idx]=time.time()-t0
        e[idx]=np.linalg.norm((dx-sigma),ord=2)/Mesh.N
    return t, e  