# -*- coding: utf-8 -*-
"""
Latest modification on Fri Jun  6 20:24:32 2022

@author: Raed Bouslama, st118415@stud.uni-stuttgart.de
"""
import numpy as np
import matplotlib.pyplot as plt
from func_FEM import *
from sol_analytical import *
from func_POD import *
from utils import *
from data_init import *


############################# Input Variables ################################
##############################################################################
Material=MATERIAL() # Define material properties
Material.E=30e9 #Young's modulus
Material.rho=7800 #density
Material.Nu=0.1 # poisson ratio
Material.L=20 #total length
Material.b=0.1 #beam depth
Material.h=0.1 #beam height
Material.A=Material.b*Material.h #beam cross sectional area
Material.I=(Material.b*Material.h**3)/12 #beam moment of inertia
Material.mass=Material.rho*Material.L*Material.A #total mass
Material.Irot=1/12*Material.mass*Material.L**2 #rotary inertia
Mesh=MESH() # Define geometry/mesh properties
Mesh.N=80 #number of elements 
Mesh.Nnodes=Mesh.N+1 #number of nodes
Mesh.Nmodes=12 #number of modes to keep
Mesh.L=20 #total length
Mesh.ell=Mesh.L/Mesh.N #length of beam element
Mesh.x=np.arange(0,Mesh.ell+Mesh.L,Mesh.ell) # spatial grid
load=-44.48 # load in Newton (-: facing downward)
t_int=np.linspace(0.1,2,20) # range of beam thicknesses
##############################################################################
##############################################################################


'''--------------------------- Test functions -----------------------------'''
disp=solve_FEM_system(Material, Mesh, load) # solve FEM
dx, dtheta=get_disp(disp, Mesh) # Split output in displacement and rotations
sigma=analytical_sol(Material, Mesh, load) # compute analytical solution
assert np.linalg.norm(dx-sigma,ord=2)<10e-5,'Large Error: Check implementation'
print('Implemented successfully - Proceed to CPU runtime analysis')

'''------------------- MOR and CPU runtime analysis -----------------------'''
X=snapshot_matrix(t_int, Material, Mesh, load, analytical=False, display=False)
r= dominant_modes(X, delta=0.9) # find index of dominant modes 
print('Number of dominant modes:            ', r+1)
N_elem=np.array([20,40,80,160,320,640,1280,2560])
t_svd, _=cpu_time(Material, Mesh, N_elem, t_int, load, r, method='svd')
t_cor, _=cpu_time(Material, Mesh, N_elem, t_int, load, r, method='corr')
t_snp, _=cpu_time(Material, Mesh, N_elem, t_int, load, r, method='snapshots')

'''------------------- Plot CPU runtime vs. n(nodes) ----------------------'''
fig, ax = plt.subplots(1, figsize=(10,6))
t_svd[t_svd == 0] = 'nan'
t_cor[t_cor == 0] = 'nan'
t_snp[t_snp == 0] = 'nan'
plt.plot(N_elem, t_svd, 'k', 
         linewidth=1, label='SVD of X')
plt.plot(N_elem, t_cor, 'b', 
         linewidth=1, label='EVD of correlation matrix XX*')
plt.plot(N_elem, t_snp, 'r', 
         linewidth=1, label='EVD of X*X, method of snapshots')
plt.yscale("log")
plt.xscale("log")
ax.set(xlabel=' n_nodes ', ylabel='t  [-]')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], frameon=False)
plt.savefig('cpu_runtime.png', bbox_inches='tight')
plt.show()
