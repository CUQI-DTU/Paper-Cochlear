#!/usr/bin/env python
# coding: utf-8

# # Modeling flow in ear aqueduct

### CASE 1: Constant diffusion coefficient

## Imports 
#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import warnings
import os

from cuqi.distribution import Gaussian, JointDistribution, GMRF
from cuqi.geometry import Continuous1D, KLExpansion, Discrete, MappedGeometry, Continuous2D, Image2D
from cuqi.pde import TimeDependentLinearPDE
from cuqi.model import PDEModel, Model
from cuqi.sampler import CWMH, MH, NUTS
from cuqi.array import CUQIarray
from my_utils import plot_time_series
np.random.seed(1)

## Parse command line arguments
Parse = True

if Parse:
    parser = argparse.ArgumentParser(description='run the ear aqueduct Bayesian model')
    parser.add_argument('animal', metavar='animal', type=str, choices=['m1', 'm2', 'm3', 'm4', 'm6'], help='the animal to model')
    parser.add_argument('ear', metavar='ear', type=str, choices=['l', 'r'], help='the ear to model')
    args = parser.parse_args()
else:
    class args:
        animal = 'm2'
        ear = 'r'
    print('Using default arguments: animal = '+str(args.animal)+', ear = '+str(args.ear))

## Read distance file
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
print(data)
times = constr_file['time'].values*60
print(times)

## Create directory for output
version = 'v3'
tag = args.animal+args.ear+version
print(tag)
dir_name = 'output'+tag
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')

#%%
## Set PDE parameters
L = 500
n_grid = 50   # Number of solution nodes
h = L/(n_grid+1)   # Space step size
grid = np.linspace(h, L-h, n_grid)

tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
dt_approx = cfl*h**2 # Defining approximate time step size
n_tau = int(tau_max/dt_approx)+1 # Number of time steps
tau = np.linspace(0, tau_max, n_tau)

## Source term (constant diffusion coefficient case)
def g_const(c, tau_current):
    f_array = np.zeros(n_grid)
    f_array[0] = c/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
    return f_array

## Differential operator (constant diffusion coefficient case)
D_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
np.diag(np.ones(n_grid-1), -1) +
np.diag(np.ones(n_grid-1), 1) ) / h**2

## Initial condition
initial_condition = np.zeros(n_grid) 

## PDE form (constant diffusion coefficient case)
def PDE_form_const(c, tau_current):
    return (D_c_const(c), g_const(c, tau_current), initial_condition)

## CUQIpy PDE object (constant diffusion coefficient case)
PDE_const = TimeDependentLinearPDE(PDE_form_const, tau, grid_sol=grid,
                             method='backward_euler', 
                             grid_obs=locations,
                            time_obs=times) 

## Domain geometry (constant diffusion coefficient case)
G_D_const =  MappedGeometry( Discrete(1),  map=lambda x: x**2 )

## Range geometry
G_cont2D = Continuous2D((locations, times))

## PDE forward model (constant diffusion coefficient case)
A_const = PDEModel(PDE_const, range_geometry=G_cont2D, domain_geometry=G_D_const)

## Prior distribution (constant diffusion coefficient case)
x_const = Gaussian(np.sqrt(400), 100, geometry=G_D_const)

## Noise standard deviation 
s_noise = 0.1 \
          *np.linalg.norm(data) \
          *np.sqrt(1/G_cont2D.par_dim) 

## Data distribution (constant diffusion coefficient case)
y_const = Gaussian(A_const(x_const), s_noise**2, geometry=G_cont2D)

## Joint distribution (constant diffusion coefficient case)
joint_const = JointDistribution(x_const, y_const)

## Wrap data in CUQIarray
#data = CUQIarray(data.ravel(), geometry=G_cont2D)

# Posterior distribution (constant diffusion coefficient case)
posterior_const = joint_const(y_const=data) # condition on y=y_obs

## Create sampler (constant diffusion coefficient case)
my_sampler_const = MH(posterior_const, scale=10, x0=20)

## Sample (constant diffusion coefficient case)
Ns_const = 10
Nb_const = int(Ns_const*0.3)  
posterior_samples_const = my_sampler_const.sample_adapt(Ns_const)

posterior_samples_const_burnthin = posterior_samples_const.burnthin(Nb_const)

## plot posterior samples ci (constant diffusion coefficient case)
plt.figure()
posterior_samples_const_burnthin.plot_ci()

## save figure
plt.savefig(dir_name+'/posterior_samples_const_ci_'+tag+'.png')

## Plot data and reconstructed data (constant diffusion coefficient case)
plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

plt.sca(ax[0]) 
plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )

plt.sca(ax[1])
recon_data = A_const(posterior_samples_const_burnthin.funvals.mean()).reshape([len(locations), len(times)]) 
plot_time_series( times, locations, recon_data)

## save figure
plt.savefig(dir_name+'/data_recon_const_'+tag+'.png')

## Plot ESS (constant diffusion coefficient case)
plt.figure()
plt.plot(posterior_samples_const_burnthin.compute_ess(), 'o')
plt.title('ESS')

## save figure
plt.savefig(dir_name+'/ESS_const_'+tag+'.png')


## save data
np.savez(dir_name+'/posterior_samples_const_'+tag+'.npz', posterior_samples_const.samples)


### CASE 2: Varying in space diffusion coefficient

# grid for the diffusion coefficient
grid_c = np.linspace(0, L, n_grid+1, endpoint=True)

## Source term (varying in space diffusion coefficient case)
def g_var(c, tau_current):
    f_array = np.zeros(n_grid)
    f_array[0] = c[0]/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
    return f_array

## Differential operator (varying in space diffusion coefficient case)
Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
vec = np.zeros(n_grid)
vec[0] = 1
Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
Dx /= h # FD derivative matrix

D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx

def PDE_form_var(c, tau_current):
    return (D_c_var(c), g_var(c, tau_current), initial_condition)


PDE_var = TimeDependentLinearPDE(PDE_form_var, tau, grid_sol=grid,
                             method='backward_euler', 
                             grid_obs=locations,
                            time_obs=times) 


# Domain geometry
G_D_var =  MappedGeometry( Continuous1D(grid_c),  map=lambda x: x**2 )


A_var = PDEModel(PDE_var, range_geometry=G_cont2D, domain_geometry=G_D_var)

x_var = GMRF( np.ones(G_D_var.par_dim),2, geometry=G_D_var, bc_type='neumann')
#GMRF( np.ones(G_D.par_dim), 1, geometry=G_D)
#LMRF(x_zero, scale=0.002)

x_var.name
 
y_var = Gaussian(A_var(x_var), s_noise**2, geometry=G_cont2D)

joint_var = JointDistribution(x_var, y_var)

posterior_var = joint_var(y_var=data) 
posterior_var.enable_FD()
my_sampler_var = NUTS(posterior_var, x0=np.ones(G_D_var.par_dim)*20, max_depth=20)

Ns_var = 1000
Nb_var = int(Ns_var*0.3)
posterior_samples_var = my_sampler_var.sample_adapt(Ns_var,50)

posterior_samples_var_burnthin = posterior_samples_var.burnthin(Nb_var)

## plot posterior samples ci (varying in space diffusion coefficient case)
plt.figure()
posterior_samples_var_burnthin.plot_ci()

## save figure
plt.savefig(dir_name+'/posterior_samples_var_ci_'+tag+'.png')

## Plot data and reconstructed data (varying in space diffusion coefficient case)
plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

plt.sca(ax[0])
plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )

plt.sca(ax[1])
recon_data = A_var(posterior_samples_var_burnthin.funvals.mean()).reshape([len(locations), len(times)])
plot_time_series( times, locations, recon_data)

## save figure
plt.savefig(dir_name+'/data_recon_var_'+tag+'.png')

## Plot ESS (varying in space diffusion coefficient case)
plt.figure()
plt.plot(posterior_samples_var_burnthin.compute_ess())
plt.title('ESS')

## save figure
plt.savefig(dir_name+'/ESS_var_'+tag+'.png')

## save data
np.savez(dir_name+'/posterior_samples_var_'+tag+'.npz', posterior_samples_var.samples)
