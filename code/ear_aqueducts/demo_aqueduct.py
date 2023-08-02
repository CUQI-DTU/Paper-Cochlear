#!/usr/bin/env python
# coding: utf-8

# # Modeling flow in ear aqueduct

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
parser = argparse.ArgumentParser(
    description='run the ear aqueduct Bayesian model')
parser.add_argument('-animal', metavar='animal', type=str,
                    choices=['m1', 'm2', 'm3', 'm4', 'm6'],
                    default='m1',
                    help='the animal to model')
parser.add_argument('-ear', metavar='ear', type=str, choices=[
                    'l', 'r'],
                    default='l',
                    help='the ear to model')
parser.add_argument('-version', metavar='version', type=str,
                    default='v10_temp',
                    help='the version of the model to run')
parser.add_argument('-sampler', metavar='sampler', type=str, choices=[
                    'CWMH', 'MH', 'NUTS'],
                    default='MH',
                    help='the sampler to use')
parser.add_argument('-unknown_par_type',
                    metavar='unknown_par_type',
                    type=str, choices=['constant',
                                       'smooth',
                                       'step',
                                       ],
                    default='constant',
                    help='Type of unknown parameter, diffusion coefficient')
parser.add_argument('-unknown_par_value', metavar='unknown_par_value',
                    type=list,
                    default=[500],
                    help='Value of unknown parameter, diffusion coefficient')
parser.add_argument('-data_type', metavar='data_type', type=str,
                    choices=[
                        'real', 'synthetic'],
                    default='synthetic',
                    help='Type of data, real or synthetic')
parser.add_argument('-inference_type', metavar='inference_type', type=str,
                    choices=[
                        'constant', 'heterogeneous', 'both'],
                    default='constant',
                    help='Type of inference, constant or heterogeneous coefficients')
args = parser.parse_args()

print('Arguments: animal = '+str(args.animal)+', ear = '+str(args.ear) +
      ', version = '+str(args.version)+', sampler = '+str(args.sampler))

## Read distance file
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file and times
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
times = constr_file['time'].values*60
print(times)
data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
data_bc = data.reshape([len(locations), len(times)])[0,:]
if args.data_type == 'synthetic':
    # Do not use real data
    data = None
else:
    # Print real data
    print(data)

## Create directory for output
version = args.version
if len(args.unknown_par_value) == 0:
    unknown_par_value_str = 'smooth_field'
elif len(args.unknown_par_value) == 1:
    unknown_par_value_str = str(args.unknown_par_value[0])
elif len(args.unknown_par_value) == 2:
    unknown_par_value_str = str(args.unknown_par_value[0])+\
        '_'+str(args.unknown_par_value[1])
else:
    raise Exception('Unknown parameter value not supported')

## Create directory for output
version = args.version
tag = args.animal+args.ear+args.sampler+args.unknown_par_type+\
    unknown_par_value_str+args.data_type+\
    args.inference_type+\
    version
print(tag)
dir_name = 'output'+tag
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')

## Save the current script in the output directory
os.system('cp '+__file__+' '+dir_name+'/')

#%%
## Set PDE parameters
L = 500
n_grid = 100   # Number of solution nodes
h = L/(n_grid+1)   # Space step size
grid = np.linspace(h, L-h, n_grid)

tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
dt_approx = cfl*h**2 # Defining approximate time step size
n_tau = int(tau_max/dt_approx)+1 # Number of time steps
tau = np.linspace(0, tau_max, n_tau)


### CASE 1: Constant diffusion coefficient 

## Source term (constant diffusion coefficient case)
def g_const(c, tau_current):
    f_array = np.zeros(n_grid)
    f_array[0] = c/h**2*np.interp(tau_current, times, data_bc)
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


### CASE 2: Varying in space diffusion coefficient

# grid for the diffusion coefficient
n_grid_c = 20
hs = L/(n_grid_c+1) 
grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)

## Source term (varying in space diffusion coefficient case)
def g_var(c, tau_current):
    f_array = np.zeros(n_grid)
    f_array[0] = c[0]/h**2*np.interp(tau_current, times, data_bc)
    return f_array

## Differential operator (varying in space diffusion coefficient case)
Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
vec = np.zeros(n_grid)
vec[0] = 1
Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
Dx /= h # FD derivative matrix

D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx

def PDE_form_var(c, tau_current):
    c = np.interp(grid_c_fine, grid_c, c)
    return (D_c_var(c), g_var(c, tau_current), initial_condition)

PDE_var = TimeDependentLinearPDE(PDE_form_var, tau, grid_sol=grid,
                             method='backward_euler', 
                             grid_obs=locations,
                            time_obs=times) 


# Domain geometry
G_D_var =  MappedGeometry( Continuous1D(grid_c),  map=lambda x: x**2 )

A_var = PDEModel(PDE_var, range_geometry=G_cont2D, domain_geometry=G_D_var)


### FOR ALL CASES: Create the data

# If the data is not provided, we create it
if args.data_type == 'synthetic':
    # if the unknown parameter is constant
    if args.unknown_par_type == 'constant':
        exact_x = args.unknown_par_value[0]
        exact_data = A_const(args.unknown_par_value[0])
    
    # if the unknown parameter is varying in space (step function)
    elif args.unknown_par_type == 'step':
        exact_x = np.zeros(n_grid_c)
        exact_x[0:n_grid_c//2] = args.unknown_par_value[0]
        exact_x[n_grid_c//2:] = args.unknown_par_value[1]
        exact_data = A_var(exact_x)

    # if the unknown parameter is varying in space (smooth function)
    elif args.unknown_par_type == 'smooth':
        low = args.unknown_par_value[0]
        high = args.unknown_par_value[1]
        exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low
        exact_data = A_var(exact_x)

## Noise standard deviation 
if args.data_type == 'synthetic':
    s_noise = 0.1 \
              *np.linalg.norm(exact_data) \
              *np.sqrt(1/G_cont2D.par_dim) 
elif args.data_type == 'real':
    s_noise = 0.1 \
              *np.linalg.norm(data) \
              *np.sqrt(1/G_cont2D.par_dim) 

### CASE 1: creating the prior and the data distribution
## Prior distribution (constant diffusion coefficient case)
x_const = Gaussian(np.sqrt(400), 100, geometry=G_D_const)

## Data distribution (constant diffusion coefficient case)
y_const = Gaussian(A_const(x_const), s_noise**2, geometry=G_cont2D)

### CASE 2: creating the prior and the data distribution
x_var = GMRF( np.ones(G_D_var.par_dim),2, geometry=G_D_var, bc_type='neumann')
#GMRF( np.ones(G_D.par_dim), 1, geometry=G_D)
#LMRF(x_zero, scale=0.002)

x_var.name
 
y_var = Gaussian(A_var(x_var), s_noise**2, geometry=G_cont2D)

### ALL CASES: creating the synthetic data
if args.data_type == 'synthetic':
    if args.unknown_par_type == 'constant':
        data = y_const(x_const=exact_x).sample()
    elif args.unknown_par_type == 'step' or args.unknown_par_type == 'smooth':
        data = y_var(x_var=exact_x).sample()

### CASE 1 SAMPLING: constant diffusion coefficient
## Joint distribution (constant diffusion coefficient case)
joint_const = JointDistribution(x_const, y_const)

## Wrap data in CUQIarray
#data = CUQIarray(data.ravel(), geometry=G_cont2D)

# Posterior distribution (constant diffusion coefficient case)
posterior_const = joint_const(y_const=data) # condition on y=y_obs

## Create sampler (constant diffusion coefficient case) and sample
Ns_const = 100
Nb_const = int(Ns_const*0.3) 
if args.sampler == 'MH':
    my_sampler_const = MH(posterior_const, scale=10, x0=20)
    posterior_samples_const = my_sampler_const.sample_adapt(Ns_const)
elif args.sampler == 'NUTS':
    posterior_const.enable_FD()
    my_sampler_const = NUTS(posterior_const, x0=20)
    posterior_samples_const = my_sampler_const.sample_adapt(
        Ns_const, int(Ns_const*0.1)) 
else:
    raise Exception('Unsuppported sampler')

posterior_samples_const_burnthin = posterior_samples_const.burnthin(Nb_const)

## plot posterior samples ci (constant diffusion coefficient case)
plt.figure()
posterior_samples_const_burnthin.plot_ci()
plt.title('Posterior samples ci (constant diffusion coefficient case)\n ESS = '+str(posterior_samples_const_burnthin.compute_ess()))

## save figure
plt.savefig(dir_name+'/posterior_samples_const_ci_'+tag+'.png')

## Plot data and reconstructed data (constant diffusion coefficient case)
plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

plt.sca(ax[0]) 
plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )

plt.sca(ax[1])
recon_data = A_const(posterior_samples_const_burnthin.funvals.mean(), is_par=False).reshape([len(locations), len(times)]) 
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


### CASE 2 SAMPLING: varying in space diffusion coefficient
joint_var = JointDistribution(x_var, y_var)

posterior_var = joint_var(y_var=data) 

Ns_var = 100
Nb_var = int(Ns_var*0.3)

if args.sampler == 'MH':
    my_sampler_var = MH(posterior_var, x0=np.ones(G_D_var.par_dim)*20)
    posterior_samples_var = my_sampler_var.sample_adapt(Ns_var)
elif args.sampler == 'NUTS':
    posterior_var.enable_FD()
    my_sampler_var = NUTS(posterior_var, x0=np.ones(G_D_var.par_dim)*20)
    posterior_samples_var = my_sampler_var.sample_adapt(Ns_var, int(Ns_var*0.1))
else:
    raise Exception('Unsuppported sampler')
#import cuqi
#BP = cuqi.problem.BayesianProblem(x_var, y_var).set_data(data)

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
recon_data = A_var(posterior_samples_var_burnthin.funvals.mean(),is_par=False).reshape([len(locations), len(times)])
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
