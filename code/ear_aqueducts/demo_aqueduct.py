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

import cuqi
print(cuqi.__version__)

from cuqi.distribution import Gaussian, JointDistribution, GMRF
from cuqi.geometry import Continuous1D, KLExpansion, Discrete,\
    MappedGeometry, Continuous2D, Image2D
from cuqi.pde import TimeDependentLinearPDE
from cuqi.model import PDEModel, Model
from cuqi.sampler import CWMH, MH, NUTS
from cuqi.array import CUQIarray
from cuqi.problem import BayesianProblem
from my_utils import plot_time_series, create_experiment_tag,\
    plot_experiment, process_experiment_par, read_experiment_data,\
    save_experiment_data
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
                    default='v_temp',
                    help='the version of the model to run')
parser.add_argument('-sampler', metavar='sampler', type=str, choices=[
                    'CWMH', 'MH', 'NUTS'],
                    default='NUTS',
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
                     nargs='*',
                     type=float,
                    default=[100],
                    help='Value of unknown parameter, diffusion coefficient')
parser.add_argument('-data_type', metavar='data_type', type=str,
                    choices=[
                        'real', 'synthetic'],
                    default='synthetic',
                    help='Type of data, real or synthetic')
parser.add_argument('-inference_type', metavar='inference_type', type=str,
                    choices=[
                        'constant', 'heterogeneous', 'both'],
                    default='both',
                    help='Type of inference, constant or heterogeneous coefficients')
parser.add_argument('-Ns_const', metavar='Ns_const', type=int,
                    default=300,
                    help='Number of samples for constant inference')
parser.add_argument('-Nb_const', metavar='Nb_const', type=int,
                    default=20,
                    help='Number of burn-in samples for constant inference')
parser.add_argument('-Ns_var', metavar='Ns_var', type=int,
                    default=10,
                    help='Number of samples for heterogeneous inference')
parser.add_argument('-Nb_var', metavar='Nb_var', type=int,
                    default=20,
                    help='Number of burn-in samples for heterogeneous inference')
parser.add_argument('-noise_level', metavar='noise_level', 
                    type=float,
                    default=0.1,
                    help='Noise level for data')
parser.add_argument('-add_data_pts', metavar='add_data_pts',
                    nargs='*',
                    type=float,
                    default=[])
# only CA or CA and ST
parser.add_argument('-data_pts_type', metavar='data_pts_type', type=str,
                    choices=[
                        'CA', 'CA_ST'],
                    default='CA',
                    help='Type of data points, CA or CA and ST')
# number of ST points used when -data_pts_type is CA_ST
parser.add_argument('-num_ST', metavar='num_ST', type=int,
                    choices=range(9),
                    default=0,
                    help='number of ST points used when -data_pts_type is CA_ST') 

args = parser.parse_known_args()[0]

## Process arguments
process_experiment_par(args)

if args.data_pts_type == 'CA':
    ## Read distance file
    dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
    locations_real = dist_file['distance microns'].values[:5]
    print(locations_real)
    print(args.add_data_pts)

    
    ## Read concentration file and times
    constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
    times = constr_file['time'].values*60
    print(times)
    data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
    data_bc = data.reshape([len(locations_real), len(times)])[0,:]
    if args.data_type == 'synthetic':
        # Do not use real data
        data = None
    else:
        # Print real data
        print(data)

elif args.data_pts_type == 'CA_ST':
    print('CA and ST')
    ## Read distance file
    dist_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
    # locations distance microns where 20210120_omnip10um_KX_M1_nosound_L is in
    # ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8']
    ST_list = ['ST'+str(i) for i in range(args.num_ST)]
    locations_keys = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5'].extend(ST_list)
    locations_real = dist_file['distance'].values


    print("locations_real")    
    print(locations_real)

    ## Read concentration file and times
    constr_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
    times = constr_file['time'].values*60
    print(times)
    data = constr_file[locations_keys].values.T.ravel()
    data_bc = data.reshape([len(locations_real), len(times)])[0,:]
    print(data)
    print(data_bc)


locations = np.concatenate((locations_real, np.array(args.add_data_pts)))
print(locations)
   

tag = create_experiment_tag(args)
print(tag)

## Create output directory
dir_name = 'results3/output'+tag
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')

## Save the current script in the output directory
os.system('cp '+__file__+' '+dir_name+'/')

#%%
## Set PDE parameters
L = locations_real[-1]*1.01
n_grid =int(L/5)   # Number of solution nodes
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
h_c = L/(n_grid_c+1) 
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
G_D_var =  MappedGeometry( Continuous1D(grid_c),  map=lambda x: x**2 )#TEMP: one map

A_var = PDEModel(PDE_var, range_geometry=G_cont2D, domain_geometry=G_D_var)


### FOR ALL CASES: Create the data

# If the data is not provided, we create it
exact_x = None
exact_data = None
if args.data_type == 'synthetic':
    # if the unknown parameter is constant
    if args.unknown_par_type == 'constant':
        exact_x = args.unknown_par_value[0]
        x_geom = G_D_const
        exact_x = CUQIarray(args.unknown_par_value[0], geometry=x_geom, is_par=False)
        exact_data = A_const(exact_x)
    
    # if the unknown parameter is varying in space (step function)
    elif args.unknown_par_type == 'step':
        exact_x = np.zeros(n_grid_c)
        exact_x[0:n_grid_c//2] = args.unknown_par_value[0]
        exact_x[n_grid_c//2:] = args.unknown_par_value[1]
        x_geom = G_D_var
        print('Exact parameter: ', exact_x)
        exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=False)
        exact_data = A_var(exact_x)

    # if the unknown parameter is varying in space (smooth function)
    elif args.unknown_par_type == 'smooth':
        low = args.unknown_par_value[0]
        high = args.unknown_par_value[1]
        exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low
        x_geom = G_D_var
        exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=False)        
        exact_data = A_var(exact_x)
    # TODO: Need to be removed later (when model output type
    # is fixed)
    exact_data = CUQIarray(exact_data, geometry=G_cont2D, is_par=False)

## Noise standard deviation 
if args.data_type == 'synthetic':
    s_noise = args.noise_level \
              *np.linalg.norm(exact_data) \
              *np.sqrt(1/G_cont2D.par_dim) 
elif args.data_type == 'real':
    s_noise = args.noise_level \
              *np.linalg.norm(data) \
              *np.sqrt(1/G_cont2D.par_dim) 

### CASE 1: creating the prior and the data distribution
## Prior distribution (constant diffusion coefficient case)
x_const = Gaussian(np.sqrt(400), 100, geometry=G_D_const)

## Data distribution (constant diffusion coefficient case)
y_const = Gaussian(A_const(x_const), s_noise**2, geometry=G_cont2D)

### CASE 2: creating the prior and the data distribution
x_var = GMRF( np.ones(G_D_var.par_dim),2, geometry=G_D_var, bc_type='neumann') #TEMP: fix loc and prec
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

    # TODO: Need to be fixed
    data = data.to_numpy()

### CASE 1 SAMPLING: constant diffusion coefficient
## Joint distribution (constant diffusion coefficient case)
joint_const = JointDistribution(x_const, y_const)

# Posterior distribution (constant diffusion coefficient case)
posterior_const = joint_const(y_const=data) # condition on y=y_obs

## Create sampler (constant diffusion coefficient case) and sample
Ns_const = args.Ns_const
Nb_const = args.Nb_const 
if args.sampler == 'MH':
    my_sampler_const = MH(posterior_const, scale=10, x0=20)
    posterior_samples_const = my_sampler_const.sample_adapt(Ns_const)
    posterior_samples_const_burnthin = posterior_samples_const.burnthin(Nb_const)
elif args.sampler == 'NUTS':
    posterior_const.enable_FD()
    my_sampler_const = NUTS(posterior_const, x0=20)
    posterior_samples_const = my_sampler_const.sample_adapt(
        Ns_const, Nb_const) 
    posterior_samples_const_burnthin = posterior_samples_const
else:
    raise Exception('Unsuppported sampler')


### CASE 2 SAMPLING: varying in space diffusion coefficient
joint_var = JointDistribution(x_var, y_var)

posterior_var = joint_var(y_var=data) 

Ns_var = args.Ns_var
Nb_var = args.Nb_var

if args.sampler == 'MH':
    my_sampler_var = MH(posterior_var, x0=np.ones(G_D_var.par_dim)*20)
    posterior_samples_var = my_sampler_var.sample_adapt(Ns_var)
    posterior_samples_var_burnthin = posterior_samples_var.burnthin(Nb_var)
elif args.sampler == 'NUTS':
    posterior_var.enable_FD()
    my_sampler_var = NUTS(posterior_var, x0=np.ones(G_D_var.par_dim)*20, max_depth=6)
    posterior_samples_var = my_sampler_var.sample_adapt(Ns_var, Nb_var)
    posterior_samples_var_burnthin = posterior_samples_var
else:
    raise Exception('Unsuppported sampler')

BP = BayesianProblem(x_var, y_var).set_data(y_var=data)

# Plot constant diffusion coefficient case
mean_recon_data_const = \
    A_const(posterior_samples_const_burnthin.funvals.mean(), is_par=False).\
        reshape([len(locations), len(times)])
plot_data = exact_data if exact_data else data
plot_x = exact_x if exact_x else CUQIarray(mean_recon_data_const, is_par=False, geometry=G_D_const)
fig = plot_experiment(plot_x, plot_data.reshape([len(locations), len(times)]),
                data.reshape([len(locations), len(times)]),
                mean_recon_data_const,
                posterior_samples_const_burnthin,
                args, locations, times)
# Save figure
fig.savefig(dir_name+'/experiment_const'+tag+'.png')

# Plot varying in space diffusion coefficient case
mean_recon_data_var = \
    A_var(posterior_samples_var_burnthin.funvals.mean(), is_par=False).\
        reshape([len(locations), len(times)])
plot_data = exact_data if exact_data else data
plot_x = exact_x if exact_x else CUQIarray(mean_recon_data_var, is_par=False, geometry=G_D_var)
fig = plot_experiment(plot_x, plot_data.reshape([len(locations), len(times)]),
                data.reshape([len(locations), len(times)]), 
                mean_recon_data_var,
                posterior_samples_var_burnthin,
                args, locations, times)
# Save figure
fig.savefig(dir_name+'/experiment_var'+tag+'.png')
    
# Save constant diffusion coefficient case
save_experiment_data(dir_name, exact_x, 
                     exact_data.reshape([len(locations), len(times)]) if exact_data else None,
                     data.reshape([len(locations), len(times)]),
                     mean_recon_data_const.reshape(
                         [len(locations), len(times)]),
                     posterior_samples_const_burnthin,
                     args, locations, times)

# Save varying in space diffusion coefficient case
save_experiment_data(dir_name, exact_x, 
                     exact_data.reshape([len(locations), len(times)]) if exact_data else None,
                     data.reshape([len(locations), len(times)]),
                     mean_recon_data_var.reshape(
                         [len(locations), len(times)]),
                     posterior_samples_var_burnthin,
                     args, locations, times)
