#!/usr/bin/env python
# coding: utf-8

# # Modeling flow in ear aqueduct

#%% Imports 
import numpy as np
import os
import cuqi
import sys
from copy import deepcopy
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.geometry import Continuous2D
from cuqi.pde import TimeDependentLinearPDE
from cuqi.model import PDEModel
from advection_diffusion_inference_utils import parse_commandline_args,\
    read_data_files,\
    create_domain_geometry,\
    create_PDE_form,\
    create_prior_distribution,\
    create_exact_solution_and_data,\
    set_the_noise_std,\
    sample_the_posterior,\
    create_experiment_tag,\
    plot_experiment,\
    save_experiment_data,\
    Args,\
    build_grids,\
    create_time_steps

print('cuqi version:')
print(cuqi.__version__)

## Set random seed for reproducibility
np.random.seed(1)

## Command line  example
## command_example: python advection_diffusion_inference.py -data_type real -inference_type heterogeneous -unknown_par_type constant -unknown_par_value 400 -noise_level 0.1 -Ns 10 -Nb 10 -sampler MH

#%% STEP 1: Parse command line arguments
#---------------------------------------
# If no arguments are passed, use the default values
if len(sys.argv) <= 2:
    args = Args()
    args.data_type = 'syntheticFromDiffusion'
    args.inference_type = 'heterogeneous'
    args.unknown_par_type = 'constant'
    args.unknown_par_value = [100.0]
    args.sampler = 'MH'
    args.Ns = 20
    args.Nb = 10
    args.num_ST = 0
    args.noise_level = 0.001
else:
    args = parse_commandline_args(sys.argv[1:])

# Add arguments that are not passed from the command line
args_predefined = Args()
args.NUTS_kwargs = args_predefined.NUTS_kwargs

# create a tag from the parameters of the experiment
tag = create_experiment_tag(args)
print('Tag:')
print(tag)

#%% STEP 2: Read time and location arrays
#----------------------------------------
real_times, real_locations, real_data, real_std_data = read_data_files(args)
# The left boundary condition is given by the data  
real_bc = real_data.reshape([len(real_locations), len(real_times)])[0,:]
# locations, including added locations that can be used in synthetic 
# case only
locations = np.concatenate((real_locations, np.array(args.add_data_pts)))
# reorder the locations
locations = np.sort(locations)
# times
times = real_times

#%% STEP 3: Create output directory
#----------------------------------
dir_name = 'results5/output'+tag
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')
# Save the current script in the output directory
os.system('cp '+__file__+' '+dir_name+'/')

#%% STEP 4: Create the PDE grid and coefficients grid
#----------------------------------------------------
# PDE and coefficients grids
L = locations[-1]*1.1
coarsening_factor = 5
n_grid_c = 20
grid, grid_c, grid_c_fine, h, n_grid = build_grids(L, coarsening_factor, n_grid_c)

#%% STEP 5: Create the PDE time steps array
#------------------------------------------
tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
tau = create_time_steps(h, cfl, tau_max)

#%% STEP 6: Create the domain geometry
#-------------------------------------
G_c = create_domain_geometry(grid_c, args.inference_type)

# STEP 7: Create the PDE form
#----------------------------
PDE_form = create_PDE_form(real_bc, grid, grid_c, grid_c_fine, n_grid, h, times,
                           args.inference_type)
# STEP 8: Create the CUQIpy PDE object
#-------------------------------------
PDE = TimeDependentLinearPDE(PDE_form,
                             tau,
                             grid_sol=grid,
                             method='backward_euler', 
                             grid_obs=locations,
                             time_obs=times) 

# STEP 9: Create the range geometry
#----------------------------------
G_cont2D = Continuous2D((locations, times))

# STEP 10: Create the CUQIpy PDE model
#-------------------------------------
A = PDEModel(PDE, range_geometry=G_cont2D, domain_geometry=G_c)

# STEP 11: Create the prior distribution
#---------------------------------------
x = create_prior_distribution(G_c, args.inference_type)

# STEP 12: Create the exact solution and exact data (synthetic case only)
#------------------------------------------------------------------------
exact_x = None
exact_data = None
if args.data_type == 'syntheticFromDiffusion':
    PDE_form_var_diff = create_PDE_form(real_bc, grid, grid_c, grid_c_fine,
                                   n_grid, h, times, args.inference_type) 
    PDE_var_diff = TimeDependentLinearPDE(PDE_form_var_diff,
                                          tau,
                                          grid_sol=grid,
                                          method='backward_euler', 
                                          grid_obs=locations,
                                          time_obs=times) 
    G_c_var = create_domain_geometry(grid_c, args.inference_type)    
    A_var_diff = PDEModel(
        PDE_var_diff, range_geometry=G_cont2D, domain_geometry=G_c_var)
    exact_x, exact_data = create_exact_solution_and_data(
        A_var_diff, args.unknown_par_type,
        args.unknown_par_value, args.true_a, grid_c=grid_c)

#%% STEP 13: Create the data distribution
#----------------------------------------
s_noise = set_the_noise_std(
    args.data_type, args.noise_level, exact_data,
    real_data, real_std_data, G_cont2D)
y = Gaussian(A(x), s_noise**2, geometry=G_cont2D)

#%% STEP 14: Specify the data for the inference
#----------------------------------------------
if args.data_type == 'syntheticFromDiffusion':
    y_temp = deepcopy(y)
    y_temp.mean = exact_data
    data = y_temp.sample()

elif args.data_type == 'real':
    data = real_data
else:
    raise Exception('Data type not supported')

#%% STEP 15: Create the joint distribution
#-----------------------------------------
joint = JointDistribution(x, y)

#%% STEP 16: Create the posterior distribution
#---------------------------------------------
posterior = joint(y=data) # condition on y=y_obs

#%% STEP 17: Create the sampler and sample
#-----------------------------------------
# time the sampling
import time
start_time = time.time()
samples = sample_the_posterior(
    args.sampler, posterior, G_c, args)

lapsed_time = time.time() - start_time
#%% STEP 18: Plot the results
#----------------------------
mean_recon_data = \
    A(samples.funvals.mean(), is_par=False).reshape([len(locations), len(times)])

# if exact_data is not defined, set it to None
if exact_data is not None:
    exact_data = exact_data.reshape([len(locations), len(times)])
fig = plot_experiment(exact_x, exact_data,
                data.reshape([len(locations), len(times)]),
                mean_recon_data,
                samples,
                args, locations, times)
# Save figure
fig.savefig(dir_name+'/experiment_'+tag+'.png')

#%% STEP 19: Save the results
#----------------------------
save_experiment_data(dir_name, exact_x, 
                     exact_data,
                     data.reshape([len(locations), len(times)]),
                     mean_recon_data,
                     samples,
                     args, locations, times, lapsed_time)