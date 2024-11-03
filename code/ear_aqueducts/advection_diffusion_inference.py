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
    create_time_steps,\
    read_experiment_data

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
    args.data_type = 'real'
    args.inference_type = 'advection_diffusion'
    args.unknown_par_type = 'custom_1'
    #args.unknown_par_value = [100.0]
    args.sampler = 'NUTSWithGibbs'
    args.Ns = 5
    args.Nb = 1
    args.num_ST = 1
    args.noise_level = 0.1
    args.true_a = 0.8 # funval
    args.rbc = 'fromData'
    args.NUTS_kwargs['max_depth'] = 5
    args.version = 'results_temp_rhs'
    args.adaptive = True

else:
    args = parse_commandline_args(sys.argv[1:])
    # Add arguments that are not passed from the command line
    #args_predefined = Args()
    #args.NUTS_kwargs = args_predefined.NUTS_kwargs


if args.sampler == 'NUTSWithGibbs':
    args.NUTS_kwargs["enable_FD"] = True

# create a tag from the parameters of the experiment
tag = create_experiment_tag(args)
print('Tag:')
print(tag)

#%% STEP 2: Read time and location arrays
#----------------------------------------
real_times, real_locations, real_data, real_std_data = read_data_files(args)
# The left boundary condition is given by the data  
real_bc_l = real_data.reshape([len(real_locations), len(real_times)])[0,:]
# The right boundary condition is given by the data (if rbc is not "zero")
if args.rbc == 'fromData':
    real_bc_r = real_data.reshape([len(real_locations), len(real_times)])[-1,:]
else:
    real_bc_r = None

# locations, including added locations that can be used in synthetic 
# case only
locations = np.concatenate((real_locations, np.array(args.add_data_pts)))
# reorder the locations
locations = np.sort(locations)
# times
times = real_times

#%% STEP 3: Create output directory
#----------------------------------
parent_dir = 'results/'+args.version
dir_name = parent_dir +'/output'+tag
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')
# Save the current script in the output directory
os.system('cp '+__file__+' '+dir_name+'/')

#%% STEP 4: Create the PDE grid and coefficients grid
#----------------------------------------------------
# PDE and coefficients grids
factor_L = 1.2 if args.rbc == 'zero' else 1.01
L = locations[-1]*factor_L
coarsening_factor = 5
n_grid_c = 20
grid, grid_c, grid_c_fine, h, n_grid = build_grids(L, coarsening_factor, n_grid_c)

#%% STEP 5: Create the PDE time steps array
#------------------------------------------
tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
tau = create_time_steps(h, cfl, tau_max, args.adaptive)

#%% STEP 6: Create the domain geometry
#-------------------------------------
G_c = create_domain_geometry(grid_c, args.inference_type)

# STEP 7: Create the PDE form
#----------------------------
PDE_form = create_PDE_form(real_bc_l, real_bc_r,
                           grid, grid_c, grid_c_fine, n_grid, h, times,
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
    temp_inf_type = args.inference_type if args.inference_type != 'constant' else 'heterogeneous'
    PDE_form_var_diff = create_PDE_form(real_bc_l, real_bc_r, grid, grid_c, grid_c_fine,
                                   n_grid, h, times, temp_inf_type) 
    PDE_var_diff = TimeDependentLinearPDE(PDE_form_var_diff,
                                          tau,
                                          grid_sol=grid,
                                          method='backward_euler', 
                                          grid_obs=locations,
                                          time_obs=times) 
    G_c_var = create_domain_geometry(grid_c, temp_inf_type)    
    A_var_diff = PDEModel(
        PDE_var_diff, range_geometry=G_cont2D, domain_geometry=G_c_var)


    exact_x, exact_data = create_exact_solution_and_data(
        A_var_diff, args.unknown_par_type,
        args.unknown_par_value, args.true_a if args.inference_type == 'advection_diffusion' else None,
        grid_c=grid_c)

#%% STEP 13: Create the data distribution
#----------------------------------------
s_noise = set_the_noise_std(
    args.data_type, args.noise_level, exact_data,
    real_data, real_std_data, G_cont2D)

if args.sampler == 'NUTSWithGibbs':
    y = Gaussian(A(x), lambda s: 1/s, geometry=G_cont2D)
else:
    y = Gaussian(A(x), s_noise**2, geometry=G_cont2D)

#%% STEP 14: Specify the data for the inference
#----------------------------------------------
if args.data_type == 'syntheticFromDiffusion':
    y_temp = deepcopy(y)
    y_temp.mean = exact_data
    if args.sampler == 'NUTSWithGibbs':
        data = y_temp(s=1/s_noise**2).sample()
    else:
        data = y_temp.sample()

elif args.data_type == 'real':
    data = real_data
else:
    raise Exception('Data type not supported')

#%% STEP 15: Create the joint distribution
#-----------------------------------------
if args.sampler == 'NUTSWithGibbs':
    s = cuqi.distribution.Gamma(1, 50000)
    joint = JointDistribution(x, s, y)
else:
    joint = JointDistribution(x, y)

#%% STEP 16: Create the posterior distribution
#---------------------------------------------
posterior = joint(y=data) # condition on y=y_obs

#%% STEP 17: Create the sampler and sample
#-----------------------------------------
# time the sampling
import time
start_time = time.time()
samples, my_sampler = sample_the_posterior(
    args.sampler, posterior, G_c, args)

lapsed_time = time.time() - start_time
#%% STEP 18: Plot the results
#----------------------------

x_samples = samples["x"] if args.sampler == 'NUTSWithGibbs' else samples
s_samples = samples["s"] if args.sampler == 'NUTSWithGibbs' else None

mean_recon_data = \
    A(x_samples.funvals.mean(), is_par=False).reshape([len(locations), len(times)])

# if exact_data is not defined, set it to None
if exact_data is not None:
    exact_data = exact_data.reshape([len(locations), len(times)])
fig = plot_experiment(exact_x, exact_data,
                data.reshape([len(locations), len(times)]),
                mean_recon_data,
                x_samples,
                s_samples,
                args, locations, times, lapsed_time=lapsed_time, L=L)
# Save figure
fig.savefig(dir_name+'/experiment_'+tag+'.png')

#%% STEP 19: Save the results
#----------------------------
save_experiment_data(dir_name, exact_x, 
                     exact_data,
                     data.reshape([len(locations), len(times)]),
                     mean_recon_data,
                     x_samples,
                     s_samples,
                     args, locations, times, lapsed_time,
                     sampler=my_sampler)

# test reading the data
data_dic = read_experiment_data(parent_dir, tag)
