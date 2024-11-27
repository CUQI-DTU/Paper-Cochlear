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
    read_experiment_data,\
    Callback

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
    args.sampler = 'NUTS'
    args.Ns = 10
    args.Nb = 5
    args.num_ST = 0
    args.noise_level = "estimated"
    args.true_a = 0.8 # funval
    args.rbc = 'fromDataClip'
    args.NUTS_kwargs['max_depth'] = 5
    args.version = 'results_temp_rhs'
    args.adaptive = True
    args.u0_from_data = True
    args.data_grad = True

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
(real_times, real_locations, real_data, real_std_data,
 diff_locations, real_data_diff, real_std_data_diff) = read_data_files(args)

# read all data as well num_ST = 4
cp_args = deepcopy(args)
cp_args.num_ST = 4
(real_times_all, real_locations_all, real_data_all, real_std_data_all,
    diff_locations_all, real_data_diff_all, real_std_data_diff_all) = read_data_files(cp_args)
# The left boundary condition is given by the data  
real_bc_l = real_data.reshape([len(real_locations), len(real_times)])[0,:]
print("real_bc_l (before)")
print(real_bc_l)
real_bc_l[real_bc_l<0] = 0
print("real_bc_l (after)")
print(real_bc_l)
# The right boundary condition is given by the data (if rbc is not "zero")
if args.rbc == 'fromData':
    raise Exception('Right boundary condition from data not supported')
elif args.rbc == 'fromDataClip':
    real_bc_r = real_data.reshape([len(real_locations), len(real_times)])[-1,:]
    print("real_bc_r (before)")
    print(real_bc_r)
    real_bc_r[real_bc_r<0] = 0
    print("real_bc_r (after)")
    print(real_bc_r)

else:
    real_bc_r = None

if args.u0_from_data:
    real_u0 = real_data.reshape([len(real_locations), len(real_times)])[:,0]
    print("real_u0 (before)")
    print(real_u0)
    real_u0[real_u0<0] = 0
    print("real_u0 (after)")
    print(real_u0)

# locations, including added locations that can be used in synthetic 
# case only
if len(args.add_data_pts) > 0:
    locations = np.concatenate((real_locations, np.array(args.add_data_pts)))
    # reorder the locations
    locations = np.sort(locations)
    diff_locations = locations[:-1]
else:
    locations = real_locations
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

#%% Step 4.1: Create u0
#-----------------------
if args.u0_from_data:
    # interpolate real_u0 to the grid
    u0 = np.interp(grid, locations, real_u0)
else:
    u0 = None

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
                           args.inference_type,
                           u0=u0)
# STEP 8: Create the CUQIpy PDE object
#-------------------------------------
PDE = TimeDependentLinearPDE(PDE_form,
                             tau,
                             grid_sol=grid,
                             method='backward_euler', 
                             grid_obs=locations,
                             time_obs=times,
                             data_grad=args.data_grad) 

# STEP 9: Create the range geometry
#----------------------------------
if args.data_grad:
    G_cont2D = Continuous2D((diff_locations, times))
else:
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
                                          time_obs=times,
                                          data_grad=args.data_grad) 
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
        real_data, real_std_data, G_cont2D,
        is_grad_data=args.data_grad, times=times, locations=locations, real_data_diff=real_data_diff,
        real_data_all=real_data_all, real_std_data_all=real_std_data_all, real_locations_all=real_locations_all)

print('s_noise')
print(s_noise)

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
    data = real_data_diff if args.data_grad else real_data
else:
    raise Exception('Data type not supported')

#%% STEP 15: Create the joint distribution
#-----------------------------------------
if args.sampler == 'NUTSWithGibbs':
    if args.data_grad:
        s = cuqi.distribution.Gamma(1.2, 5)
    else:
        s = cuqi.distribution.Gamma(1, 50000)
    joint = JointDistribution(x, s, y)
else:
    joint = JointDistribution(x, y)

#%% STEP 16: Create the posterior distribution
#---------------------------------------------
posterior = joint(y=data) # condition on y=y_obs

#%% STEP 17: Create the sampler and sample
#-----------------------------------------
# create the callback object
callback_obj = Callback(
                 dir_name=dir_name,
                 exact_x=exact_x,
                 exact_data=exact_data,
                 data=data.reshape(G_cont2D.fun_shape),
                 args=args, 
                 locations=diff_locations if args.data_grad else locations,
                 times=times, 
                 non_grad_data=real_data.reshape((len(locations), len(real_times))),            
                 non_grad_locations=locations,
                 L=L)

# time the sampling
import time
start_time = time.time()
# print A
print(A)
# print A domain geometry
print(A.domain_geometry)
# print A range geometry
print(A.range_geometry)

callback = None
if args.sampler_callback:
    callback = callback_obj

samples, my_sampler = sample_the_posterior(
    args.sampler, posterior, G_c, args, callback=callback)

lapsed_time = time.time() - start_time
#%% STEP 18: Plot the results
#----------------------------

x_samples = samples["x"] if args.sampler == 'NUTSWithGibbs' else samples
s_samples = samples["s"] if args.sampler == 'NUTSWithGibbs' else None

callback_obj(sampler=my_sampler, sample_index=None, s_samples=s_samples, plot_anyway=True)

# test reading the data
data_dic = read_experiment_data(parent_dir, tag)
