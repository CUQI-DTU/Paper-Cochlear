import argparse
from my_utils import process_experiment_par
import pandas as pd
import numpy as np
from cuqi.geometry import MappedGeometry, Discrete, Continuous1D
from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, GMRF
from cuqi.sampler import MH, NUTS

def parse_commandline_args(myargs):
    """Function to parse command line arguments."""
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
                         nargs='*',
                         type=float,
                        default=[100],
                        help='Value of unknown parameter, diffusion coefficient')
    parser.add_argument('-data_type', metavar='data_type', type=str,
                        choices=[
                            'real', 'synthetic_from_diffusion', 'synthetic_from_advection_diffusion'],
                        default='synthetic_from_diffusion',
                        help='Type of data, real or synthetic')
    parser.add_argument('-inference_type', metavar='inference_type', type=str,
                        choices=[
                            'constant', 'heterogeneous', 'advection_diffusion'],
                        default='constant',
                        help='Type of inference, constant or heterogeneous coefficients')
    parser.add_argument('-Ns', metavar='Ns', type=int,
                        default=20,
                        help='Number of samples')
    parser.add_argument('-Nb', metavar='Nb', type=int,
                        default=20,
                        help='Number of burn-in samples')
    parser.add_argument('-noise_level', metavar='noise_level', 
                        type=float,
                        default=0.1,
                        help='Noise level for data')
    parser.add_argument('-add_data_pts', metavar='add_data_pts',
                        nargs='*',
                        type=float,
                        default=[])
    # number of CA points used when -data_pts_type is CA
    parser.add_argument('-num_CA', metavar='num_CA', type=int,
                        choices=range(6),
                        default=5,
                        help='number of CA points')
    # number of ST points used when -data_pts_type is CA_ST
    parser.add_argument('-num_ST', metavar='num_ST', type=int,
                        choices=range(9),
                        default=0,
                        help='number of ST points used when -data_pts_type is CA_ST') 
    
    args = parser.parse_args(myargs)
    #parser.parse_known_args()[0]
    
    ## Process arguments
    process_experiment_par(args)

    return args

def read_data_files(args):
    """Function to read times array, locations array, concentration data from
    file. 
    Parameters
    ----------
    args : argparse.Namespace
        Arguments from command line.
    """

    if args.num_ST == 0: # Only CA data
        print('CA data.')
        ## Read distance file
        dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
        real_locations = dist_file['distance microns'].values[:args.num_CA]
        
        ## Read concentration file and times
        constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        CA_list = ['CA'+str(i+1) for i in range(args.num_CA)]
        real_data = constr_file[CA_list].values.T.ravel()
  
    elif args.num_ST > 0: # CA and ST data
        print('CA and ST data.')

        ## Read distance file
        dist_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
        # locations distance microns where 20210120_omnip10um_KX_M1_nosound_L is in
        # ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8']
        real_locations = dist_file['distance'].values
        ST_list = ['ST'+str(i+1) for i in range(args.num_ST)]
        CA_CT_list = CA_list.extend(ST_list)
    
        ## Read concentration file and times
        constr_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        real_data = constr_file[CA_CT_list].values.T.ravel()

    return real_times, real_locations, real_data

def create_domain_geometry(grid, coefficient_type):
    """Function to create domain geometry. """
    _map = lambda x: x**2
    if coefficient_type == 'constant':
        geometry = MappedGeometry( Discrete(1), map=_map)
    elif coefficient_type == 'heterogeneous':
        geometry = MappedGeometry( Continuous1D(grid),  map=_map)
    return geometry

def create_PDE_form(real_bc, grid, grid_c, grid_c_fine, n_grid, h, times,
                    coefficient_type):
    """Function to create PDE form. """
    ## Initial condition
    initial_condition = np.zeros(n_grid)

    if coefficient_type == 'constant':
        ## Source term (constant diffusion coefficient case)
        def g_const(c, tau_current):
            f_array = np.zeros(n_grid)
            f_array[0] = c/h**2*np.interp(tau_current, times, real_bc)
            return f_array
        
        ## Differential operator (constant diffusion coefficient case)
        D_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
        np.diag(np.ones(n_grid-1), -1) +
        np.diag(np.ones(n_grid-1), 1) ) / h**2
        
        ## PDE form (constant diffusion coefficient case)
        def PDE_form(c, tau_current):
            return (D_c_const(c), g_const(c, tau_current), initial_condition)
        
    elif coefficient_type == 'heterogeneous':
        ## Source term (varying in space diffusion coefficient case)
        def g_var(c, tau_current):
            f_array = np.zeros(n_grid)
            f_array[0] = c[0]/h**2*np.interp(tau_current, times, real_bc)
            return f_array
        
        ## Differential operator (varying in space diffusion coefficient case)
        Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
        vec = np.zeros(n_grid)
        vec[0] = 1
        Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
        Dx /= h # FD derivative matrix
        
        D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx
        
        ## PDE form (varying in space diffusion coefficient case)
        def PDE_form(c, tau_current):
            c = np.interp(grid_c_fine, grid_c, c)
            return (D_c_var(c), g_var(c, tau_current), initial_condition)
   

    return PDE_form

def create_prior_distribution(G_c, coefficient_type):
    """Function to create prior distribution. """
    if coefficient_type == 'constant':
        prior = Gaussian(np.sqrt(400), 100, geometry=G_c)
    elif coefficient_type == 'heterogeneous':
        prior = GMRF(
            np.ones(G_c.par_dim), 2, geometry=G_c, bc_type='neumann')
    return prior

def create_exact_solution_and_data(A, unknown_par_type, unknown_par_value):
    """Function to create exact solution and exact data. """

    n_grid_c = A.domain_geometry.par_dim
    x_geom = A.domain_geometry
    grid_c = x_geom.grid
    L = grid_c[-1]

    # if the unknown parameter is constant
    if unknown_par_type == 'constant':
        exact_x = np.zeros(n_grid_c)
        exact_x[:] = unknown_par_value[0]

    # if the unknown parameter is varying in space (step function)
    elif unknown_par_type == 'step':
        exact_x = np.zeros(n_grid_c)
        exact_x[0:n_grid_c//2] = unknown_par_value[0]
        exact_x[n_grid_c//2:] = unknown_par_value[1]

    # if the unknown parameter is varying in space (smooth function)
    elif unknown_par_type == 'smooth':
        low = unknown_par_value[0]
        high = unknown_par_value[1]
        exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low

    exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=False)
    exact_data = A(exact_x)
    return exact_x, exact_data

def set_the_noise_std(data_type, noise_level, exact_data, real_data, G_cont2D):
    """Function to set the noise standard deviation. """
    ## Noise standard deviation 
    if data_type == 'synthetic_from_diffusion':
        s_noise = noise_level \
                  *np.linalg.norm(exact_data) \
                  *np.sqrt(1/G_cont2D.par_dim)
    elif data_type == 'real':
        s_noise = noise_level \
                  *np.linalg.norm(real_data) \
                  *np.sqrt(1/G_cont2D.par_dim)
    else:
        raise Exception('Data type not supported')
    
    return s_noise

def sample_the_posterior(sampler, posterior, Ns, Nb, G_c):
    """Function to sample the posterior. """
    x0 = np.zeros(G_c.par_dim) + 20
    x0 = x0[0] if len(x0) == 1 else x0 # convert to float

    if sampler == 'MH':
        my_sampler = MH(posterior, scale=10, x0=x0)
        posterior_samples = my_sampler.sample_adapt(Ns)
        posterior_samples_burnthin = posterior_samples.burnthin(Nb)
    elif sampler == 'NUTS':
        posterior.enable_FD()
        my_sampler = NUTS(posterior, x0=x0, max_depth=6)
        posterior_samples = my_sampler.sample_adapt(Ns, Nb) 
        posterior_samples_burnthin = posterior_samples
    else:
        raise Exception('Unsuppported sampler')
    
    return posterior_samples_burnthin