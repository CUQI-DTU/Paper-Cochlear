import argparse
import json
import pandas as pd
import numpy as np
import scipy.stats as sps
from cuqi.geometry import MappedGeometry, Discrete, Continuous1D, Continuous2D
from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, GMRF
from cuqi.pde import TimeDependentLinearPDE, FD_spatial_gradient
from cuqi.model import PDEModel
import matplotlib.pyplot as plt
from custom_distribution import MyDistribution
from scipy.interpolate import interp1d
from cuqi.sampler import (HybridGibbs,
                                    NUTS,
                                    Conjugate,
                                    MH)
import cuqi
import time
import os
# choose Helvetica as the default font
rc = {"font.family" : "Helvetica", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)

try:
    import dill as pickle
except:
    # append local dill path in cluster
    import sys
    sys.path.append('../../../../tools')
    import dill as pickle

#Arg class
class Args:
    def __init__(self):
        self.animal = 'm1'
        self.ear = 'l'
        self.version = 'v_temp'
        self.sampler = 'MH'
        self.unknown_par_type = 'constant'
        self.unknown_par_value = [100.0]
        self.data_type = 'synthetic'
        self.inference_type = 'constant'
        self.Ns = 20
        self.Nb = 20
        self.noise_level = 0.1
        self.add_data_pts = []
        self.num_CA = 5
        self.num_ST = 0
        self.NUTS_kwargs = {'max_depth': 10}
        self.true_a = None
        self.rbc = "zero"
        self.adaptive = False
        self.data_grad = False
        self.u0_from_data = False
        self.sampler_callback = False
        self.pixel_data = False

class Callback:
    def __init__(self,
                 exact_x,
                 exact_data,
                 data,
                 args, 
                 locations,
                 times, 
                 non_grad_data=None,            
                 non_grad_locations=None,
                 L=None,
                 dir_name=None):
        self.exact_x = exact_x
        self.exact_data = exact_data
        self.data = data
        self.args = args
        self.locations = locations
        self.times = times
        self.non_grad_data = non_grad_data
        self.non_grad_locations = non_grad_locations
        self.L = L
        self.dir_name = dir_name
        #----------
        self.sampler = None #
        self.mean_recon_data = None
        self.lapsed_time = None #
        self._current_time = time.time() #
        self.x_samples = None #
        self.s_samples = None # will not be filled
        self.non_grad_mean_recon_data = None


 
    def __call__(self, sampler, sample_index, num_of_samples, plot_anyway=False, s_samples=None):
        # if a 10th of the samples have been generated save the
        # data and plot the results

        chunk_size = max(1, num_of_samples//10)
        if plot_anyway or (sample_index % chunk_size == 0 and sample_index > 4):
            self.sampler = sampler
            self.lapsed_time = time.time() - self._current_time
            self._current_time = time.time()
            # if sampler is Gibbs,
            if isinstance(sampler, cuqi.sampler.HybridGibbs):
                self.x_samples = sampler.get_samples()['x']
                s_samples = sampler.get_samples()['s']
                A = sampler.target._likelihoods[0].model
            else:
                self.x_samples = sampler.get_samples()
                A = sampler.target.model
            G_cont2D = A.range_geometry
            mean_recon_data = \
            A(self.x_samples.funvals.mean(), is_par=False).reshape(G_cont2D.fun_shape)
            non_grad_mean_recon_data = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])
            self.mean_recon_data = mean_recon_data
            self.non_grad_mean_recon_data = non_grad_mean_recon_data.reshape((len(self.non_grad_locations), len(self.times)))
            
            # Save data
            save_experiment_data(dir_name=self.dir_name,
                                 exact=self.exact_x, 
                                 exact_data=self.exact_data,
                                 data=self.data,
                                 mean_recon_data=self.mean_recon_data,
                                 x_samples=self.x_samples,
                                 s_samples=s_samples,
                                 experiment_par=self.args, 
                                 locations=self.locations,
                                 times=self.times,
                                 lapsed_time=self.lapsed_time,
                                 sampler=self.sampler)
            # suppress all plotting warnings temporarily
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)

            fig = plot_experiment(exact=self.exact_x,
                                  exact_data=self.exact_data,
                                  data=self.data,
                                  mean_recon_data=self.mean_recon_data,
                                  x_samples=self.x_samples,
                                  s_samples=s_samples,
                                  experiment_par=self.args, 
                                  locations=self.locations,
                                  times=self.times, 
                                  non_grad_data=self.non_grad_data,
                                  non_grad_mean_recon_data=self.non_grad_mean_recon_data,
                                  non_grad_locations=self.non_grad_locations,
                                  lapsed_time=self.lapsed_time,
                                  L=self.L)

            # Save figure
            tag = create_experiment_tag(self.args)
            
            fig.savefig(self.dir_name+'/experiment_'+tag+'_idx'+str(sample_index)+'.png')
            warnings.filterwarnings("default", category=UserWarning)
            









def all_animals():
    """Function to return all animals. """
    return ['m1', 'm2', 'm3', 'm4', 'm6']

def all_ears():
    """Function to return all ears. """
    return ['l', 'r']

def parse_commandline_args(myargs):
    """Function to parse command line arguments."""
    arg_obj = Args()
    parser = argparse.ArgumentParser(
        description='run the ear aqueduct Bayesian model')
    parser.add_argument('-animal', metavar='animal', type=str,
                        choices=all_animals(),
                        default=arg_obj.animal,
                        help='the animal to model')
    parser.add_argument('-ear', metavar='ear', type=str, choices=all_ears(),
                        default=arg_obj.ear,
                        help='the ear to model')
    parser.add_argument('-version', metavar='version', type=str,
                        default=arg_obj.version,
                        help='the version of the model to run')
    parser.add_argument('-sampler', metavar='sampler', type=str, choices=[
                        'CWMH', 'MH', 'NUTS', 'NUTSWithGibbs'],
                        default=arg_obj.sampler,
                        help='the sampler to use')
    parser.add_argument('-unknown_par_type',
                        metavar='unknown_par_type',
                        type=str, choices=['constant',
                                           'smooth',
                                           'step',
                                           'sampleMean',
                                            'custom_1',
                                            'synth_diff1.npz',
                                            'synth_diff2.npz',
                                            'synth_diff3.npz'
                                           ],
                        default=arg_obj.unknown_par_type,
                        help='Type of unknown parameter, diffusion coefficient')
    parser.add_argument('-unknown_par_value', metavar='unknown_par_value',
                         nargs='*',
                         type=str,
                        default=arg_obj.unknown_par_value,
                        help='Value of unknown parameter, diffusion coefficient, if unknown_par_type is constant, provide one value, if unknown_par_type is step, provide two values, if unknown_par_type is smooth, provide two values, if unknown_par_type is sampleMean, provide tag of the experiment concatenated with the directory name where the samples are stored, separated by @')
    parser.add_argument('-data_type', metavar='data_type', type=str,
                        choices=[
                            'real', 'synthetic'],
                        default=arg_obj.data_type,
                        help='Type of data, real or synthetic')
    parser.add_argument('-inference_type', metavar='inference_type', type=str,
                        choices=[
                            'constant', 'heterogeneous', 'advection_diffusion'],
                        default=arg_obj.inference_type,
                        help='Type of inference, constant or heterogeneous coefficients')
    parser.add_argument('-Ns', metavar='Ns', type=int,
                        default=arg_obj.Ns,
                        help='Number of samples')
    parser.add_argument('-Nb', metavar='Nb', type=int,
                        default=arg_obj.Nb,
                        help='Number of burn-in samples')
    parser.add_argument('-noise_level', metavar='noise_level', 
                        type=str,
                        default=arg_obj.noise_level,
                        help='Noise level for data, set to "fromDataVar" to read noise level from data that varies for each data point and set to "fromDataAvg" to compute average noise level from data and use it for all data points, set to "avgOverTime" to compute average noise level over time for each location, set to "estimated" to use the estimated noise level, or set to a float representing the noise level (e.g 0.1 for 10% noise). Noise level can also be a string that starts with "std_" then the std value. For example "std_5" means std of value 5') 
    parser.add_argument('-add_data_pts', metavar='add_data_pts',
                        nargs='*',
                        type=float,
                        default=arg_obj.add_data_pts)
    # number of CA points used when -data_pts_type is CA
    parser.add_argument('-num_CA', metavar='num_CA', type=int,
                        choices=range(20),
                        default=arg_obj.num_CA,
                        help='number of CA points')
    # number of ST points used when -data_pts_type is CA_ST
    parser.add_argument('-num_ST', metavar='num_ST', type=int,
                        choices=range(9),
                        default=arg_obj.num_ST,
                        help='number of ST points used when -data_pts_type is CA_ST') 
    parser.add_argument('-true_a', metavar='true_a', type=float, 
                        default=arg_obj.true_a,
                        help='true advection speed')
    parser.add_argument('-rbc', metavar='rbc', type=str, choices=['zero', 'fromData', 'fromDataClip'],
                        default=arg_obj.rbc,
                        help='right boundary condition')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive if passed, the adaptive time step is used')
    parser.add_argument('-NUTS_kwargs', metavar='NUTS_kwargs', type=str,
                        default=arg_obj.NUTS_kwargs,
                        help='kwargs for NUTS sampler')
    parser.add_argument('--data_grad', action='store_true',
                        help='data_grad if passed, the data is gradient data')
    parser.add_argument('--u0_from_data', action='store_true',
                        help='u0_from_data if passed, the initial condition is set using the data')
    parser.add_argument('--sampler_callback', action='store_true',
                        help='sampler_callback if passed, the sampler callback is used')
    parser.add_argument('--pixel_data',  action='store_true',
                        help='pixel_data if passed, the data is pixel data')

    args = parser.parse_args(myargs)
    #parser.parse_known_args()[0]
    
    ## Process arguments
    process_experiment_par(args)

    return args

def read_data_files(args):
    """Function to read times array, locations array, concentration data, std data from
    file. 
    Parameters
    ----------
    args : argparse.Namespace
        Arguments from command line.
    """


    if args.pixel_data:
        # assert num_ST is 0
        assert args.num_ST == 0, 'num_ST should be 0 when using pixel data'
        data_path =  './data/parsed/CT/ca1pixel'
        pre='ca1pixel'
        sep=''
        sep2=' ' 
    else:
        # assert num_CA is 5 or less
        assert args.num_CA <= 5, 'num_CA should be 5 or less when using averaged data'
        data_path = './data/parsed/CT'
        pre='20210120'
        sep='_'
        sep2=''

    CA_list = ['CA'+sep2+str(i+1) for i in range(args.num_CA)]

    if args.num_ST == 0: # Only CA data
        ## Read distance file
        dist_file = pd.read_csv(data_path+'/'+pre+'_'+args.animal+sep+args.ear+'_distances.csv')
        real_locations = dist_file['distance microns'].values[:args.num_CA]
        
        ## Read concentration file and times
        constr_file = pd.read_csv(data_path+'/'+pre+'_'+args.animal+sep+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        real_data = constr_file[CA_list].values.T

        ## Read std data
        CA_std_list = [item+' std' for item in CA_list]
        real_std_data = constr_file[CA_std_list].values.T
  
    elif args.num_ST > 0: # CA and ST data

        ## Read distance file
        dist_file = pd.read_csv(data_path+'/combined_CA_ST/'+pre+'_'+args.animal+sep+args.ear+'_distances.csv')
        # locations distance microns where 20210120_omnip10um_KX_M1_nosound_L is in
        # ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8']
        real_locations = dist_file['distance'].values
        real_locations = real_locations[:args.num_CA+args.num_ST]
        ST_list = ['ST'+str(i+1) for i in range(args.num_ST)]
        CA_ST_list = CA_list + ST_list
    
        ## Read concentration file and times
        constr_file = pd.read_csv(data_path+'/combined_CA_ST/'+pre+'_'+args.animal+sep+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        real_data = constr_file[CA_ST_list].values.T
        ## Read std data
        std_file = pd.read_csv(data_path+'/'+pre+'_'+args.animal+sep+args.ear+'_parsed.csv')
        CA_ST_std_list = [item+' std' for item in CA_ST_list]
        real_std_data = std_file[CA_ST_std_list].values.T
    if args.data_grad:
        real_data_diff = np.zeros((real_data.shape[0]-1, real_data.shape[1]))

        for i in range(real_data.shape[0]-1):
            real_data_diff[i] = (real_data[i] - real_data[i+1])/(real_locations[i] - real_locations[i+1])
        diff_locations = real_locations[:-1]
        real_std_data_diff = np.zeros_like(real_data)*np.nan
    # ravel the arrays
    real_data = real_data.ravel()
    real_std_data = real_std_data.ravel()
    if args.data_grad:
        real_data_diff = real_data_diff.ravel()
        real_std_data_diff = real_std_data_diff.ravel()
    else:
        real_data_diff = None
        real_std_data_diff = None
        diff_locations = None

    return real_times, real_locations, real_data, real_std_data, diff_locations, real_data_diff, real_std_data_diff

def build_grids(L, coarsening_factor, n_grid_c):
    # PDE grid
    n_grid =int(L/coarsening_factor)   # Number of solution nodes
    h = L/(n_grid+1)   # Space step size
    grid = np.linspace(h, L-h, n_grid)
    # Coefficients grid
    h_c = L/(n_grid_c+1) 
    grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
    grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)
    assert np.isclose(grid_c[-1], L)
    return grid, grid_c, grid_c_fine, h, n_grid

def create_time_steps(h, cfl, tau_max, adaptive):
    """Function to create time steps array. """
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)
    if adaptive:
        # insert 4 time steps between the first two time steps
        additional_timesteps = np.array(
            [tau[0]+(tau[1]-tau[0])*frac for frac in [0.2, 0.4, 0.6, 0.8]])
        tau = np.concatenate((tau[:1], additional_timesteps, tau[1:]))
    return tau

def create_domain_geometry(grid, inference_type):
    """Function to create domain geometry. """
    _map = lambda x: x**2
    if inference_type == 'constant':
        geometry = MappedGeometry( Discrete(1), map=_map)
    elif inference_type == 'heterogeneous':
        geometry = MappedGeometry( Continuous1D(grid),  map=_map)
    elif inference_type == 'advection_diffusion':
        # advection diffusion map is x^2 except for the last element which is x
        def _map_advection_diffusion(x):
            output = np.zeros_like(x)
            output[:-1] = x[:-1]**2
            output[-1] = x[-1]
            x = output
            return x
        geometry = MappedGeometry( Discrete(len(grid)+1),
                                  map=_map_advection_diffusion)
    return geometry

def create_PDE_form(real_bc_l, real_bc_r,
                    grid, grid_c, grid_c_fine, n_grid, h,
                    times, inference_type, u0=None):
    """Function to create PDE form. """
    ## Initial condition
    if u0 is None:
        u0 = np.zeros(n_grid)
        u0[0] = real_bc_l[0]
        if real_bc_r is not None:
            u0[-1] = real_bc_r[0]
    initial_condition = u0

    if inference_type == 'constant':
        ## Source term (constant diffusion coefficient case)
        def g_const(c, tau_current):
            f_array = np.zeros(n_grid)
            f_array[0] = c/h**2*np.interp(tau_current, times, real_bc_l)
            if real_bc_r is not None:
              f_array[-1] = c/h**2*np.interp(tau_current, times, real_bc_r)
            return f_array
        
        ## Differential operator (constant diffusion coefficient case)
        D_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
        np.diag(np.ones(n_grid-1), -1) +
        np.diag(np.ones(n_grid-1), 1) ) / h**2
        
        ## PDE form (constant diffusion coefficient case)
        def PDE_form(c, t):
            return (D_c_const(c), g_const(c, t), initial_condition)
        
    elif inference_type == 'heterogeneous':
        ## Source term (varying in space diffusion coefficient case)
        def g_var(c, t):
            f_array = np.zeros(n_grid)
            f_array[0] = c[0]/h**2*np.interp(t, times, real_bc_l)
            if real_bc_r is not None:
              f_array[-1] = c[-1]/h**2*np.interp(t, times, real_bc_r)
            return f_array
        
        ## Differential operator (varying in space diffusion coefficient case)
        Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
        vec = np.zeros(n_grid)
        vec[0] = 1
        Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
        Dx /= h # FD derivative matrix
        
        D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx
        
        ## PDE form (varying in space diffusion coefficient case)
        def PDE_form(c, t):
            c = np.interp(grid_c_fine, grid_c, c)
            return (D_c_var(c), g_var(c, t), initial_condition)

    elif inference_type == 'advection_diffusion':
        ## Source term (varying in space diffusion coefficient case)
        def g_var(c, t):
            f_array = np.zeros(n_grid)
            u_0_mplus1 = np.interp(t, times, real_bc_l) 
            f_array[0] += u_0_mplus1*c[0]/h**2 + c[-1]*u_0_mplus1/(2*h)
            if real_bc_r is not None:
                u_L_m = np.interp(t, times, real_bc_r)
                f_array[-1] += c[-2]/h**2*u_L_m - c[-1]*u_L_m/(2*h)
            return f_array
        
        ## Differential operator (varying in space diffusion coefficient case)
        Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
        vec = np.zeros(n_grid)
        vec[0] = 1
        Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
        Dx /= h # FD derivative matrix
 
        def DA_a(a):
            if True:
                # centered difference
                DA =  (np.diag(np.ones(n_grid-1), 1) +\
            -np.diag(np.ones(n_grid-1), -1)) * (a/(2*h))
                return DA

        def D_c_var(c):
            Mat = - Dx.T @ np.diag(c) @ Dx 
            return Mat
        
        ## PDE form (varying in space diffusion coefficient case)
        def PDE_form(x, t):
            c = np.interp(grid_c_fine, grid_c, x[:-1])
            return (D_c_var(c) - DA_a(x[-1]), g_var(x, t), initial_condition)   

    return PDE_form

def create_prior_distribution(G_c, inference_type):
    """Function to create prior distribution. """
    if inference_type == 'constant':
        prior = Gaussian(np.sqrt(400), 100, geometry=G_c)
    elif inference_type == 'heterogeneous':
        prior = Gaussian(20, 5**2, geometry=G_c) 
        #prior = GMRF(
        #    np.ones(G_c.par_dim)*np.sqrt(300),
        #    0.2,
        #    geometry=G_c,
        #    bc_type='neumann')

        #prior = Gaussian(50, 10**2, geometry=G_c)
        # Gauss3 Gaussian(30, 10**2, geometry=G_c)
        # Gauss4 Gaussian(20, 5**2, geometry=G_c) 
            #np.ones(G_c.par_dim)*np.sqrt(600),
            #0.04,
            #geometry=G_c,
            #bc_type='neumann')
    # 5 x = GMRF(np.ones(G_c.par_dim)*np.sqrt(1000), 0.05, geometry=G_c, bc_type='neumann')
    # 6 x = GMRF(np.ones(G_c.par_dim)*np.sqrt(1000), 0.04, geometry=G_c, bc_type='neumann')
    # 7 x = GMRF(np.ones(G_c.par_dim)*np.sqrt(600), 0.04, geometry=G_c, bc_type='neumann')
    # GMRF 2:         prior = GMRF(
    #        np.ones(G_c.par_dim)*np.sqrt(300),
    #        0.1,
    #        geometry=G_c,
    #        bc_type='neumann') 
    elif inference_type == 'advection_diffusion':
        # Gauss4 Gaussian(np.ones(G_c.par_dim-1)*20, 5**2)
        # Gauss7 Gaussian(np.ones(G_c.par_dim-1)*20, 10**2)
        prior1 = Gaussian(np.ones(G_c.par_dim-1)*20, 5**2)
        # GMRF2 prior1 = GMRF(np.ones(G_c.par_dim-1)*np.sqrt(300),
        #    0.1,
        #    bc_type='neumann')
        #prior1 = GMRF(np.ones(G_c.par_dim-1)*np.sqrt(300),
        #    0.2,
        #    bc_type='neumann')
        # Gibb gmrf true 6: prec 0.2


        # TODO: change the "a" prior mean and std to be 0 and 0.752 (which is
        # the square root of advection speed that results in a peclet number
        # of 1)
        var_a_sqrt = 0.752**2
        var_a = 2*var_a_sqrt**2
        prior2 =Gaussian(0, var_a)# Gaussian(0.5, 0.3**2)
        prior = MyDistribution([prior1, prior2], geometry=G_c )
    return prior

def create_exact_solution_and_data(A, unknown_par_type, unknown_par_value, a=None, grid_c=None):
    """Function to create exact solution and exact data. """
    # if unknown_par_value is a list of strings, convert it to a list of floats
    if isinstance(unknown_par_value, list):
        try:
            unknown_par_value = [float(item) for item in unknown_par_value]
        except:
            pass
    
    #TODO: add a mechanism to insure that a is not None if and only if 
    # the inference_type is advection_diffusion (also find a better way to pass
    # the grid_c)
    n_grid_c = A.domain_geometry.par_dim
    x_geom = A.domain_geometry


    # if the unknown parameter is constant
    if unknown_par_type == 'constant':
        exact_x = np.zeros(n_grid_c)
        exact_x[:] = unknown_par_value[0]
        is_par = False

    # if the unknown parameter is varying in space (step function)
    elif unknown_par_type == 'step':
        exact_x = np.zeros(n_grid_c)
        exact_x[0:n_grid_c//2] = unknown_par_value[0]
        exact_x[n_grid_c//2:] = unknown_par_value[1]
        is_par = False

    # if the unknown parameter is varying in space (smooth function)
    elif unknown_par_type == 'smooth':
        if a is None:
            grid_c = grid_c#x_geom.grid
        else:
            grid_c = grid_c
        L = grid_c[-1]
        low = unknown_par_value[0]
        high = unknown_par_value[1]
        exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low
        is_par = False

    elif unknown_par_type == 'sampleMean':
        # Read data from pickle file
        print('Reading data from: ', unknown_par_value)
        tag = unknown_par_value[0].split('@')[0].replace(':', '_')
        data_dict = read_experiment_data(unknown_par_value[0].split('@')[1], tag)
        samples = data_dict['samples']
        exact_x = samples.mean()
        exact_x = exact_x.to_numpy() if isinstance(exact_x, CUQIarray) else exact_x
        is_par = True
        if a is not None:
            raise NotImplementedError
            #a = np.sqrt(a)

    elif unknown_par_type == 'custom_1':
        #TODO: this if else is repeated (refactor)
        if a is None:
            grid_c = x_geom.grid
        else:
            grid_c = grid_c
        true_custom_grid = \
        np.array([   0.        ,   96.25220602,  192.50441205,  288.75661807,
        385.0088241 ,  481.26103012,  577.51323615,  673.76544217,
        770.0176482 ,  866.26985422,  962.52206025, 1058.77426627,
       1155.0264723 , 1251.27867832, 1347.53088435, 1443.78309037,
       1540.03529639, 1636.28750242, 1732.53970844, 1828.79191447,
       1925.04412049])
        true_custom_data = \
        np.array([14.11491597, 12.53944334,  9.02154746,  5.72036963,
            4.68179363,  7.80182922, 10.1970911 , 11.54031693,
           12.46494568, 13.05172206, 13.40779027, 13.61382111,
           13.77862088, 13.86445104, 13.99664015, 14.02779958,
           14.08190319, 14.10144676, 14.11415816, 14.10891881,
           14.11514915])
        # spline interpolation
        f = interp1d(true_custom_grid, true_custom_data, kind='cubic')
        exact_x = f(grid_c)
        is_par = True
        if a is not None:
            raise NotImplementedError
            #a = np.sqrt(a)

    elif unknown_par_type.endswith('.npz'):
        # Read data from npz file
        print('Reading data from: ', unknown_par_type)
        # get current file path
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        # load the npz file
        exact_x = np.load(current_file_path+"/../data/synth_diff/"+unknown_par_type)['arr_0']
        is_par = False


    ## append "a" value to the end
    if a is not None and unknown_par_type != 'constant':
        exact_x = np.append(exact_x, a)
    exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=is_par)
    exact_data = A(exact_x)
    exact_nongrad_data = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])
    return exact_x, exact_data, exact_nongrad_data


def estimate_noise_std(locations, times, real_data, real_std_data):
    """Function to estimate the noise standard deviation. """

    data_for_noise_estimation_per_case = []
    std_data_for_noise_estimation_per_case = []

    for j, loc in enumerate(locations):
        for k, t in enumerate(times/60):
            orig_data = real_data[j, k]
            orig_std_data = real_std_data[j, k]

            # if data below line (500, 0) to (3000, 15), add to noise estimation
            line_eq = lambda loc_x:  15/2500*(loc_x-500)
            # draw line
            if t < line_eq(loc):
                data_for_noise_estimation_per_case.append(orig_data)
                std_data_for_noise_estimation_per_case.append(orig_std_data)
    
    return np.sqrt(np.average(np.array(std_data_for_noise_estimation_per_case)**2))

def estimate_grad_data_noise_std(data_noise_std, locations, data_diff):
    std_not_scaled = np.sqrt(2)*data_noise_std
    diff_locations = np.diff(locations)
    std_per_loc = np.zeros_like(data_diff)
    # loop over rows of location factor
    for i in range(std_per_loc.shape[0]):
        std_per_loc[i, :] = std_not_scaled/diff_locations[i]
    std_matrix = np.diag(std_per_loc.flatten())
    return std_not_scaled, std_matrix

def set_the_noise_std(
        data_type, noise_level, exact_data,
        real_data, real_std_data, G_cont2D,
        is_grad_data, times, locations, real_data_diff,
        real_data_all, real_std_data_all, real_locations_all):
    """Function to set the noise standard deviation. """
    # Use noise levels read from the file
    if noise_level == "fromDataVar":
        ## Noise standard deviation
        if is_grad_data:
            raise Exception('Noise level "fromDataVar" not supported yet for gradient data')
        s_noise = real_std_data
    # Use noise level specified in the command line
    elif noise_level == "fromDataAvg":
        if is_grad_data:
            raise Exception('Noise level "fromDataAvg" not supported yet for gradient data')
        s_noise = np.mean(real_std_data)

    elif noise_level == "avgOverTime":
        if is_grad_data:
            raise Exception('Noise level "avgOverTime" not supported yet for gradient data')
        s_noise = real_std_data.reshape(G_cont2D.fun_shape)
        s_noise = np.mean(s_noise, axis=1)
        s_noise = np.repeat(s_noise, G_cont2D.fun_shape[1])
    elif noise_level == "estimated":
        if not is_grad_data:
            raise Exception('Noise level not supported yet for non gradient data')
        estimated_noise_std = estimate_noise_std(
            locations=real_locations_all,
            times=times,
            real_data=real_data_all.reshape((len(real_locations_all),-1)),
            real_std_data=real_std_data_all.reshape((len(real_locations_all),-1)))
        
        std_scaled, std_matrix = estimate_grad_data_noise_std(data_noise_std=estimated_noise_std,
                                                       locations=locations,
                                                       data_diff=real_data_diff.reshape((len(locations)-1,-1))
                                                       )
        s_noise = np.diag(std_matrix)

    elif noise_level.startswith('std_'):
        std_value = float(noise_level.split('_')[1])
        s_noise = std_value*np.ones(G_cont2D.par_dim)
    else:
        try:
            noise_level = float(noise_level)
        except:
            raise Exception('Noise level not supported')
        ## Noise standard deviation 
        if data_type == 'synthetic':
            if is_grad_data:
                raise Exception('Noise level not supported yet for gradient data and synthetic data')
            s_noise = noise_level \
                      *np.linalg.norm(exact_data) \
                      *np.sqrt(1/G_cont2D.par_dim)
        elif data_type == 'real':
            used_data = real_data if not is_grad_data else real_data_diff
            s_noise = noise_level \
                      *np.linalg.norm(used_data) \
                      *np.sqrt(1/G_cont2D.par_dim)
        else:
            raise Exception('Data type not supported')
    
    return s_noise

def sample_the_posterior(sampler, posterior, G_c, args, callback=None):
    """Function to sample the posterior. """
    Ns = args.Ns
    Nb = args.Nb

    x0 = np.zeros(G_c.par_dim) + 20
    x0 = x0[0] if len(x0) == 1 else x0 # convert to float

    # if the parameters contains advection speed, set the initial
    # value of the advection speed to 0
    if (args.inference_type == 'advection_diffusion' and
        isinstance(x0, np.ndarray)):
        x0[-1] = 0

    if sampler == 'MH':
        my_sampler = MH(posterior, scale=10,
                           initial_point=x0,
                           callback=callback)
        my_sampler.warmup(Nb)
        my_sampler.sample(Ns)
        posterior_samples = my_sampler.get_samples()
        posterior_samples_burnthin = posterior_samples.burnthin(Nb)
    elif sampler == 'NUTS':
        posterior.enable_FD()
        NUTS_kwargs = args.NUTS_kwargs
        my_sampler = NUTS(posterior, initial_point=x0, **NUTS_kwargs, callback=callback)
        my_sampler.warmup(Nb)
        my_sampler.sample(Ns)
        posterior_samples = my_sampler.get_samples()
        posterior_samples_burnthin = posterior_samples
    elif sampler == 'NUTSWithGibbs':
        posterior.enable_FD()
        sampling_strategy = {
            "x" : NUTS(initial_point=x0, **args.NUTS_kwargs),
            "s" : Conjugate()
        }
        
        # Here we do 1 internal steps with NUTS for each Gibbs step
        num_sampling_steps = {
            "x" : 1,
            "s" : 1
        }
        
        my_sampler = HybridGibbs(posterior, sampling_strategy, num_sampling_steps, callback=callback)
        my_sampler.warmup(Nb)
        my_sampler.sample(Ns)
        posterior_samples = my_sampler.get_samples()
        posterior_samples_burnthin = posterior_samples

    else:
        raise Exception('Unsupported sampler')
    
    return posterior_samples_burnthin, my_sampler

def plot_time_series(times, locations, data, plot_legend=True, plot_type='over_time', d3_alpha=0, marker=None, linestyle='-', colormap=None, y_log=False, plot_against=None, clip_on=True):
    # Plot data
    # plot type can be 'over_time' or 'over_location' or 'surface'
    if colormap is None:
        color = ['r', 'g', 'b', 'k', 'm', 'c', 'brown']
    else:
        no_colors = len(locations) if plot_type == 'over_time' else len(times)
        color = colormap(np.linspace(0, 1, no_colors))
    if plot_type == 'over_time':

        legends = ["{}".format(int(obs))+ u" (\u03bcm)" for obs in locations]
        lines = []
        for i in range(len(locations)):
            lines.append(plt.plot(times/60, data[i,:],  color=color[i%len(color)],marker=marker, linestyle=linestyle, clip_on=clip_on)[0])
        
        if plot_legend:
            plt.legend(lines, legends)
        plt.xlabel('Time (min)')
        plt.ylabel('Concentration')
        

    elif plot_type == 'over_location':
        legends = ["{} (min.)".format(int(obs/60)) for obs in times]
        lines = []
        for i in range(len(times)):
            lines.append(plt.plot(locations, data[:,i],  color=color[i%len(color)],marker=marker, linestyle=linestyle, clip_on=clip_on)[0])
        
        if plot_legend:
            plt.legend(lines, legends)
        plt.xlabel('Location')
        plt.ylabel('Concentration')
        

    elif plot_type == 'surface': 
        #fig = plt.figure()
        ax = plt.gcf().add_subplot(111, projection='3d')
        X, Y = np.meshgrid(times/60, locations)
        ax.plot_surface(X, Y, data, cmap='viridis', alpha=d3_alpha)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Location')
        ax.set_zlabel('Concentration')
        lines = None
        legends = None
        # rotate the plot

    elif plot_type == 'against_data':
        if plot_against is None:
            raise Exception('plot_against must be provided when plot_type is "against_data"')
        if len(plot_against) != len(data):
            raise Exception('plot_against must have the same length as data')
        legends = ["{}".format(int(obs))+u" (\u03bcm)" for obs in locations]
        lines = []
        for i in range(len(locations)):
            lines.append(plt.scatter(plot_against[i,:], data[i,:],  color=color[i%len(color)],marker=marker, clip_on=clip_on))

        if plot_legend:
            plt.legend(lines, legends)
        plt.xlabel('data')
        plt.ylabel('reconstruction')
        if y_log:
            plt.xscale('log')

        
    
    else:
        raise Exception('Unsupported plot type')
    # set y scale to log
    if y_log:
        plt.yscale('log')

    return lines, legends

def save_experiment_data(dir_name, exact, exact_data, data, mean_recon_data,
                    x_samples, s_samples, experiment_par, locations, times, lapsed_time, sampler):
    # is const inference
    #const = True if samples.geometry.par_dim == 1 else False

    #if const:
    #    name_str = 'const'
    #else:
    #    name_str = 'var'
    name_str = 'var'
    
    # This is a workaround solution to not pickle the CUQIarray object
    # exact because it loses properties with pickling. 
    # We pickle its components instead (the geometry, the flag is_par 
    # and the numpy array).
    if isinstance(exact,CUQIarray):
        # convert exact to array and save its geometry
        exact_geometry = exact.geometry
        exact_is_par = exact.is_par
        exact = exact.to_numpy()
    else:
        exact_geometry = None
        exact_is_par =None

    # This is a workaround solution to not pickle the CUQIarray object
    # exact_data because it loses properties with pickling. 
    # We pickle its components instead (the geometry, the flag is_par 
    # and the numpy array).
    if isinstance(exact_data, CUQIarray):
        # convert exact_data to array and save its geometry
        exact_data_geometry = exact_data.geometry
        exact_data_is_par = exact_data.is_par
        exact_data = exact_data.to_numpy()
    else:
        exact_data_geometry = None
        exact_data_is_par =None

    # Save data in pickle file named with tag
    tag = create_experiment_tag(experiment_par)
    data_dict = {'exact': exact,
                 'exact_geometry': exact_geometry,
                 'exact_is_par': exact_is_par,
                 'exact_data': exact_data,
                 'exact_data_geometry': exact_data_geometry,
                 'exact_data_is_par': exact_data_is_par,
                 'data': data,
                 'mean_recon_data': mean_recon_data,
                 'x_samples': x_samples,
                 's_samples': s_samples,
                 'experiment_par': experiment_par, 'locations': locations,
                 'times': times,
                 'lapse_time': lapsed_time,
                 'num_tree_node_list': None,
                 'epsilon_list': None}
    # if sampler is NUTs, save the number of tree nodes
    if isinstance(sampler, cuqi.sampler.NUTS):
        data_dict['num_tree_node_list'] = sampler.num_tree_node_list
        data_dict['epsilon_list'] = sampler.epsilon_list
    
    # if sampler is HybridGibbs, save the number of tree nodes if the
    # underlying sampler is NUTS
    elif isinstance(sampler, cuqi.sampler.HybridGibbs):
        if isinstance(sampler.samplers['x'], cuqi.sampler.NUTS):
            data_dict['num_tree_node_list'] = sampler.samplers['x'].num_tree_node_list
            data_dict['epsilon_list'] = sampler.samplers['x'].epsilon_list

    with open(dir_name +'/'+tag+'_'+name_str+'.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

def read_experiment_data(dir_name, tag):
    # Read data from pickle file
    file_name = dir_name +'/output'+ tag+'/'+tag+'_var.pkl'
    with open(file_name, 'rb') as f:
        data_dict = pickle.load(f)

    # Convert exact to CUQIarray with geometry
    if data_dict['exact_geometry'] is not None:
        exact = CUQIarray(data_dict['exact'], 
                          geometry=data_dict['exact_geometry'],
                          is_par=data_dict['exact_is_par'])
    else:
        exact = data_dict['exact']

    data_dict['exact'] = exact
    # drop geometry and is_par
    data_dict.pop('exact_geometry')
    data_dict.pop('exact_is_par')

    # Convert exact_data to CUQIarray with geometry
    if data_dict['exact_data_geometry'] is not None:
        exact_data = CUQIarray(data_dict['exact_data'].flatten(), 
                               geometry=data_dict['exact_data_geometry'],
                               is_par=data_dict['exact_data_is_par'])
    else:
        exact_data = data_dict['exact_data']
    data_dict['exact_data'] = exact_data
    # drop geometry and is_par
    data_dict.pop('exact_data_geometry')
    data_dict.pop('exact_data_is_par')

    return data_dict

def plot_experiment(exact, exact_data, data, mean_recon_data,
                    x_samples, s_samples, experiment_par, locations, times,
                    non_grad_data=None,
                    non_grad_mean_recon_data=None,
                    non_grad_locations=None,
                    lapsed_time=None, L=None):
    """Method to plot the numerical experiment results."""
    # Create tag
    tag = create_experiment_tag(experiment_par)
    # if experiment tag is so long, break it into two lines
    if len(tag) > 80:
        tag = tag[:80]+'\n'+tag[80:]

    # Expr type (const or var)
    const_inf = True if x_samples.geometry.par_dim == 1 else False
    const_true_x = True 
    if exact is not None:
        const_true_x = True if exact.geometry.par_dim == 1 else False
    
    # x_sample funval mean
    x_samples_funvals_mean = x_samples.funvals.mean()

    # s_sample mean
    if s_samples is not None:
        s_samples_mean = s_samples.mean()
    else:
        s_samples_mean = np.nan

    # Set up that depdneds on the whether inference is constant or variable
    # and whether true parameter is constant or variable:
    # a. Number of rows in bottom subfigure
    axsBottom_rows= 3 if const_inf else 3 
    # b. Set exact_for_plot to None if inferred parameter and true parameter
    # are of different geometries
    exact_for_plot = None
    if exact is not None:
        exact_for_plot = exact if const_true_x==const_inf else None
    # Hight ratio of top and bottom subfigures
    height_ratios = [0.2, 1.3, 1, 0.2] if const_inf else [0.2, 1.3, 1, 0.2]
    # Trace index list
    trace_idx_list = [0] if const_inf else [0, 5, -1] # last one is advective 
                                                      # speed in case of 
                                                      # advection_diffusion
                                                      # model

    # Marker
    marker = 'o' if const_true_x else ''

    # Create figure: 
    fig = plt.figure(figsize=(12, 20), layout='constrained')

    subfigs = fig.subfigures(4, 1, height_ratios=height_ratios)

    axsSecond = subfigs[1].subplots(5, 2,
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.1, top=0.95,
                         hspace=0.64, wspace=0.5))
    axsFirst = subfigs[0].subplots(1, 1, 
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.1, top=0.5))
    axesThird = subfigs[2].subplots(axsBottom_rows, 2,
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.1, top=0.96,
                         hspace=0.5, wspace=0.5))
    
    # last subfigure is empty and used to write text
    axesLast = subfigs[3].subplots(1, 2, 
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.15, top=0.9))
    

    # Add super title
    subfigs[0].suptitle('Experiment results: '+tag)

    # Plot exact data
    if exact_data is not None:
        plt.sca(axsSecond[0, 0])
        plot_time_series(times, locations, exact_data, plot_legend=False)
        plt.title('Exact data')

    # Plot data
    plt.sca(axsSecond[0, 1])
    plot_time_series(times, locations, data, plot_legend=False)
    plt.title('Noisy data')

    # Plot reconstructed data
    plt.sca(axsSecond[1, 0])
    lines, legends = plot_time_series(times, locations, mean_recon_data, plot_legend=False)
    plt.title('Mean reconstructed data')

    # noisy Noisy data - exact data
    if exact_data is not None:
        plt.sca(axsSecond[1, 1])
        plot_time_series(times, locations, data - exact_data, plot_legend=False)
        plt.title(' Noisy data - exact data\n relative error to noisy data = {:.2f}%'.format(np.linalg.norm(data - exact_data)/np.linalg.norm(data)*100))

    # plot Noisy data - mean reconstructed data
    plt.sca(axsSecond[2, 0])
    plot_time_series(times, locations, data - mean_recon_data, plot_legend=False)
    plt.title('Noisy data - mean reconstructed data\n relative error to noisy data = {:.2f}%'.format(np.linalg.norm(data - mean_recon_data)/np.linalg.norm(data)*100))

    # Plot credible intervals
    plt.sca(axsSecond[2, 1])
    x_samples.funvals.plot_ci(exact = exact_for_plot)
    # If inference type is not constant, plot data locations as vertical lines
    if not const_inf and experiment_par.inference_type != 'advection_diffusion':
        for loc in locations:
            plt.axvline(x = loc, color = 'gray', linestyle = '--')
    # TODO: print out the means only in advection diffusion case
    plt.title('Posterior samples CI (mean advection= {:.2f})'.format(x_samples_funvals_mean[-1]))

    # Plot ESS
    plt.sca(axsSecond[3, 0])
    ESS_list = np.array(x_samples.compute_ess()) 
    plt.plot(ESS_list, marker=marker)
    plt.title('ESS (min = {:.2f})'.format(np.min(ESS_list)))

    # Plot exact   
    if exact is not None:
        plt.sca(axsSecond[3, 1])
        exact.plot(marker=marker) 
        plt.title('Exact solution')
    # plot legend 
    axsFirst.axis('off')
    axsFirst.legend(lines, legends, loc='center', ncol=3)
    
    # plot data (not grad)
    plt.sca(axsSecond[4, 0])
    plot_time_series(times, non_grad_locations, non_grad_data, plot_legend=False)
    plt.title('Noisy data (not grad)')

    # plot mean reconstructed data (not grad)
    plt.sca(axsSecond[4, 1])
    plot_time_series(times, non_grad_locations, non_grad_mean_recon_data, plot_legend=False)
    plt.title('Mean reconstructed data (not grad)')


    # Plot trace
    x_samples.plot_trace(trace_idx_list, axes=axesThird, tight_layout=False)

    # write lapse time, exact a , exact peclet number, and mean peclet number
    # in the last subfigure
    axesLast[0].axis('off')
    axesLast[0].text(0.1, 0.8, 'Lapse time: {:.2f} s'.format(lapsed_time))
    if experiment_par.true_a is not None:
        # print exact a
        axesLast[0].text(0.1, 0.65, 'Exact a: {:.2f}'.format(experiment_par.true_a))
        # print inferred a
        axesLast[0].text(0.1, 0.5, 'Inferred a: {:.2f}'.format(x_samples_funvals_mean[-1]))
    if experiment_par.inference_type == 'advection_diffusion':
        if exact is not None:
            min_exact_peclet = peclet_number(a=experiment_par.true_a,
                                         d=np.max(exact[:-1])**2,
                                         L=L)
            max_exact_peclet = peclet_number(a=experiment_par.true_a,
                                         d=np.min(exact[:-1])**2,
                                         L=L)
            axesLast[0].text(0.1, 0.55, 'Exact peclet number range: [{:.2f}, {:.2f}]'.format(min_exact_peclet, max_exact_peclet))

        min_inferred_peclet = peclet_number(a=x_samples_funvals_mean[-1],
                                            d=np.max(x_samples_funvals_mean[:-1]),
                                            L=L)
        max_inferred_peclet = peclet_number(a=x_samples_funvals_mean[-1],
                                            d=np.min(x_samples_funvals_mean[:-1]),
                                            L=L)

        axesLast[0].text(0.1, 0.4, 'Inferred peclet number range: [{:.2f}, {:.2f}]'.format(min_inferred_peclet, max_inferred_peclet))
    # if s samples is not None, print mean and std of std samples
    if s_samples is not None:
        std_samples = np.sqrt(1/s_samples.samples.flatten())
        axesLast[0].text(0.1, 0.25, 'Mean of std samples: {:.2f}'.format(np.mean(std_samples)))
        axesLast[0].text(0.1, 0.05, 'Std of std samples: {:.2f}'.format(np.std(std_samples)))
    # print NUTS kwargs
    if experiment_par.NUTS_kwargs is not None:
        axesLast[0].text(0.1, -0.1, 'NUTS kwargs: {}'.format(experiment_par.NUTS_kwargs))
    # plot the histogram of the std samples
    if s_samples is not None:
        plt.sca(axesLast[1])
        plt.hist(std_samples, bins=20, color='blue', alpha=0.7)
        plt.title('Histogram of std samples')
    return fig

def process_experiment_par(experiment_par):
    """Method to create a tag from the parameters of the experiment. """
    # Assert if real data, you cannot add data points
    if experiment_par.data_type == 'real' and len(experiment_par.add_data_pts) > 0:
        raise Exception('Cannot add data points to real data')
    
    if len(experiment_par.unknown_par_value) not in [1, 2]:
        raise Exception('Unknown parameter value not supported')
    
    # use json to convert NUTS_kwargs to dictionary
    if experiment_par.NUTS_kwargs is not None:
        experiment_par.NUTS_kwargs = json.loads(experiment_par.NUTS_kwargs)
    
    # Raise exception if more than one data point is added, unable to
    # create tag
    #if len(experiment_par.add_data_pts) > 1:
    #    raise Exception('Only one data point can be added')
    
    # If inference type is not both, raise exception
    #if experiment_par.inference_type not in ['both']:
    #    raise Exception('Inference type not supported')

def create_experiment_tag(experiment_par):
    """Method to create a tag from the parameters of the experiment. """
    # Create directory for output
    version = experiment_par.version
    if isinstance(experiment_par.unknown_par_value, list):
        if len(experiment_par.unknown_par_value) == 1:
            unknown_par_value_str = str(experiment_par.unknown_par_value[0])
            if '@' in unknown_par_value_str:
                unknown_par_value_str = unknown_par_value_str.split('@')[0]
        elif len(experiment_par.unknown_par_value) == 2:
            unknown_par_value_str = str(experiment_par.unknown_par_value[0])+\
                '_'+str(experiment_par.unknown_par_value[1])
    elif isinstance(experiment_par.unknown_par_value, str):
        unknown_par_value_str = experiment_par.unknown_par_value.split('@')[0]
    else:
        print("experiment_par.unknown_par_value", experiment_par.unknown_par_value)
        raise Exception('Unknown parameter value not supported')
    
    if experiment_par.true_a is not None:
        true_a_str = str(experiment_par.true_a)
    else:
        true_a_str = 'none'
    # Concatenate data points
    data_pt_str = '' #'pt'+'pt'.join([str(i) for i in experiment_par.add_data_pts]) if len(experiment_par.add_data_pts) > 0 else ''
    # Create directory for output
    tag = experiment_par.animal+'_'+experiment_par.ear+'_'+\
        experiment_par.sampler+'_'+experiment_par.unknown_par_type+'_'+\
        unknown_par_value_str+'_'+experiment_par.data_type+'_'+\
        experiment_par.inference_type+'_'+\
        str(experiment_par.Ns)+'_'+\
        str(experiment_par.noise_level)+'_'+\
        version+'_'+\
        data_pt_str+'_'+\
        str(experiment_par.num_ST)+'_'+\
        str(experiment_par.num_CA)+'_'+\
        true_a_str+'_'+\
        experiment_par.rbc
    
    return tag

def matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 


def create_args_list(animals, ears, noise_levels, num_ST_list, add_data_pts_list, unknown_par_types, unknown_par_values, data_type, version, samplers, Ns_s, Nb_s, inference_type_s=['heterogeneous'], true_a_s=None, rbc_s=None, NUTS_kwargs = None, data_grad=False, u0_from_data=False, sampler_callback=False, pixel_data=False, adaptive=False):
    args_list = []
    # Loop over all animals, ears, noise levels and num_ST
    for animal in animals:
        for ear in ears:
            for noise_level in noise_levels:
                for num_ST in num_ST_list:
                    for add_data_pts in add_data_pts_list:
                        for i_unknown_par_type, unknown_par_type in enumerate(unknown_par_types):
                            #for unknown_par_value in unknown_par_values:
                                for i_sampler, sampler in enumerate(samplers):
                                    for true_a in true_a_s:
                                        for rbc in rbc_s:
                                            for inference_type in inference_type_s: 

                                                args = Args()
                                                args.animal = animal if animal is not None else unknown_par_values[i_unknown_par_type].split(':')[0]
                                                args.ear = ear if ear is not None else unknown_par_values[i_unknown_par_type].split(':')[1]
                                                args.version = version
                                                args.sampler = sampler
                                                args.data_type = data_type
                                                args.Ns = Ns_s[i_sampler]
                                                args.Nb = Nb_s[i_sampler]
                                                args.noise_level = noise_level
                                                args.num_ST = num_ST
                                                args.add_data_pts = add_data_pts
                                                args.inference_type = inference_type
                                                args.unknown_par_type = unknown_par_type
                                                args.unknown_par_value = unknown_par_values[i_unknown_par_type]
                                                args.true_a = true_a
                                                args.rbc = rbc
                                                if NUTS_kwargs is not None:
                                                    args.NUTS_kwargs = NUTS_kwargs
                                                args_list.append(args)
                                                args.data_grad = data_grad
                                                args.u0_from_data = u0_from_data
                                                args.sampler_callback = sampler_callback
                                                args.pixel_data = pixel_data
                                                args.adaptive = adaptive
    return args_list


def peclet_number(a, d, L):
    """Function to compute the peclet number.
    Parameters
    ----------
    a : float
        Advection speed.
    d : float
        Diffusion coefficient.
    L : float
        Length of the domain.
    """ 
    return a*L/d

def advection_speed(peclet_number, d, L):
    """Function to compute the advection speed.
    Parameters
    ----------
    peclet_number : float
        Peclet number.
    d : float
        Diffusion coefficient.
    L : float
        Length of the domain.
    """ 
    return peclet_number*d/L



def create_A(data_diff):
    # STEP 2: Read time and location arrays
    #----------------------------------------
    args = data_diff['experiment_par']
    (real_times, real_locations, real_data, real_std_data,
     diff_locations, real_data_diff, real_std_data_diff) = read_data_files(args)
    

    # The left boundary condition is given by the data  
    real_bc_l = real_data.reshape([len(real_locations), len(real_times)])[0,:]

    real_bc_l[real_bc_l<0] = 0

    # The right boundary condition is given by the data (if rbc is not "zero")
    if args.rbc == 'fromData':
        raise Exception('Right boundary condition from data not supported')
    elif args.rbc == 'fromDataClip':
        real_bc_r = real_data.reshape([len(real_locations), len(real_times)])[-1,:]

        real_bc_r[real_bc_r<0] = 0

    
    else:
        real_bc_r = None
    
    if args.u0_from_data:
        real_u0 = real_data.reshape([len(real_locations), len(real_times)])[:,0]

        real_u0[real_u0<0] = 0

    
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
    # STEP 4: Create the PDE grid and coefficients grid
    #----------------------------------------------------
    # PDE and coefficients grids
    factor_L = 1.2 if args.rbc == 'zero' else 1.01
    L = locations[-1]*factor_L
    coarsening_factor = 5
    n_grid_c = 20
    grid, grid_c, grid_c_fine, h, n_grid = build_grids(L, coarsening_factor, n_grid_c)
    
    # Step 4.1: Create u0
    #-----------------------
    if args.u0_from_data:
        # interpolate real_u0 to the grid
        u0 = np.interp(grid, locations, real_u0)
    else:
        u0 = None
    
    # STEP 5: Create the PDE time steps array
    #------------------------------------------
    tau_max = 30*60 # Final time in sec
    cfl = 5 # The cfl condition to have a stable solution
             # the method is implicit, we can choose relatively large time steps 
    tau = create_time_steps(h, cfl, tau_max, args.adaptive)
    
    # STEP 6: Create the domain geometry
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
    observation_map = FD_spatial_gradient if args.data_grad else None
    PDE = TimeDependentLinearPDE(PDE_form,
                                 tau,
                                 grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                 time_obs=times,
                                 observation_map=observation_map) 
    
    # STEP 9: Create the range geometry
    #----------------------------------
    if args.data_grad:
        G_cont2D = Continuous2D((diff_locations, times))
    else:
        G_cont2D = Continuous2D((locations, times))
    
    # STEP 10: Create the CUQIpy PDE model
    #-------------------------------------
    A = PDEModel(PDE, range_geometry=G_cont2D, domain_geometry=G_c)

    return A

def read_all_control_scenarios(scenarios_dir, scenarios_subdir, scenario_tags_list):
    
    data_list = []
    for i, case in enumerate(scenario_tags_list):
            
            # Read the experiment data
            data_list.append(
                read_experiment_data(
                    scenarios_dir+"/"+scenarios_subdir[i],
                    scenario_tags_list[i]
                    ))

    return data_list
def read_all_real_scenarios(scenario_dir, scenario_tag, animal_ear):
    
    # Read all cases for the scenario
    data_diff_list = []
    data_adv_list = []
    for animal, ear in animal_ear:
            # Read the experiment data
            data_diff_list.append(
                read_experiment_data(
                    scenario_dir+"/"+scenario_tag[0],
                    animal+"_"+ear+"_"+scenario_tag[2]
                    ))
            data_adv_list.append(
                read_experiment_data(
                    scenario_dir+"/"+scenario_tag[1],
                    animal+"_"+ear+"_"+scenario_tag[3]))
    return data_diff_list, data_adv_list


def plot_control_case(data_list, plot_type='over_time', colormap=None, d_y_coor=0.4):
    # create a 4 by 5 plot, each row is for a different case
    # in the data_list. The first column is for the exact data,
    # the second column is for the predicted data (mean reconstruction data)
    # the third column is for the diffusion CI
    # the fourth column is for the advection prior and posterior
    # the fifth column is for the hyperparamter prior and posterior
    # note that the first row does not have the advection parameter
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

    legend_y = -0.5
    legend_x = 0.5

    fig, axs = plt.subplots(4, 5, figsize=(7, 5.5), layout="constrained")
    fig.subplots_adjust(wspace=0.0)

    for i in range(len(data_list)):
        # create A
        A = create_A(data_list[i])

        (
            real_times,
            real_locations,
            real_data,
            real_std_data,
            diff_locations,
            real_data_diff,
            real_std_data_diff,
        ) = read_data_files(data_list[i]["experiment_par"])

        # noisy non gradient data
        # print keys of data_list[i]
        # print(data_list[i].keys())
        noisy_grad_data = data_list[i]["data"]
        mean_recon_data = A(data_list[i]["x_samples"].funvals.mean(), is_par=False)
        non_grad_mean_recon_data = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])

        exact_data = A(data_list[i]["exact"], is_par=False)
        non_grad_exact_data = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])
        # Plot the mean reconstruction data
        plt.sca(axs[i, 0])
        plot_legend = False
        lines, labels = plot_time_series(
            real_times, real_locations, non_grad_mean_recon_data, plot_legend=plot_legend,
            plot_type=plot_type,
            colormap=colormap
        )
        plot_time_series(
            real_times,
            real_locations,
            non_grad_exact_data,
            plot_legend=False,
            marker="*",
            linestyle="",
            plot_type=plot_type,
            colormap=colormap
        )
        plt.ylim(-500, 4000)    
        plt.xlim(real_times[0]/60, real_times[-1]/60)
        plt.ylabel(r"$\boldsymbol{c}$")
        plt.gca().yaxis.set_label_coords(0.13, 0.5)
        if i == 3:
            plt.legend(lines, labels, loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            plt.xlabel("Time (min.)")
        else:
            plt.xlabel("")
            # keep ticks but remove ticks labels for x 
            plt.gca().tick_params(labelbottom=False) 


        # Plot the credibility interval of the inferred diffusion parameter
        plt.sca(axs[i, 1])
        if i == 0:

            l_ci1 = data_list[i]["x_samples"].funvals.plot_ci(68)
            exact=data_list[i]["exact"]

        else:
            l_ci2 = cuqi.samples.Samples(
                data_list[i]["x_samples"].samples[:-1, :],
                geometry=data_list[0]["x_samples"].geometry,
            ).plot_ci(68, plot_envelope_kwargs={"facecolor": "gray", "edgecolor": "gray"}, color="gray")
            exact = cuqi.array.CUQIarray(
                data_list[i]["exact"].to_numpy()[:-1],
                is_par=False,
                geometry=data_list[0]["x_samples"].geometry,
            )
        l_exact = exact.plot(color="red", linestyle="--")
            # remove legend
        plt.gca().legend().remove()
        plt.xlim(real_locations[0], real_locations[-1])
        plt.ylabel(r"$\boldsymbol{D}$")
        plt.gca().yaxis.set_label_coords(0.15, d_y_coor)
        # for i != 0, plot the prior and posterior of the advection parameter
        if i == 3:
            plt.legend([l_ci1[0], l_ci1[2], l_ci2[0], l_ci2[2], l_exact[0]], ['mean',  '68% CI ', 'mean', '68% CI', 'exact'], loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            #plt.xlabel("Location ("""+r"$\mu\mathrm{m}$"+")")
            plt.xlabel("Location ("+u"\u03bcm"+")")
        else:
            plt.xlabel("")
            # ticks off
            plt.gca().tick_params(labelbottom=False) 


        plt.sca(axs[i, 3])
        v_min = -3
        v_max = 3
        if i != 0:

            # plot the prior and posterior of the advection parameter

            var_a_sqrt = 0.752**2
            var_a = 2 * var_a_sqrt**2
            prior2 = cuqi.distribution.Gaussian(0, var_a)  # TODO: store

            cuqi.utilities.plot_1D_density(
                prior2, v_min=v_min, v_max=v_max, color="b", label="prior"
            )
            # plt.hist(data_adv_list[i]['x_samples'].samples[-1,:].flatten(), bins=50, alpha=0.5, label='$a$', color='orange')

            kde = sps.gaussian_kde(data_list[i]["x_samples"].samples[-1, :].flatten())
            x = np.linspace(v_min, v_max, 100)
            l1 = plt.plot(x, kde(x), color="black", label="posterior")

            # plot vertical line true value
            true_a = data_list[i]["experiment_par"].true_a
            plt.axvline(true_a, color="r", linestyle="--", label="exact")

        # write ess
        ESS_val = data_list[i]["x_samples"].compute_ess()
        print('ESS (min): '+str(int(np.min(ESS_val))) , 'ESS (mean): '+str(int(np.mean(ESS_val))) , 'ESS (max): '+str(int(np.max(ESS_val)))   )
        plt.ylim(0, 1.1)
        plt.xlim(v_min, v_max)
        if i==3:
            plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            plt.xlabel(r"$a$"+u" (\u03bcm/s)")
        else:
            plt.xlabel("")
            # ticks off
            plt.gca().tick_params(labelbottom=False) 
        plt.ylabel(r"$p(a)$")
        plt.gca().yaxis.set_label_coords(0.17, 0.5)

        # Define function and its inverse
        if i != 0:
            f = lambda x: 0.3*x
            g = lambda x: x/0.3
            ax2 = plt.gca().secondary_xaxis("top", functions=(f, g))
            
        if i == 1:
            #ax2.set_xlabel("Volume flow rate (nl/min)")
            pass
            ax2.tick_params(direction="in", pad=0, colors='blue')
            # add text insted of label
            plt.text(1.7, 0.85, r"$Q$"+"\n(nl/min.)", ha='center', va='center', color='blue') #transform=ax2.transAxes

        
        elif i > 1:
            ax2.tick_params(labeltop=False, direction="in", colors='blue')
            
        # plot the gibbs hyperparameter
        plt.sca(axs[i, 2])
        np.random.seed(0)
        s = cuqi.distribution.Gamma(0.9, 0.5)  # TODO: store
        s_samples = s.sample(100000)
        v_min = 0
        v_max = 2

        # cuqi.utilities.plot_1D_density(s, v_min=v_min, v_max=v_max, color='b',label='prior')
        kde_1 = sps.gaussian_kde(1 / np.sqrt(s_samples.samples.flatten()))
        posterior_sigma_noise_samples = 1 / np.sqrt(data_list[i]["s_samples"].samples.flatten())
        kde_2 = sps.gaussian_kde(posterior_sigma_noise_samples)
        x = np.linspace(v_min, v_max, 100)
        l1 = plt.plot(x, kde_1(x), color="blue", label="prior")
        l2 = plt.plot(x, kde_2(x), color="black", label="posterior")
        # log y scal
        #plt.yscale("log")
        #plt.ylim(1e-15, 10)


        # plot true value
        true_s = float(data_list[i]["experiment_par"].noise_level.split("_")[1])
        plt.axvline(true_s, color="r", linestyle="--", label="exact")
        if i==3:
            plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            plt.xlabel(r"$\sigma_\mathrm{noise}$"+r" (for $c$ gradient)")
        else:
            plt.xlabel("")
            # ticks off
            plt.gca().tick_params(labelbottom=False) 
        plt.ylim(0, 6)
        plt.xlim(v_min, v_max)
        plt.ylabel(r"$p(\sigma_\mathrm{noise})$")
        plt.gca().yaxis.set_label_coords(0.17, 0.5)

        # print the average inferred sigma noise
        inferred_sigma_noise_mean = np.mean(posterior_sigma_noise_samples)
        print("Inferred sigma noise mean: ", inferred_sigma_noise_mean)

        # plot peclet number
        plt.sca(axs[i, 4])
        if i != 0:
            np.random.seed(0)
            #a_prior_samples = prior2.sample(100000)

            samples_diff_avg = np.array([np.average(data_list[i]['x_samples'].samples[:-1,j]**2) for j in range(data_list[i]['x_samples'].Ns)])
            samples_a = data_list[i]["x_samples"].samples[-1, :].flatten()
            samples_peclet = np.zeros_like(samples_diff_avg)
            for j in range(data_list[i]['x_samples'].Ns):
                samples_peclet[j] = peclet_number(a=samples_a[j], d=samples_diff_avg[j], L=real_locations[-1])
            kde_peclet = sps.gaussian_kde(samples_peclet)
            pec_min = -3
            pec_max = 3
            pec_num = 100
            x_peclet = np.linspace(pec_min, pec_max, pec_num)
            l1 = plt.plot(x_peclet, kde_peclet(x_peclet), color="black", label="posterior")
            exact_pec = peclet_number(a=true_a, d=np.average(exact.to_numpy()), L=real_locations[-1])
            plt.axvline(exact_pec, color="r", linestyle="--", label="exact")



            if i==3:
                plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
                plt.xlabel("Pe")
            else:
                plt.xlabel("")
                
                # ticks off
                plt.gca().tick_params(labelbottom=False) 
            plt.xlim(pec_min, pec_max)
            plt.ylabel(r"$p(\text{Pe})$")
            plt.gca().yaxis.set_label_coords(0.17, 0.5)
            plt.ylim(0, 1.5)
            #plt.ylim(0, 0.08)


        


    # remove the upper right plot
    axs[0, 3].axis('off')
    axs[0, 4].axis('off')

    # Add labels for the columns:
    axs[0, 0].set_title("Prediction\n")
    axs[0, 1].set_title("Inferred "+r"$\boldsymbol{D}$"+u" (\u03bcm"+r"$^2$"+"/s)\n")
    axs[0, 2].set_title("Inferred "+r"$\sigma_\mathrm{noise}$"+"\n"+r" (for $c$ gradient)")
    axs[0, 3].set_title("Inferred "+r"$a$"+u" (\u03bcm/s)\n ")
    axs[0, 4].set_title("Inferred "+"Pe\n")

    # Add labels for the rows not using the y label
    row_l_x = -18
    row_l_y = 2000
    plt.sca(axs[0, 0])
    plt.text(row_l_x, row_l_y, "Diffusion only \n" +r"$(a="+str(0)+r")$", fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center') 
    plt.sca(axs[1, 0])
    plt.text(row_l_x, row_l_y,"Advection-\ndiffusion\n" + r"$a="+str(data_list[1]["experiment_par"].true_a)+r"$", fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center')
    plt.sca(axs[2, 0])
    plt.text(row_l_x, row_l_y, "Advection-\ndiffusion\n" +r"$a="+str(data_list[2]["experiment_par"].true_a)+r"$", fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center')
    plt.sca(axs[3, 0])
    plt.text(row_l_x, row_l_y, "Advection-\ndiffusion\n" +r"$a="+str(data_list[3]["experiment_par"].true_a)+r"$", fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center')


def plot_misfit_real( data_diff_list, data_adv_list, y_log=False, colormaps=None, pad_text_insert=0, pad_row_label=0):
    # fig_v = "I" or "II" or "III"
    misfit_info = "case name, data vs. diffusion only, data vs. advection-diffusion, data vs. zero advection and constant diffusion, D_bar, a, peclet\n"            
    # Create a figure with 10 rows and 3 columns. First column is for the
    # data, second column is for the prediction from the diffusion model,
    # and the third column is for the prediction from the advection model.
    num_cases = len(data_diff_list)
    # set matplotlib parameters
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

    y_min = 0
    y_max = 3000
    x_min = 0
    x_max= 300
    clip_on=False
    #plt.rcParams['figure.constrained_layout.use'] = True
    #plt.rcParams['figure.subplot.hspace']= 0.02
    fig =  plt.figure(figsize=(7.2, 1.4*(num_cases))) # exception width is 7.2 instead of 7 (because of white space on the right side)
    
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1], wspace=0.0)
    #fig.subplots_adjust(wspace=0.0)

    axs_left = subfigs[0].subplots(num_cases, 2 , sharex=True, sharey=True)
    # set hspace and wspace for the top subfigure

                                   #, hspace=0.2, wspace=0.4)
    axs_right = subfigs[1].subplots(num_cases, 2, sharex=True, sharey=True)
                                   #, hspace=0.2, wspace=0.4)
    # hspace=0.2, wspace=0.4)
    subfigs[0].subplots_adjust(hspace=0.25, wspace=0.43, right=1, left=0.17)#, bottom

    subfigs[1].subplots_adjust(hspace=0.25, wspace=0.45, right=0.87, left=0.04)#, bottom=0.02)
    

    #fig.subplots_adjust(hspace=0.02)
    #plt.subplots_adjust(hspace=0.0)


           
            
    if True:
        # same as version one but two columns, plot the real data overlaid with the prediction
        #fig, axs = plt.subplots(num_cases, 4, figsize=(7, 1.6*(num_cases)), layout="constrained")

        # increase h space
        #plt.subplots_adjust(wspace=0.3)
        # print(data_adv_list[0].keys())
        # dict_keys(['exact', 'exact_data', 'data', 'mean_recon_data', 'x_samples', 's_samples', 'experiment_par', 'locations', 'times', 'lapse_time', 'num_tree_node_list', 'epsilon_list'])
        for i in range(len(data_diff_list)):
            plot_type = 'over_time'
            colormap = colormaps[0]
            # Plot the data


            #if i == 0:
            #    plt.title("Real data")

            real_times, real_locations, real_data, real_std_data, diff_locations, real_data_diff, real_std_data_diff = read_data_files(data_diff_list[i]['experiment_par'])

            # Update y_min and y_max, and x_max
            y_min = min(y_min, real_data.min())
            y_max = max(y_max, real_data.max())
            x_max = max(x_max, real_locations.max())
            # Plot the prediction from the diffusion model
            #---
            A = create_A(data_diff_list[i]) 
            mean_recon_data = \
                A(data_diff_list[i]["x_samples"].funvals.mean(), is_par=False)
            non_grad_mean_recon_data_diffu = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])


            plt.sca(axs_left[i, 0])
            lines, legends = plot_time_series(real_times, real_locations,
                             non_grad_mean_recon_data_diffu, plot_legend=False, plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
            #switch of ticks and label of y

            

            plot_time_series(real_times, real_locations,
                             real_data.reshape(len(real_locations), len(real_times)), plot_legend=False, marker = '*', linestyle = 'None', plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
            #print(data_diff_list[i]['experiment_par'])
            ear_str = 'left' if data_diff_list[i]['experiment_par'].ear == 'l' else 'right'
            row_l_x = -14
            row_l_y = 3000
            plt.text(row_l_x, row_l_y+pad_row_label, 'Mouse'+data_diff_list[i]['experiment_par'].animal[1]+', '+ear_str, fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center')

            plt.ylabel(r"$\boldsymbol{c}$")
            plt.gca().yaxis.set_label_coords(0.1,0.5)
            #plt.ylabel('Mouse'+data_diff_list[i]['experiment_par'].animal[1]+', '+ear_str)#, horizontalalignment='center', verticalalignment='center')
            # plot legend inside the plot
            # bbox_to_anchor=(1, 0.5)
            if i == 0:
                plt.title("Diffusion\nmodel prediction\n(plotted over time)")
            # plot legend outside the plot to the left
            plt.legend(lines, legends,
                        fontsize=SMALL_SIZE, frameon=False, ncol=1,
                        columnspacing=0.1, handletextpad=0.1, handlelength=0.5,
                        loc='upper right', labelspacing=0.75,
                        bbox_to_anchor=(0.97, 1.07),  mode="expand")
            #---
            # Plot the prediction from the advection model
            #---
            try:
                A = create_A(data_adv_list[i])
                mean_recon_data = \
                    A(data_adv_list[i]["x_samples"].funvals.mean(), is_par=False)
                non_grad_mean_recon_data_adv = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])

                plt.sca(axs_left[i, 1])
                lines, legends = plot_time_series(real_times, real_locations,
                                 non_grad_mean_recon_data_adv, plot_legend=False, plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
                #switch of ticks and label of y
                plot_time_series(real_times, real_locations,
                real_data.reshape(len(real_locations), len(real_times)), plot_legend=False, marker = '*', linestyle = 'None', plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
                plt.ylabel('')

                

                if i == 0:
                    plt.title("Advection-diffusion\nmodel prediction\n(plotted over time)")
                #plt.text(20, 4000, "mean a\n{:.2f}".format(data_adv_list[i]['x_samples'].funvals.mean()[-1]), fontsize=10)
            except:
                pass

            # Plot the difference between the advection and the diffusion model
            #---
            plot_type = 'over_location'
            colormap = colormaps[1]
            # Plot the data


            #if i == 0:
            #    plt.title("Real data")


            # Plot the prediction from the diffusion model
            #---
            A = create_A(data_diff_list[i]) 
            mean_recon_data = \
                A(data_diff_list[i]["x_samples"].funvals.mean(), is_par=False)
            non_grad_mean_recon_data_diffu = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])


            plt.sca(axs_right[i, 0])
            lines, legends = plot_time_series(real_times, real_locations,
                             non_grad_mean_recon_data_diffu, plot_legend=False, plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
            #switch of ticks and label of y
            plt.ylabel('')
            

            plot_time_series(real_times, real_locations,
                             real_data.reshape(len(real_locations), len(real_times)), plot_legend=False, marker = '*', linestyle = 'None', plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
            #print(data_diff_list[i]['experiment_par'])
            plt.ylabel('')
            # legend
            plt.legend(lines, legends,
                       fontsize=SMALL_SIZE, frameon=False, ncol=1,
                       columnspacing=0.1, handletextpad=0.1, handlelength=0.5,
                       loc='upper right',
                        bbox_to_anchor=(0.97, 1.1), mode="expand")              
            if i == 0:
                plt.title("Diffusion\nmodel prediction\n(plotted over location)") 
            #---
            # Plot the prediction from the advection model
            #---
            try:
                A = create_A(data_adv_list[i])
                mean_recon_data = \
                    A(data_adv_list[i]["x_samples"].funvals.mean(), is_par=False)
                non_grad_mean_recon_data_adv = A.pde.interpolate_on_observed_domain(A.pde.solve()[0])

                plt.sca(axs_right[i, 1])
                lines, legends = plot_time_series(real_times, real_locations,
                                 non_grad_mean_recon_data_adv, plot_legend=False, plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
                #switch of ticks and label of y
                plot_time_series(real_times, real_locations,
                real_data.reshape(len(real_locations), len(real_times)), plot_legend=False, marker = '*', linestyle = 'None', plot_type=plot_type, colormap=colormap, y_log=y_log, clip_on=clip_on)
                plt.ylabel('')
        

                if i == 0:
                    plt.title("Advection-diffusion\nmodel prediction\n(plotted over location)")
                    #("Diffusion-advection\n model prediction")
                plt.text(260, 3750+pad_text_insert, r"mean $a=$"+"\n{:.2f}".format(data_adv_list[i]['x_samples'].funvals.mean()[-1])+"\n("+u"\u03bcm"+"/s)", fontsize=8, horizontalalignment='center')
            except:
                pass
            
            # Fill in misfit info
            A_mistfit = create_A(data_diff_list[i]) 
            #_ = A_mistfit(np.ones(21)*420, is_par=False)
            D_bar = np.average(data_diff_list[i]["x_samples"].funvals.mean())
            D_bar_adv = np.average(data_adv_list[i]["x_samples"].funvals.mean()[:-1])
            _ = A_mistfit(np.ones(21)*D_bar, is_par=False)
            non_grad_mean_recon_data_const = A_mistfit.pde.interpolate_on_observed_domain(A_mistfit.pde.solve()[0]) 
            real_data_reshaped = real_data.reshape(len(real_locations), len(real_times))
            a_mean = data_adv_list[i]["x_samples"].funvals.mean()[-1]
            ear_str = 'left  ' if data_diff_list[i]['experiment_par'].ear == 'l' else 'right'
            misfit_info += "mouse"+data_diff_list[i]['experiment_par'].animal[1] + '-' + ear_str + ', ' \
            + str(np.linalg.norm(non_grad_mean_recon_data_diffu-real_data_reshaped)/np.linalg.norm(real_data_reshaped)) + ', ' \
            + str(np.linalg.norm(non_grad_mean_recon_data_adv-real_data_reshaped)/np.linalg.norm(real_data_reshaped)) + ', ' \
            + str(np.linalg.norm(non_grad_mean_recon_data_const-real_data_reshaped)/np.linalg.norm(real_data_reshaped)) +', '\
            + str(D_bar) + ', ' \
            + str(a_mean) + ', '\
            + str(peclet_number(a=a_mean, d=D_bar_adv, L=real_locations[-1])) + '\n' 
            # Plot the difference between the advection and the diffusion model
            #---
        aspect = 1.2 * (x_max - x_min) / (y_max - y_min)
        for i in range(len(data_diff_list)):
            plt.sca(axs_left[i, 0])
            plt.ylim(y_min, y_max)
            #plt.gca().set(aspect=aspect)

            # if not last row, remove x label and ticks
            if i != len(data_diff_list) - 1:
                plt.tick_params(labelbottom=False)
                #plt.xlabel('')
                plt.gca().axes.get_xaxis().get_label().set_visible(False)
            else:
                plt.xlabel("Time (min.)")

            plt.sca(axs_left[i, 1])
            plt.ylim(y_min, y_max)
            #plt.gca().set(aspect=aspect)
            # remove y label and ticks
            plt.tick_params(labelleft=False)
            plt.ylabel('')

            # if not last row, remove x label and ticks
            if i != len(data_diff_list) - 1:
                plt.tick_params(labelbottom=False)
                #plt.xlabel('')
                plt.gca().axes.get_xaxis().get_label().set_visible(False)
            else:
                plt.xlabel("Time (min.)")

            plt.sca(axs_right[i, 0])
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            #plt.gca().set(aspect=aspect)
            # remove y label and ticks
            plt.tick_params(labelleft=False)
            plt.ylabel('')

            # if not last row, remove x label and ticks
            if i != len(data_diff_list) - 1:
                plt.tick_params(labelbottom=False)
                #plt.xlabel('')
                plt.gca().axes.get_xaxis().get_label().set_visible(False)
            else:
                plt.xlabel("Location ("+u"\u03bcm"+")")

            plt.sca(axs_right[i, 1])
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            #plt.gca().set(aspect=aspect)
            # remove y label and ticks labels
            plt.tick_params(labelleft=False)
            plt.ylabel('')

            # if not last row, remove x label and ticks
            if i != len(data_diff_list) - 1:
                plt.tick_params(labelbottom=False)
                #plt.xlabel('')
                # turn off xlabel
                plt.gca().axes.get_xaxis().get_label().set_visible(False)

            else:
                plt.xlabel("Location ("+u"\u03bcm"+")")
    return misfit_info        


def plot_v3_intro_data(data_diff_list, data_adv_list, plot_type='over_time'):
    # plot real data time series for first animal and ear (one row/ one column)
    
    if plot_type == 'over_time' or plot_type == 'over_location':
        fig = plt.figure(figsize=(4, 2))
    else:
        fig = plt.figure(figsize=(4, 4))
    # set left, right, top, bottom to 0.1, 0.9, 0.9, 0.1
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    #plt.sca(plt.subplot(111))

    real_times, real_locations, real_data, real_std_data, diff_locations, real_data_diff, real_std_data_diff = read_data_files(data_diff_list[0]['experiment_par'])
    lines, legend = plot_time_series(real_times, real_locations,
                     real_data.reshape(len(real_locations), len(real_times)), plot_legend=False, plot_type=plot_type, d3_alpha=0.85)
    return lines, legend
    # fix label does not show properly
    #plt.ylabel('Concentration')


def plot_inference_real(data_diff_list, data_adv_list, data_diff_list_all, data_adv_list_all, diff_min=100, diff_max=750, add_last_row=True):
    # create a 10 by 3 plot, each row is for a different animal and ear
    # the first column is for the credibility interval of the inferred
    # diffusion parameter, the second column is for the prior and posterior
    # of the advection parameter, and the third column is for the prior and
    # posterior of the GIBBS parameter
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)
    num_cases = len(data_diff_list) 
    # create 2 subfigures top and bottom with aspec ratio 4:1

    fig =  plt.figure(figsize=(7, (num_cases+1)*(6.5/5)))

    if add_last_row:
        subfigs = fig.subfigures(2, 1, height_ratios=[num_cases, 1])
        axs_top = subfigs[0].subplots(num_cases, 5)
        # set hspace and wspace for the top subfigure
        subfigs[0].subplots_adjust(hspace=0.2, wspace=0.4, bottom=0.2)
                                   #, hspace=0.2, wspace=0.4)
        axs_bottom = subfigs[1].subplots(1, 5)# hspace=0.2, wspace=0.4)
        subfigs[1].subplots_adjust(hspace=0.2, wspace=0.4, top=0.65, bottom=0.06)
        subfigs[1].delaxes(axs_bottom[-1])
    else:
        subfigs = fig.subfigures(1, 1)
        axs_top = subfigs.subplots(num_cases, 5)
        # set hspace and wspace for the top subfigure
        subfigs.subplots_adjust(hspace=0.2, wspace=0.4, bottom=0.2)


    #fig, axs = plt.subplots(num_cases+2, 5, figsize=(7, (num_cases+2)*(6.5/5)))
    #fig.subplots_adjust(wspace=0.4, hspace=0.2)
    # determine loc max: 
    geom_list = [data["x_samples"].geometry for data in data_diff_list]
    loc_max = max([geom.grid[-1] for geom in geom_list])

    #loc_max = 360
    row_l_x = -180
    row_l_y = 400
    pec_min = -3
    pec_max = 3
    pec_num = 100


    for i in range(len(data_diff_list)):
        # Plot the credibility interval of the inferred diffusion parameter
        plt.sca(axs_top[i, 0])
        l_ci1 = data_diff_list[i]['x_samples'].funvals.plot_ci(68)
        
        try:
            l_ci2 = cuqi.samples.Samples(data_adv_list[i]['x_samples'].samples[:-1,:], geometry=data_diff_list[i]['x_samples'].geometry).funvals.plot_ci( 68, plot_envelope_kwargs={'facecolor': 'gray', 'edgecolor': 'gray'}, color='gray')
            # plot line of level 368 (thin line)
            l_ref = plt.plot([0, geom_list[i].grid[-1]], [368, 368], color='black', linestyle='--', linewidth=0.5)
        except:
            pass
        # plot legend before first column
        legend_x = 0.5
        legend_y = -.5
        if i == num_cases-1:
            plt.legend([l_ci1[0], l_ci1[2], l_ci2[0], l_ci2[2], l_ref[0]], ['mean (diffusion only)',  '68% CI (diffusion only)', 'mean (advection-diffusion)', '68% CI (advection-diffusion)', r'$\mathrm{D}_\mathrm{E}\approx 368$'+" ("+"\u03bcm"+r"$^2$"+"/s"+".)"], loc="upper center", ncol=1, frameon=False)
            plt.gca().legend_.set_bbox_to_anchor((legend_x, legend_y))
            #plt.legend([l_ci1[0], l_ci1[2], l_ci2[0], l_ci2[2]], ['mean (Diff.)',  '68% CI (Diff.)', 'mean (Adv.-Diff.)', '68% CI (Adv.-Diff.)'], loc='center left', bbox_to_anchor=(-.2, -1.7), ncol=4)
        else:
            # set legend to off
            plt.legend().set_visible(False)

        # compute ESS min
        ESS_diff = np.min(data_diff_list[i]['x_samples'].compute_ess())
                # write ess
        print('ESS_diff (min): '+str(int(np.min(ESS_diff))) , 'ESS (mean): '+str(int(np.mean(ESS_diff))) , 'ESS (max): '+str(int(np.max(ESS_diff)))   )
        try:
            ESS_adv = np.min(data_adv_list[i]['x_samples'].compute_ess())
            print('ESS_adv (min): '+str(int(np.min(ESS_adv))) , 'ESS (mean): '+str(int(np.mean(ESS_adv))) , 'ESS (max): '+str(int(np.max(ESS_adv)))   )
        except:
            ESS_adv=0
            pass
        plt.ylim(diff_min, diff_max)
        # add mouse number and ear in the y label
        plt.ylabel(r"$\boldsymbol{D}$")
        plt.gca().yaxis.set_label_coords(0.17, 0.9)


        ear_str = 'left' if data_diff_list[i]['experiment_par'].ear == 'l' else 'right'

        plt.text(row_l_x, row_l_y, 'Mouse'+data_diff_list[i]['experiment_par'].animal[1]+', '+ear_str, fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center') 

        plt.xlim(0, loc_max)
        if i==num_cases-1:
            plt.xlabel(u"Location (\u03bcm)")
        else:
            plt.xlabel("")
            # keep ticks but remove ticks labels for x 
            plt.gca().tick_params(labelbottom=False)

        if ESS_adv != 0:
            # plot the prior and posterior of the advection parameter
            plt.sca(axs_top[i, 2])
            var_a_sqrt = 0.752**2
            var_a = 2*var_a_sqrt**2
            prior2 =cuqi.distribution.Gaussian(0, var_a) #TODO: store
            v_min = -3
            v_max = 3
            cuqi.utilities.plot_1D_density(prior2, v_min=v_min, v_max=v_max, color='b',label='prior')
            #plt.hist(data_adv_list[i]['x_samples'].samples[-1,:].flatten(), bins=50, alpha=0.5, label='$a$', color='orange')

            
            kde = sps.gaussian_kde(data_adv_list[i]['x_samples'].samples[-1,:].flatten())
            x = np.linspace(v_min, v_max, 100)
            l1 = plt.plot(x, kde(x), color='black', label='posterior')
            if i == num_cases-1:
                plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)

            #if i==0:
            #    plt.title("Adv. parameter inference") 

            plt.xlim(v_min, v_max)


            plt.ylabel(r"$p(a)$")
            plt.gca().yaxis.set_label_coords(0.18, 0.5)
            plt.ylim(0, 1)


            f = lambda x: 0.3*x
            g = lambda x: x/0.3
            ax2 = plt.gca().secondary_xaxis("top", functions=(f, g))
                
            if i == 0:
                #ax2.set_xlabel("Volume flow rate (nl/min)")
                pass
                ax2.tick_params(direction="in", pad=1, colors='blue')
                # add text insted of label
                plt.text(-1.45, 0.8, r"$Q$"+"\n(nl/min.)", ha='center', va='center', color='blue') #transform=ax2.transAxes

            else:
                ax2.tick_params(labeltop=False, direction="in", colors='blue')

            if i==num_cases-1:
                plt.xlabel(r"$a$"+u" (\u03bcm/s)")
            else:
                plt.xlabel("")
                # keep ticks but remove ticks labels for x 
                plt.gca().tick_params(labelbottom=False)
            # plot the gibbs
            plt.sca(axs_top[i, 1])
            s = cuqi.distribution.Gamma(1.2, 5) #TODO: store
            s_samples = s.sample(10000)
            v_min2 = -1
            v_max2 = 10
            #if i == 0:  
            #    plt.title("Noise level inference")
                
            #cuqi.utilities.plot_1D_density(s, v_min=v_min, v_max=v_max, color='b',label='prior')
            kde_1 = sps.gaussian_kde(1/np.sqrt(s_samples.samples.flatten()))
            posterior_sigma_noise_samples_adv = 1/np.sqrt(data_adv_list[i]['s_samples'].samples.flatten()) 
            kde_2 = sps.gaussian_kde(posterior_sigma_noise_samples_adv)
            x2 = np.linspace(v_min2, v_max2, 100)
            l1 = plt.plot(x2, kde_1(x2), color='blue', label='prior')
            l2 = plt.plot(x2, kde_2(x2), color='black', label='posterior')
            
            posterior_sigma_noise_samples_diff = 1/np.sqrt(data_diff_list[i]['s_samples'].samples.flatten()) 
            inferred_sigma_noise_mean_diff = np.mean(posterior_sigma_noise_samples_diff)
            print("Inferred sigma noise mean (diff): ", inferred_sigma_noise_mean_diff)

            inferred_sigma_noise_mean_adv = np.mean(posterior_sigma_noise_samples_adv)
            print("Inferred sigma noise mean (adv): ", inferred_sigma_noise_mean_adv)

            #plt.legend()
            if i==num_cases-1:
                plt.xlabel(r"$\sigma_\mathrm{noise}$"+r" (for $c$ gradient)")
                plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            else:
                plt.xlabel("")
                # keep ticks but remove ticks labels for x 
                plt.gca().tick_params(labelbottom=False)
            plt.xlim(v_min2, v_max2)


            plt.ylabel(r"$p(\sigma_\mathrm{noise})$")
            plt.gca().yaxis.set_label_coords(0.20, 0.55)
            plt.ylim(0, 1.5)

            # plot peclet number
            plt.sca(axs_top[i, 3])
            np.random.seed(0)
            #a_prior_samples = prior2.sample(100000)

            samples_avg_diff = np.array([np.average(data_adv_list[i]['x_samples'].samples[:-1,j]**2) for j in range(data_adv_list[i]['x_samples'].Ns)])
            samples_a = data_adv_list[i]["x_samples"].samples[-1, :].flatten()
            samples_peclet = np.zeros_like(samples_avg_diff)
            for j in range(data_adv_list[i]['x_samples'].Ns):
                samples_peclet[j] = peclet_number(a=samples_a[j], d=samples_avg_diff[j], L=data_diff_list[i]["x_samples"].geometry.grid[-1])
            kde_peclet = sps.gaussian_kde(samples_peclet)
            x_peclet = np.linspace(pec_min, pec_max, pec_num)
            l1 = plt.plot(x_peclet, kde_peclet(x_peclet), color="black", label="posterior")



            if i==num_cases-1:
                plt.xlabel("Pe")
                plt.legend(loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False)
            else:
                plt.xlabel("")
                plt.gca().tick_params(labelbottom=False) 

                # ticks off
            
            plt.xlim(pec_min, pec_max)
            #plt.ylim(0, 1.4)
            plt.ylabel(r"$p(\text{Pe})$")
            plt.gca().yaxis.set_label_coords(0.21, 0.5)
            plt.ylim(0, 1.2)
##



            # plot scatter plot of mean diffusion and advection
            plt.sca(axs_top[i, 4])
                # samples of average diffusion
            samples_avg_diff = np.array([np.average(data_adv_list[i]['x_samples'].samples[:-1,j]**2) for j in range(data_adv_list[i]['x_samples'].Ns)])
            # stack advection and diffusion
            diff_adv = cuqi.samples.Samples(np.vstack(
                                    (samples_avg_diff, data_adv_list[i]['x_samples'].samples[-1,:])))
            diff_adv.geometry =  cuqi.geometry.Discrete(['$c^2_\mathrm{avg}$', r"$a$"+" ("+u"\u03bcm"+"/s)"])

            # plot the correlation
            color_list = ['black']*num_cases
            diff_adv.funvals.plot_pair(ax=plt.gca(), scatter_kwargs={'alpha':0.5, 'color':color_list[i], 's':7})
            plt.xlim(v_min, v_max)
            plt.ylim(diff_min, diff_max)
            if i != num_cases-1:
                plt.xlabel("")
                # keep ticks but remove ticks labels for x 
                plt.gca().tick_params(labelbottom=False)
            else:
                plt.legend(['posterior'], loc="upper center", bbox_to_anchor=(legend_x, legend_y), ncol=1, frameon=False) 

            plt.ylabel(r"$\bar \boldsymbol{D}$")
            plt.gca().yaxis.set_label_coords(0.21, 0.2)



    color_map_string = 'coolwarm' #'twilight' #'terrain'#'coolwarm'#'berlin' #'brg'
    skip_middle = False
    colormap=plt.colormaps.get_cmap(color_map_string)
    # create color list for the len(data_diff_list_all) lines but skip the white in the middle
    if skip_middle:
        color_list = [colormap(i) for i in np.linspace(0, 1, len(data_diff_list_all)+4)]
        color_list = color_list[:4]+color_list[5:] 
    else:
        color_list = [colormap(i) for i in np.linspace(0, 1, len(data_diff_list_all))]
    
    if add_last_row:
        lines_list = []
        labels_list = []
        all_samples_a = []
        for i in range(len(data_diff_list_all)):
            labels_list.append(data_diff_list_all[i]['experiment_par'].animal+', '+data_diff_list_all[i]['experiment_par'].ear)
            plt.sca(axs_bottom[0])
            # plot all the means
            l_ci4 = cuqi.samples.Samples(data_adv_list_all[i]['x_samples'].samples[:-1,:], geometry=data_diff_list_all[i]['x_samples'].geometry).plot_mean(color=color_list[i])
            plt.title("")
            plt.ylim(diff_min, diff_max)
    
            plt.xlim(0, loc_max)
            plt.xlabel("Location ("+u"\u03bcm"+")")
            plt.ylabel(r"$\boldsymbol{D}$")
            plt.gca().yaxis.set_label_coords(0.17, 0.9)
    
            plt.sca(axs_bottom[2])
            # plot all the posteriors and a prior of the advection parameter
            if i == 0:
                cuqi.utilities.plot_1D_density(prior2, v_min=v_min, v_max=v_max, color='b',label='prior')
                        
            kde = sps.gaussian_kde(data_adv_list_all[i]['x_samples'].samples[-1,:].flatten())
            x = np.linspace(v_min, v_max, 100)
    
            l1 = plt.plot(x, kde(x), color=color_list[i],
                          label='posterior')
            plt.xlim(v_min, v_max)
            plt.xlabel(r"$a$"+" ("+u"\u03bcm"+"/s)")
    
            plt.ylabel(r"$p(a)$")
            plt.gca().yaxis.set_label_coords(0.18, 0.5) 
    
            plt.sca(axs_bottom[1])
            # plot all the posteriors and a prior of the gibbs parameter
            if i == 0:
                l1 = plt.plot(x2, kde_1(x2), color='blue', label='prior')
            kde_2 = sps.gaussian_kde(1/np.sqrt(data_adv_list_all[i]['s_samples'].samples.flatten()))
            x2 = np.linspace(v_min2, v_max2, 100)
            l1 = plt.plot(x2, kde_2(x2), color=color_list[i], label='posterior')
            plt.xlim(v_min2, v_max2)
            plt.xlabel(r"$\sigma_\mathrm{noise}$"+r" (for $c$ gradient)")
    
            plt.ylabel(r"$p(\sigma_\mathrm{noise})$")
            plt.gca().yaxis.set_label_coords(0.20, 0.5)
    
            plt.sca(axs_bottom[ 3])
            # plot peclet number
            np.random.seed(0)
            #a_prior_samples = prior2.sample(100000)
    
            samples_avg_diff = np.array([np.average(data_adv_list_all[i]['x_samples'].samples[:-1,j]**2) for j in range(data_adv_list_all[i]['x_samples'].Ns)])
            samples_a = data_adv_list_all[i]["x_samples"].samples[-1, :].flatten()
            all_samples_a.append(samples_a)
            samples_peclet = np.zeros_like(samples_avg_diff)
            for j in range(data_adv_list_all[i]['x_samples'].Ns):
                samples_peclet[j] = peclet_number(a=samples_a[j], d=samples_avg_diff[j], L=data_diff_list_all[i]["x_samples"].geometry.grid[-1])
            kde_peclet = sps.gaussian_kde(samples_peclet)
            x_peclet = np.linspace(pec_min, pec_max, pec_num)
            l1 = plt.plot(x_peclet, kde_peclet(x_peclet), color=color_list[i], label="posterior")
            lines_list.append(l1[0])
            plt.xlabel("Pe")
    
            plt.xlim(pec_min, pec_max)
            #plt.ylim(0, 1.4)
            plt.ylabel(r"$p(\text{Pe})$")
            plt.gca().yaxis.set_label_coords(0.21, 0.5)
            #plt.legend(lines_list, labels_list, loc="upper right", bbox_to_anchor=(2.8, 0.9), ncol=2, frameon=False)
            plt.legend(lines_list, labels_list, loc="upper right", bbox_to_anchor=(2.3, 1.35), ncol=1, frameon=False)
    
    

    # remove axs[4, 4]

    #fig.delaxes(axs[num_cases, 3])

    # Add labels for the columns:
    axs_top[0, 0].set_title("Inferred "+r"$\boldsymbol{D}$"+" ("+u"\u03bcm"+r"$^2$"+"/s)\n ")
    axs_top[0, 1].set_title("Inferred "+r"$\sigma_\mathrm{noise}$"+"\n"+r" (for $c$ gradient)") # \rho()
    axs_top[0, 2].set_title("Inferred "+r"$a$" +" ("+u"\u03bcm"+"/s)"+"",pad=17)
    axs_top[0, 3].set_title("Inferred "+"Pe\n ")
    axs_top[0, 4].set_title("Correlation \nplot")

    if add_last_row:
        # Add labels for the rows not using the y label
        plt.sca(axs_bottom[0])
        plt.text(row_l_x, row_l_y, "All cases", fontsize=BIGGER_SIZE, rotation=90, va='center', ha='center')

        return all_samples_a
    else:
        return None