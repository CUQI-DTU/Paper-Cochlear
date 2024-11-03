import argparse
import json
import pandas as pd
import numpy as np
from cuqi.geometry import MappedGeometry, Discrete, Continuous1D
from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, GMRF
from cuqi.sampler import MH, NUTS
import matplotlib.pyplot as plt
from custom_distribution import MyDistribution
from scipy.interpolate import interp1d
from cuqi.experimental.mcmc import (HybridGibbs as HybridGibbsNew,
                                    NUTS as NUTSNew,
                                    Conjugate as ConjugateNew)
import cuqi

try:
    import dill as pickle
except:
    # append local dill path in cluster
    import sys
    sys.path.append('../../../../tools')
    import dill as pickle

# Print dill version
print('dill version: ', pickle.__version__)

#Arg class
class Args:
    def __init__(self):
        self.animal = 'm1'
        self.ear = 'l'
        self.version = 'v_temp'
        self.sampler = 'MH'
        self.unknown_par_type = 'constant'
        self.unknown_par_value = [100.0]
        self.data_type = 'syntheticFromDiffusion'
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
        self.adaptive = True

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
                            'real', 'syntheticFromDiffusion', 'syntheticFromAdvectionDiffusion'],
                        default=arg_obj.data_type,
                        help='Type of data, real or synthetic')
    #TODO: syntheticFromAdvectionDiffusion is not used, however, syntheticDiffusion work for both
    # cases. Maybe you need to combine the two cases in one.
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
                        help='Noise level for data, set to "fromDataVar" to read noise level from data that varies for each data point and set to "fromDataAvg" to compute average noise level from data and use it for all data points, set to "avgOverTime" to compute average noise level over time for each location, set to a float representing the noise level')
    parser.add_argument('-add_data_pts', metavar='add_data_pts',
                        nargs='*',
                        type=float,
                        default=arg_obj.add_data_pts)
    # number of CA points used when -data_pts_type is CA
    parser.add_argument('-num_CA', metavar='num_CA', type=int,
                        choices=range(6),
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
    parser.add_argument('-rbc', metavar='rbc', type=str, choices=['zero', 'fromData'],
                        default=arg_obj.rbc,
                        help='right boundary condition')
    parser.add_argument('-adaptive', metavar='adaptive', type=bool,
                        default=arg_obj.adaptive,
                        help='static adaptive time step size, fine at the beginning'+\
                        'and coarse at the end, default is True')
    parser.add_argument('-NUTS_kwargs', metavar='NUTS_kwargs', type=str,
                        default=arg_obj.NUTS_kwargs,
                        help='kwargs for NUTS sampler')
    
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
    CA_list = ['CA'+str(i+1) for i in range(args.num_CA)]

    if args.num_ST == 0: # Only CA data
        print('CA data.')
        ## Read distance file
        dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
        real_locations = dist_file['distance microns'].values[:args.num_CA]
        
        ## Read concentration file and times
        constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        real_data = constr_file[CA_list].values.T.ravel()

        ## Read std data
        CA_std_list = [item+' std' for item in CA_list]
        real_std_data = constr_file[CA_std_list].values.T.ravel()
  
    elif args.num_ST > 0: # CA and ST data
        print('CA and ST data.')

        ## Read distance file
        dist_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
        # locations distance microns where 20210120_omnip10um_KX_M1_nosound_L is in
        # ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8']
        real_locations = dist_file['distance'].values
        real_locations = real_locations[:args.num_CA+args.num_ST]
        ST_list = ['ST'+str(i+1) for i in range(args.num_ST)]
        CA_ST_list = CA_list + ST_list
    
        ## Read concentration file and times
        constr_file = pd.read_csv('../../data/parsed/CT/combined_CA_ST/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
        real_times = constr_file['time'].values*60
        real_data = constr_file[CA_ST_list].values.T.ravel()
        ## Read std data
        std_file = pd.read_csv('../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
        CA_ST_std_list = [item+' std' for item in CA_ST_list]
        real_std_data = std_file[CA_ST_std_list].values.T.ravel()

    return real_times, real_locations, real_data, real_std_data

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
        def PDE_form(c, tau_current):
            return (D_c_const(c), g_const(c, tau_current), initial_condition)
        
    elif inference_type == 'heterogeneous':
        ## Source term (varying in space diffusion coefficient case)
        def g_var(c, tau_current):
            f_array = np.zeros(n_grid)
            f_array[0] = c[0]/h**2*np.interp(tau_current, times, real_bc_l)
            if real_bc_r is not None:
              f_array[-1] = c[-1]/h**2*np.interp(tau_current, times, real_bc_r)
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

    elif inference_type == 'advection_diffusion':
        ## Source term (varying in space diffusion coefficient case)
        def g_var(c, tau_current):
            f_array = np.zeros(n_grid)
            u_0_mplus1 = np.interp(tau_current, times, real_bc_l) 
            f_array[0] += u_0_mplus1*c[0]/h**2 + c[-1]*u_0_mplus1/(2*h)
            if real_bc_r is not None:
                u_L_m = np.interp(tau_current, times, real_bc_r)
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
        def PDE_form(x, tau_current):
            c = np.interp(grid_c_fine, grid_c, x[:-1])
            return (D_c_var(c) - DA_a(x[-1]), g_var(x, tau_current), initial_condition)   

    return PDE_form

def create_prior_distribution(G_c, inference_type):
    """Function to create prior distribution. """
    if inference_type == 'constant':
        prior = Gaussian(np.sqrt(400), 100, geometry=G_c)
    elif inference_type == 'heterogeneous':
        prior = GMRF(
            np.ones(G_c.par_dim)*np.sqrt(300),
            0.2,
            geometry=G_c,
            bc_type='neumann')
    elif inference_type == 'advection_diffusion':
        prior1 = GMRF(np.ones(G_c.par_dim-1)*np.sqrt(300),
            0.2,
            bc_type='neumann')
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
            a = np.sqrt(a)

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
            a = np.sqrt(a)

    ## append "a" value to the end
    if a is not None and unknown_par_type != 'constant':
        exact_x = np.append(exact_x, a)
    exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=is_par)
    exact_data = A(exact_x)
    return exact_x, exact_data

def set_the_noise_std(
        data_type, noise_level, exact_data,
        real_data, real_std_data, G_cont2D):
    """Function to set the noise standard deviation. """
    # Use noise levels read from the file
    if noise_level == "fromDataVar":
        ## Noise standard deviation
        s_noise = real_std_data
    # Use noise level specified in the command line
    elif noise_level == "fromDataAvg":
        s_noise = np.mean(real_std_data)

    elif noise_level == "avgOverTime":
        s_noise = real_std_data.reshape(G_cont2D.fun_shape)
        s_noise = np.mean(s_noise, axis=1)
        s_noise = np.repeat(s_noise, G_cont2D.fun_shape[1])

    else:
        try:
            noise_level = float(noise_level)
        except:
            raise Exception('Noise level not supported')
        ## Noise standard deviation 
        if data_type == 'syntheticFromDiffusion':
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

def sample_the_posterior(sampler, posterior, G_c, args):
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
        my_sampler = MH(posterior, scale=10, x0=x0)
        posterior_samples = my_sampler.sample_adapt(Ns)
        posterior_samples_burnthin = posterior_samples.burnthin(Nb)
    elif sampler == 'NUTS':
        posterior.enable_FD()
        NUTS_kwargs = args.NUTS_kwargs
        my_sampler = NUTS(posterior, x0=x0, **NUTS_kwargs)
        posterior_samples = my_sampler.sample_adapt(Ns, Nb) 
        posterior_samples_burnthin = posterior_samples
    elif sampler == 'NUTSWithGibbs':
        sampling_strategy = {
            "x" : NUTSNew(initial_point=x0, **args.NUTS_kwargs),
            "s" : ConjugateNew()
        }
        
        # Here we do 1 internal steps with NUTS for each Gibbs step
        num_sampling_steps = {
            "x" : 1,
            "s" : 1
        }
        
        my_sampler = HybridGibbsNew(posterior, sampling_strategy, num_sampling_steps)
        my_sampler.warmup(Nb)
        my_sampler.sample(Ns)
        posterior_samples = my_sampler.get_samples()
        posterior_samples_burnthin = posterior_samples

    else:
        raise Exception('Unsuppported sampler')
    
    return posterior_samples_burnthin, my_sampler

def plot_time_series(times, locations, data, plot_legend=True):
    # Plot data
    color = ['r', 'g', 'b', 'k', 'm', 'c']
    legends = ['loc = '+"{:.2f}".format(obs) for obs in locations]
    lines = []
    for i in range(len(locations)):
        lines.append(plt.plot(times/60, data[i,:],  color=color[i%len(color)])[0])
    
    if plot_legend:
        plt.legend(lines, legends)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')

    return lines, legends

def save_experiment_data(dir_name, exact, exact_data, data, mean_recon_data,
                    x_samples, s_samples, experiment_par, locations, times, lapse_time, sampler):
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
                 'lapse_time': lapse_time,
                 'num_tree_node_list': None,
                 'epsilon_list': None}
    # if sampler is NUTs, save the number of tree nodes
    if isinstance(sampler, cuqi.experimental.mcmc.NUTS):
        data_dict['num_tree_node_list'] = sampler.num_tree_node_list
        data_dict['epsilon_list'] = sampler.epsilon_list
    
    # if sampler is HybridGibbs, save the number of tree nodes if the
    # underlying sampler is NUTS
    elif isinstance(sampler, cuqi.experimental.mcmc.HybridGibbs):
        if isinstance(sampler.samplers['x'], cuqi.experimental.mcmc.NUTS):
            data_dict['num_tree_node_list'] = sampler.samplers['x'].num_tree_node_list
            data_dict['epsilon_list'] = sampler.samplers['x'].epsilon_list

    with open(dir_name +'/'+tag+'_'+name_str+'.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

def read_experiment_data(dir_name, tag):
    # Read data from pickle file
    file_name = dir_name +'/output'+ tag+'/'+tag+'_var.pkl'
    print('Reading file: ', file_name)
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
                    x_samples, s_samples, experiment_par, locations, times, lapsed_time=None, L=None):
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
    fig = plt.figure(figsize=(12, 18), layout='constrained')

    subfigs = fig.subfigures(4, 1, height_ratios=height_ratios)

    axsSecond = subfigs[1].subplots(4, 2,
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

    # Plot trace
    x_samples.plot_trace(trace_idx_list, axes=axesThird)

    # write lapse time, exact a , exact peclet number, and mean peclet number
    # in the last subfigure
    axesLast[0].axis('off')
    axesLast[0].text(0.1, 0.8, 'Lapse time: {:.2f} sec'.format(lapsed_time))
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


def create_args_list(animals, ears, noise_levels, num_ST_list, add_data_pts_list, unknown_par_types, unknown_par_values, data_type, version, samplers, Ns_s, Nb_s, inference_type_s=['heterogeneous'], true_a_s=None, rbc_s=None, NUTS_kwargs = None):
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