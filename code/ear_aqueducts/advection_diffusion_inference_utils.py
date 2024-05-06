import argparse
import pandas as pd
import numpy as np
from cuqi.geometry import MappedGeometry, Discrete, Continuous1D
from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, GMRF
from cuqi.sampler import MH, NUTS
import matplotlib.pyplot as plt
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
                        'CWMH', 'MH', 'NUTS'],
                        default=arg_obj.sampler,
                        help='the sampler to use')
    parser.add_argument('-unknown_par_type',
                        metavar='unknown_par_type',
                        type=str, choices=['constant',
                                           'smooth',
                                           'step',
                                           'sampleMean',
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

def create_time_steps(h, cfl, tau_max):
    """Function to create time steps array. """
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)
    return tau

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
            np.ones(G_c.par_dim)*np.sqrt(300),
            0.2,
            geometry=G_c,
            bc_type='neumann')
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
        is_par = False

    # if the unknown parameter is varying in space (step function)
    elif unknown_par_type == 'step':
        exact_x = np.zeros(n_grid_c)
        exact_x[0:n_grid_c//2] = unknown_par_value[0]
        exact_x[n_grid_c//2:] = unknown_par_value[1]
        is_par = False

    # if the unknown parameter is varying in space (smooth function)
    elif unknown_par_type == 'smooth':
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
    else:
        raise Exception('Unsuppported sampler')
    
    return posterior_samples_burnthin

def plot_time_series(times, locations, data):

    # Plot data
    color = ['r', 'g', 'b', 'k', 'm', 'c']
    legends = ['loc = '+"{:.2f}".format(obs) for obs in locations]
    lines = []
    for i in range(len(locations)):
        lines.append(plt.plot(times/60, data[i,:],  color=color[i%len(color)])[0])
    
    plt.legend(lines, legends)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')

def save_experiment_data(dir_name, exact, exact_data, data, mean_recon_data,
                    samples, experiment_par, locations, times):
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
    if exact:
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
    if exact_data:
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
                 'mean_recon_data': mean_recon_data, 'samples': samples,
                 'experiment_par': experiment_par, 'locations': locations,
                 'times': times}

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
        exact_data = CUQIarray(data_dict['exact_data'], 
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
                    samples, experiment_par, locations, times):
    """Method to plot the numerical experiment results."""
    # Create tag
    tag = create_experiment_tag(experiment_par)

    # Expr type (const or var)
    const_inf = True if samples.geometry.par_dim == 1 else False
    const_true_x = True 
    if exact is not None:
        const_true_x = True if exact.geometry.par_dim == 1 else False

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
    height_ratios = [1, 1] if const_inf else [1, 1]
    # Trace index list
    trace_idx_list = [0] if const_inf else [0, 5, 15]
    # Marker
    marker = 'o' if const_true_x else ''

    # Create figure: 
    fig = plt.figure(figsize=(12, 14), layout='constrained')

    subfigs = fig.subfigures(2, 1, height_ratios=height_ratios)

    axsTop = subfigs[0].subplots(3, 2,
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.1, top=0.9,
                         hspace=0.5, wspace=0.5))
    axsBottom = subfigs[1].subplots(axsBottom_rows, 2,
        gridspec_kw=dict(left=0.1, right=0.9,
                         bottom=0.1, top=0.96,
                         hspace=0.5, wspace=0.5))

    # Add super title
    subfigs[0].suptitle('Experiment results: '+tag)

    # Plot exact data
    if exact_data is not None:
        plt.sca(axsTop[0, 0])
        plot_time_series(times, locations, exact_data)
        plt.title('Exact data')

    # Plot data
    plt.sca(axsTop[0, 1])
    plot_time_series(times, locations, data)
    plt.title('Data')

    # Plot reconstructed data
    plt.sca(axsTop[1, 0])
    plot_time_series(times, locations, mean_recon_data)
    plt.title('Mean reconstructed data')

    # Plot cridible intervals
    plt.sca(axsTop[1, 1])
    samples.funvals.plot_ci(exact = exact_for_plot)
    # If inference type is not constant, plot data locations as vertical lines
    if not const_inf:
        for loc in locations:
            plt.axvline(x = loc, color = 'gray', linestyle = '--')
    plt.title('Posterior samples CI')

    # Plot ESS
    plt.sca(axsTop[2, 0])
    ESS_list = np.array(samples.compute_ess()) 
    plt.plot(ESS_list, marker=marker)
    plt.title('ESS')

    # Plot exact   
    if exact is not None:
        plt.sca(axsTop[2, 1])
        exact.plot(marker=marker) 
        plt.title('Exact solution')

    # Plot trace
    samples.plot_trace(trace_idx_list, axes=axsBottom)

    return fig

def process_experiment_par(experiment_par):
    """Method to create a tag from the parameters of the experiment. """
    # Assert if real data, you cannot add data points
    if experiment_par.data_type == 'real' and len(experiment_par.add_data_pts) > 0:
        raise Exception('Cannot add data points to real data')
    
    if len(experiment_par.unknown_par_value) not in [1, 2]:
        raise Exception('Unknown parameter value not supported')
    
    # Raise exception if more than one data point is added, unable to
    # create tag
    if len(experiment_par.add_data_pts) > 1:
        raise Exception('Only one data point can be added')
    
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
    #elif isinstance(experiment_par.unknown_par_value, str):
    #    unknown_par_value_str = experiment_par.unknown_par_value.split('@')[0]
    else:
        raise Exception('Unknown parameter value not supported')

    data_pt_str = str(experiment_par.add_data_pts[0]) if len(experiment_par.add_data_pts) > 0 else ''
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
        str(experiment_par.num_CA)
    
    return tag

def matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
