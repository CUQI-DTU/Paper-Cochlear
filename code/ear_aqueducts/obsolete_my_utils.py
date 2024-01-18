import numpy as np
import matplotlib.pyplot as plt
from cuqi.array import CUQIarray
try:
    import dill as pickle
except:
    # append local dill path in cluster
    import sys
    sys.path.append('../../../../tools')
    import dill as pickle

# Print dill version
print('dill version: ', pickle.__version__)

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
    const = True if samples.geometry.par_dim == 1 else False

    if const:
        name_str = 'const'
    else:
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

def read_experiment_data(dir_name, tag, const=True):
    # Read data from pickle file
    if const:
        name_str = 'const'
    else:
        name_str = 'var'
    with open(dir_name +'/output'+ tag+'/'+tag+'_'+name_str+'.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # Convert exact to CUQIarray with geometry
    exact = CUQIarray(data_dict['exact'], 
                      geometry=data_dict['exact_geometry'],
                      is_par=data_dict['exact_is_par'])
    data_dict['exact'] = exact
    # drop geometry and is_par
    data_dict.pop('exact_geometry')
    data_dict.pop('exact_is_par')

    # Convert exact_data to CUQIarray with geometry
    exact_data = CUQIarray(data_dict['exact_data'], 
                           geometry=data_dict['exact_data_geometry'],
                           is_par=data_dict['exact_data_is_par'])
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
    exact_for_plot = False
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
    if experiment_par.inference_type not in ['both']:
        raise Exception('Inference type not supported')

def create_experiment_tag(experiment_par):
    """Method to create a tag from the parameters of the experiment. """
    # Create directory for output
    version = experiment_par.version
    if len(experiment_par.unknown_par_value) == 1:
        unknown_par_value_str = str(experiment_par.unknown_par_value[0])
    elif len(experiment_par.unknown_par_value) == 2:
        unknown_par_value_str = str(experiment_par.unknown_par_value[0])+\
            '_'+str(experiment_par.unknown_par_value[1])

    data_pt_str = str(experiment_par.add_data_pts[0]) if len(experiment_par.add_data_pts) > 0 else ''
    # Create directory for output
    tag = experiment_par.animal+'_'+experiment_par.ear+'_'+\
        experiment_par.sampler+'_'+experiment_par.unknown_par_type+'_'+\
        unknown_par_value_str+'_'+experiment_par.data_type+'_'+\
        experiment_par.inference_type+'_'+\
        str(experiment_par.Ns_const)+'_'+str(experiment_par.Ns_var)+'_'+\
        str(experiment_par.noise_level)+'_'+\
        version+'_'+\
        data_pt_str+'_'+\
        str(experiment_par.data_pts_type)
    
    return tag

def matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
