import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
#import pickle

def plot_time_series(times, locations, data):

    # Plot data
    color = ['r', 'g', 'b', 'k', 'm', 'c']
    legends = ['loc = '+"{:.2f}".format(obs) for obs in locations]
    lines = []
    for i in range(len(locations)):
        lines.append(plt.plot(times/60, data[i,:],  color=color[i])[0])
    
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


    # Save data in pickle file named with tag
    tag = create_experiment_tag(experiment_par)
    data_dict = {'exact': exact, 'exact_data': exact_data, 'data': data,
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
    return data_dict['exact'], data_dict['exact_data'], data_dict['data'],\
        data_dict['mean_recon_data'], data_dict['samples'],\
        data_dict['experiment_par'], data_dict['locations'], data_dict['times']

def plot_experiment(exact, exact_data, data, mean_recon_data,
                    samples, experiment_par, locations, times):
    """Method to plot the numerical experiment results."""
    # Create tag 
    tag = create_experiment_tag(experiment_par)

    # Expr type (const or var))
    const_inf = True if samples.geometry.par_dim == 1 else False 

    # Create figure: 
    fig = plt.figure( figsize=(12, 14))
    fig.suptitle(tag)

    subfigs = fig.subfigures(2, 1)
            
    axsTop = subfigs[0].subplots(3, 2)
    axsBottom = subfigs[1].subplots(3, 2)
                
    # Add super title
    plt.suptitle('Experiment results: '+tag)

    # Plot exact data
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
    samples.funvals.plot_ci(exact = exact)
    # If inference type is not constant, plot data locations as vertical lines
    if not const_inf:
        for loc in locations:
            plt.axvline(x = loc, color = 'gray', linestyle = '--')
    plt.title('Posterior samples CI')

    # Plot ESS
    plt.sca(axsTop[2, 0])
    ESS_list = np.array(samples.compute_ess()) 
    plt.plot(ESS_list)
    plt.title('ESS')

    # Plot trace
    trace_idx_list = [0] if const_inf else [0, 5, 15]
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
        data_pt_str
    
    return tag
