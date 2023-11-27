#%% Script to generate plots for a presentation (CUQI seminar Nov 2023)

# Imports
from my_utils_aug28 import plot_experiment, plot_time_series, read_experiment_data, matplotlib_setup #, set_matlab_defaults
import matplotlib.pyplot as plt
from cuqi.pde import TimeDependentLinearPDE
from cuqi.geometry import MappedGeometry, Discrete, Continuous2D, Continuous1D
from cuqi.model import PDEModel
import numpy as np
import cuqi

# Functions:
def const_diff_model_aug25(input, times, locations, data, L=500, n_grid=100, tau_max=30*60, cfl=5):
    ## Set PDE parameters
    h = L/(n_grid+1)   # Space step size
    grid = np.linspace(h, L-h, n_grid)
    
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)
    
    ## Source term (constant diffusion coefficient case)
    def g_const(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = c/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
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

    return A_const(input)
    
def var_diff_model_aug25(input, times, locations, data, L=500, n_grid=100, tau_max=30*60, cfl=5):
    ## Set PDE parameters
    h = L/(n_grid+1)   # Space step size
    grid = np.linspace(h, L-h, n_grid)
    
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)
    

    # grid for the diffusion coefficient
    n_grid_c = 20
    hs = L/(n_grid_c+1) 
    grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
    grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)
    
    ## Source term (varying in space diffusion coefficient case)
    def g_var(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = c[0]/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
        return f_array
    
    ## Differential operator (varying in space diffusion coefficient case)
    Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
    vec = np.zeros(n_grid)
    vec[0] = 1
    Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
    Dx /= h # FD derivative matrix
    
    D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx
    
    ## Initial condition
    initial_condition = np.zeros(n_grid)

    def PDE_form_var(c, tau_current):
        c = np.interp(grid_c_fine, grid_c, c)
        return (D_c_var(c), g_var(c, tau_current), initial_condition)
    
    
    PDE_var = TimeDependentLinearPDE(PDE_form_var, tau, grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                time_obs=times) 
    
    # Domain geometry
    G_D_var =  MappedGeometry( Continuous1D(grid_c),  map=lambda x: x**2 )
    
    ## Range geometry
    G_cont2D = Continuous2D((locations, times))
    
    A_var = PDEModel(PDE_var, range_geometry=G_cont2D, domain_geometry=G_D_var)

    return A_var(input)
    
    
# Output directory
dir_name = 'data/ear_aqueducts'

#%%
# PLOT 1: REAL PARAMETERS AND DATA
# ================================
# Load data
dir_name = 'data/ear_aqueducts'
tag = 'ear_aqueducts'

#plot_time_series(times, locations, data)

## PLOT 2: RECONSTRUCTION FOR SYNTHETIC DATA (exact is constant)
## =============================================
#
## A: Constant inference
## Load data
#dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
#tag = 'm1_l_NUTS_constant_100.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'
#const = True
#exact, exact_data, data, mean_recon_data, samples,\
#experiment_par, locations, times =\
#    read_experiment_data(dir_name, tag, const=const)
#
#
## Plot
#plot_experiment(exact, exact_data, data, mean_recon_data, samples, experiment_par, locations, times, plot_exact=True)
#
## save figure
#
## B: Variable inference
#const = False
#exact, exact_data, data, mean_recon_data, samples,\
#experiment_par, locations, times =\
#    read_experiment_data(dir_name, tag, const=const)
#
## Plot
#plot_experiment(exact, exact_data, data, mean_recon_data, samples, experiment_par, locations, times, plot_exact=False)
#
## PLOT 3: RECONSTRUCTION FOR SYNTHETIC DATA (exact is variable)
## =============================================
## A: Constant inference
## Load data
#dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
#tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'
#
#const = True
#exact, exact_data, data, mean_recon_data, samples,\
#experiment_par, locations, times =\
#    read_experiment_data(dir_name, tag, const=const)
#
## Plot
#plot_experiment(exact, exact_data, data, mean_recon_data, samples, experiment_par, locations, times, plot_exact=False)
#
## B: Variable inference
#const = False
#exact, exact_data, data, mean_recon_data, samples,\
#experiment_par, locations, times =\
#    read_experiment_data(dir_name, tag, const=const)
#
## Plot
#plot_experiment(exact, exact_data, data, mean_recon_data, samples, experiment_par, locations, times)
#


# %%
#PLOT 4: Summary of results (reconstruction)
# Create and save figure of 2 rows and 3 columns for
# Data, 95% credible interval (constant inference), 95% credible interval (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
tag = 'm1_l_NUTS_constant_100.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'
const = True
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=const)

# Create figure
figure, axs = plt.subplots(2, 3, figsize=(9, 4.5))
#---------------------- time series for the constant real parameter case
plt.sca(axs[0, 0])
plot_time_series(times, locations, data)

# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)

#---------------------- 95% CI (constant inference)
plt.sca(axs[0, 1])
samples.is_par = True
samples.is_vec = True
exact.geometry = samples.geometry
exact.is_par = False
samples.plot_ci(exact = exact)
plt.ylim([70, 150])
# set ticks labels
plt.gca().set_xticklabels(["$x_0$"])
plt.xlabel('unknown parameter')
plt.ylabel('$c^2$')

#---------------------- 95% CI (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[0, 2])
samples.is_par = True
samples.is_vec = True
exact.geometry = samples.geometry
samples.plot_ci(exact = None)
plt.ylim([70, 150])
# set ticks labels
plt.xlabel('$\\xi$')
plt.ylabel('$c^2$')
# plot vertical dotted lines for locations
for loc in locations:
    plt.axvline(x = loc, color = 'gray', linestyle = '--')
#---------------------- time series for the variable real parameter case
# Load data
tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'

exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=True)
plt.sca(axs[1, 0])
plot_time_series(times, locations, data)


#---------------------- 95% CI (constant inference)
plt.sca(axs[1, 1])
samples.is_par = True
samples.is_vec = True   
samples.plot_ci()
plt.ylim([300, 5000])
# set ticks labels
plt.gca().set_xticklabels(["$x_0$"])
plt.xlabel('unknown parameter')
plt.ylabel('$c^2$')

#---------------------- 95% CI (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[1, 2])
samples.is_par = True
samples.is_vec = True
exact.geometry = samples.geometry
exact.is_par = False
samples.plot_ci(exact = exact)
plt.ylim([300, 1500])
plt.xlabel('$\\xi$')
plt.ylabel('$c^2$')
# change legend of last plot
plt.legend(['95% CI' ,'Mean', 'Exact', ])
# plot vertical dotted lines for locations
for loc in locations:
    plt.axvline(x = loc, color = 'gray', linestyle = '--')


# %%
# %% PLOT 5: Summary of results (reconstruction) with additional data point at 480

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'

# Create figure
figure, axs = plt.subplots(2, 3, figsize=(9, 4.5))
#---------------------- time series for the constant real parameter case
plt.sca(axs[0, 0])

# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)

#---------------------- 95% CI (constant inference)
plt.sca(axs[0, 1])


#---------------------- 95% CI (variable inference)
plt.sca(axs[0, 2])

#---------------------- time series for the variable real parameter case
# Load data
tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_480.0'

plt.sca(axs[1, 0])



#---------------------- 95% CI (constant inference)
plt.sca(axs[1, 1])


#---------------------- 95% CI (variable inference)
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_480.0'
const = True
npz_file = np.load(dir_name+'/output'+tag+'/posterior_samples_var_'+tag+'.npz')


samples2 = cuqi.samples.Samples(npz_file['arr_0'], geometry = samples.geometry)


plt.sca(axs[1, 2])

samples2.plot_ci(exact = exact)
plt.ylim([300, 1500])
plt.xlabel('$\\xi$')
plt.ylabel('$c^2$')
# change legend of last plot
plt.legend(['95% CI' ,'Mean', 'Exact', ])
# plot vertical dotted lines for locations
locations2 = np.append(locations, 480)
for loc in locations2:
    plt.axvline(x = loc, color = 'gray', linestyle = '--')



# %%
# %%
#PLOT 6: Summary of results (mean-reconstructed data)
# Create and save figure of 2 rows and 3 columns for
# Data, mean-reconstructed data (constant inference), mean-reconstructed data (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
tag = 'm1_l_NUTS_constant_100.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'
const = True
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=const)

# Create figure
figure, axs = plt.subplots(2, 3, figsize=(9, 4.5))

#---------------------- time series for the constant real parameter case
plt.sca(axs[0, 0])
plot_time_series(times, locations, data)

# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)

#---------------------- mean-reconstructed data (constant inference)
plt.sca(axs[0, 1])
plot_time_series(times, locations, mean_recon_data)



#---------------------- mean-reconstructed data (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[0, 2])
plot_time_series(times, locations, mean_recon_data)


#---------------------- time series for the variable real parameter case
# Load data
tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'

exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=True)
plt.sca(axs[1, 0])
plot_time_series(times, locations, data)


#---------------------- mean-reconstructed data (constant inference)
plt.sca(axs[1, 1])
plot_time_series(times, locations, mean_recon_data)

#---------------------- mean-reconstructed data (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[1, 2])
plot_time_series(times, locations, mean_recon_data)


# %%


# %%
#PLOT 7: Summary of results (ESS)
# Create and save figure of 2 rows and 3 columns for
# Data, ESS (constant inference), ESS (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts_aug_25'
tag = 'm1_l_NUTS_constant_100.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'
const = True
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=const)

# Create figure
figure, axs = plt.subplots(2, 3, figsize=(9, 4.5))

#---------------------- time series for the constant real parameter case
plt.sca(axs[0, 0])
plot_time_series(times, locations, data)

# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)

#---------------------- ESS (constant inference)
plt.sca(axs[0, 1])
samples.is_par = True
samples.is_vec = True
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('unknown parameter')
plt.ylabel('ESS')




#---------------------- ESS (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[0, 2])
samples.is_par = True
samples.is_vec = True
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('$\\xi$')
plt.ylabel('ESS')



#---------------------- time series for the variable real parameter case
# Load data
tag = 'm1_l_NUTS_smooth_400.0_1200.0_synthetic_both_5000_5000_0.01_v_aug25_a_rerun_'

exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=True)
plt.sca(axs[1, 0])
plot_time_series(times, locations, data)
const_geom = samples.geometry


#---------------------- ESS (constant inference)
plt.sca(axs[1, 1])
samples.is_par = True
samples.is_vec = True
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('unknown parameter')
plt.ylabel('ESS')

#---------------------- ESS (variable inference)
exact, exact_data, data, mean_recon_data, samples,\
experiment_par, locations, times =\
    read_experiment_data(dir_name, tag, const=False)
plt.sca(axs[1, 2])
samples.is_par = True
samples.is_vec = True
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('$\\xi$')
plt.ylabel('ESS')
var_geom = samples.geometry


# %% PLOT 8: Summary of results for real data (reconstruction)
# Create and save figure of 1 rows and 3 columns for 
# Data, 95% credible interval (constant inference), 95% credible interval (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts'
animal = 'm1'
ear = 'l'


# Create figure
figure, axs = plt.subplots(1, 3, figsize=(9, 2.1))
# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)
#---------------------- time series for the real data
plt.sca(axs[0])
#read real data

# distance file
import pandas as pd
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_distances.csv')
real_locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_parsed.csv')
real_data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
real_times = constr_file['time'].values*60

real_data = real_data.reshape((len(real_locations), len(real_times)))
plot_time_series(real_times, real_locations, real_data)

#---------------------- 95% CI (constant inference)
plt.sca(axs[1])

tag = animal+ear+'NUTSv8'
samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_const_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = const_geom)

samples.plot_ci(exact = None)
plt.ylim([500, 4000])
# set ticks labels
plt.xlabel('unknown parameter')
plt.ylabel('$c^2$')

#---------------------- 95% CI (variable inference)
plt.sca(axs[2])

samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_var_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = var_geom)

samples.plot_ci(exact = None)
plt.ylim([0, 1000])
# set ticks labels
plt.xlabel('$\\xi$')
plt.ylabel('$c^2$')

# %% PLOT 9: Summary of results for real data (mean-reconstructed data)
# Create and save figure of 1 rows and 3 columns for 
# Data, mean-reconstructed data (constant inference), mean-reconstructed data (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts'
animal = 'm1'
ear = 'l'


# Create figure
figure, axs = plt.subplots(1, 3, figsize=(9, 2.1))
# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)
#---------------------- time series for the real data
plt.sca(axs[0])
#read real data

# distance file
import pandas as pd
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_distances.csv')
real_locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_parsed.csv')
real_data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
real_times = constr_file['time'].values*60

real_data = real_data.reshape((len(real_locations), len(real_times)))
plot_time_series(real_times, real_locations, real_data)


#---------------------- mean-reconstructed data (constant inference)
plt.sca(axs[1])
tag = animal+ear+'NUTSv8'
samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_const_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = const_geom)
samples_mean = samples.mean()

mean_recon_data = const_diff_model_aug25(samples_mean,real_times, real_locations, real_data)
plot_time_series(real_times, real_locations, mean_recon_data.reshape((len(real_locations), len(real_times))))

#---------------------- mean-reconstructed data (variable inference)
plt.sca(axs[2])
samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_var_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = var_geom)
samples_mean = samples.mean()

mean_recon_data = var_diff_model_aug25(samples_mean,real_times, real_locations, real_data)
plot_time_series(real_times, real_locations, mean_recon_data.reshape((len(real_locations), len(real_times))))


#%%PLOT 10: Summary of results for real data (ESS)
# Create and save figure of 1 rows and 3 columns for
# Data, ESS (constant inference), ESS (variable inference)

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts'
animal = 'm1'
ear = 'l'


# Create figure

figure, axs = plt.subplots(1, 3, figsize=(9, 2.1))
# increase spacing between subplots
figure.subplots_adjust(hspace=0.4, wspace=0.4)

#---------------------- time series for the real data
plt.sca(axs[0])
#read real data

# distance file
import pandas as pd
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_distances.csv')
real_locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_parsed.csv')
real_data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
real_times = constr_file['time'].values*60

real_data = real_data.reshape((len(real_locations), len(real_times)))
plot_time_series(real_times, real_locations, real_data)

#---------------------- ESS (constant inference)
plt.sca(axs[1])
tag = animal+ear+'NUTSv8'
samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_const_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = const_geom)
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('unknown parameter')
plt.ylabel('ESS')

#---------------------- ESS (variable inference)
plt.sca(axs[2])
samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_var_'+tag+'.npz')
samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = var_geom)
samples.geometry.plot(samples.compute_ess(), is_par=False)
plt.xlabel('$\\xi$')
plt.ylabel('ESS')




# %% Plot 11: Summary of results for real data (reconstruction) for different animals
# Create and save figure of 4 rows and 2 columns for
# Data, 95% credible interval (variable inference) 
# and add the average ESS in the title of CI plots

matplotlib_setup(8, 9, 10)
# Load data
dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts'
animals = ['m1', 'm3', 'm4', 'm6']
ears = ['l']

# Create figure
figure, axs = plt.subplots(4, 2, figsize=(9, 7))
# increase spacing between subplots
figure.subplots_adjust(hspace=0.5, wspace=0.4)

for i, animal in enumerate(animals):
    for j, ear in enumerate(ears):
        #---------------------- time series for the real data
        plt.sca(axs[i, 0])
        #read real data
        
        # distance file
        import pandas as pd
        dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_distances.csv')
        real_locations = dist_file['distance microns'].values[:5]
        
        ## Read concentration file
        constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_parsed.csv')
        real_data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
        real_times = constr_file['time'].values*60
        
        real_data = real_data.reshape((len(real_locations), len(real_times)))
        plot_time_series(real_times, real_locations, real_data, loc='upper right', ncol=2)
        plt.ylim([0, 7000])
        if i!=0:
            # remove legend
            plt.gca().legend_.remove()

        plt.title("animal "+animal+", ear "+ear)
        # set xlabel location
        plt.gca().xaxis.set_label_coords(0.5, 0.12)
        
        #---------------------- 95% CI (variable inference)
        plt.sca(axs[i, 1])
        tag = animal+ear+'NUTSv8'
        samples_numpy = np.load(dir_name+'/output'+tag+'/posterior_samples_var_'+tag+'.npz')
        samples = cuqi.samples.Samples(samples_numpy['arr_0'], geometry = var_geom)
        samples.plot_ci(exact = None)
        plt.ylim([0, 1500])
        # set ticks labels
        plt.xlabel('$\\xi$')
        plt.ylabel('$c^2$')
        # set xlabel location
        plt.gca().xaxis.set_label_coords(0.5, -.05)
        # add ESS to title 
        plt.title('ESS = '+str(int(np.average(samples.compute_ess()))))



# %%
