#%% Script to generate plots for a presentation (CUQI seminar Nov 2023)

# Imports
from my_utils_aug28 import plot_experiment, plot_time_series, read_experiment_data, matplotlib_setup #, set_matlab_defaults
import matplotlib.pyplot as plt

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
import numpy as np
import cuqi
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
#PLOT 4: Summary of results (mean-reconstructed data)
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
