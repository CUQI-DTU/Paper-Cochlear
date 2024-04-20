# PLOT AND SAVE ALL DATA FOR EAR AQUEDUCTS FOR ALL ANIMALS (CONCENTRATION AND STD)

#%% Imports 
import numpy as np
import os
import cuqi
import sys
from advection_diffusion_inference_utils import\
    read_data_files,\
    plot_time_series
import matplotlib.pyplot as plt

print('cuqi version:')
print(cuqi.__version__)

class Args:
    def __init__(self, animal, ear, num_CA, num_ST):
        self.animal = animal
        self.ear = ear
        self.num_CA = num_CA
        self.num_ST = num_ST

args = Args('m1', 'l', 5, 0)
ears = ['l', 'r']
animals = ['m1', 'm2', 'm3', 'm4', 'm6']

# Create 2 figures each with 5 rows and 2 columns
fig1, axs1 = plt.subplots(5, 2, figsize=(10, 20))
plt.suptitle('Concentration', y=0.9)
plt.subplots_adjust(hspace=0.3)
fig2, axs2 = plt.subplots(5, 2, figsize=(10, 20))
plt.suptitle('Std', y=0.9)

# Plot concentration and std for all animals and ears (including reading
# the data)
for i, animal in enumerate(animals):
    for j, ear in enumerate(ears):
        # Read data
        args.ear = ear
        args.animal = animal
        real_times, real_locations, real_data, real_std_data = read_data_files(args)
        # Plot concentration
        plt.sca(axs1[i, j])
        plot_time_series(real_times, real_locations, real_data.reshape([len(real_locations), len(real_times)]))
        plt.title(f'{animal} {ear}',loc='left')
        # Plot std
        plt.sca(axs2[i, j])
        plot_time_series(real_times, real_locations, real_std_data.reshape([len(real_locations), len(real_times)]))
        plt.title(f'{animal} {ear}',loc='left')

# Save figures
# create directory if it does not exist
if not os.path.exists('plots'):
    os.makedirs('plots')
fig1.savefig('plots/all_data_concentration.png')
fig2.savefig('plots/all_data_std.png')