#%%
import sys
sys.path.append('code/')
import os
from advection_diffusion_inference_utils import *
import matplotlib.pyplot as plt
import numpy as np
from job_submit import submit

max_depth = 10 # Depth of the NUTS tree.
Ns = 2000 # Number of samples to draw from the posterior distribution.
Nb = 20 # Number of burn-in samples.

version = "realdata_diff" # Label of the inference, used to distinguish different runs.
sampler = "NUTSWithGibbs" # Sampler to use
unknown_par_type = "constant" # Read the "true" diffusion coefficient" from this file.
unknown_par_value = 100.0 # Value of the diffusion coefficient (not used here since we read it from the file).
data_type = "real" # Type of data to use, here we generate and use synthetic data.
inference_type = "heterogeneous" # Type of inference to perform, here we use heterogeneous to infer spatially varying diffusion coefficients.
num_CA = 5 # Number of CA data points to use, was set to 5 in the paper.
num_ST = 0 # Number of ST data points to use, was set to 0 in the paper.
rbc = "fromDataClip" # Use real data to set the right boundary conditions (clipping the negative values to zero).
noise_level = "0.2" # Noise level of the synthetic data, (not used here because we use real data and noise level is inferred from gibbs sampling).
true_a = 0.0 # True advection coefficient.

# Lambda function to generate the command for running the inference script.
command_function = lambda animal, ear, inference_type: \
f"""
python code/advection_diffusion_inference.py -animal {animal} -ear {ear} -version {version} -sampler {sampler} -unknown_par_type {unknown_par_type} -unknown_par_value {unknown_par_value} -data_type {data_type} -inference_type {inference_type} -Ns {Ns} -Nb {Nb} -noise_level {noise_level} -num_CA {num_CA} -num_ST {num_ST} -true_a {true_a} -rbc {rbc} -NUTS_kwargs '{{"max_depth": {max_depth}, "step_size": 0.1}}' --data_grad  --sampler_callback  --adaptive --u0_from_data"""

# Diffusion inference cases run
all_animals_list = all_animals()
all_ears_list = all_ears()
diffusion_tag = f"{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}"
for animal in all_animals_list:
    for ear in all_ears_list:
        # Command to run the inference script
        command = command_function(animal, ear, inference_type)
        # Run the command
        submit(diffusion_tag, command) # commented out to avoid running it automatically since it takes long time to run


# Advection-diffusion inference cases run
inference_type = "advection_diffusion" # Type of inference to perform, here we infer both advection and diffusion coefficients.
advection_diffusion_tag = f"{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}"
for animal in all_animals_list:
    for ear in all_ears_list:
        # Command to run the inference script
        command = command_function(animal, ear, "advection_diffusion")
        # Run the command
        submit(advection_diffusion_tag, command) # commented out to avoid running it automatically since it takes long time to run
