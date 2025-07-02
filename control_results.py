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

animal, ear = "m1", "r" # Which animal/ear case (to set observation locations, and boundary and initial conditions).
version = "control1" # Label of the inference, used to distinguish different runs.
sampler = "NUTSWithGibbs" # Sampler to use
unknown_par_type = "synth_diff1.npz" # Read the "true" diffusion coefficient" from this file.
unknown_par_value = 100.0 # Value of the diffusion coefficient (not used here since we read it from the file).
data_type = "synthetic" # Type of data to use, here we generate and use synthetic data.
inference_type = "heterogeneous" # Type of inference to perform, here we use heterogeneous to infer spatially varying diffusion coefficients.
num_CA = 5 # Number of CA data points to use, was set to 5 in the paper.
num_ST = 0 # Number of ST data points to use, was set to 0 in the paper.
rbc = "fromDataClip" # Use real data to set the right boundary conditions (clipping the negative values to zero).
noise_level = "std_0.5" # Noise level of the synthetic data, of std of 0.5
true_a = 0.0 # True advection coefficient.

# Lambda function to generate the command for running the inference script.
command_function = lambda inference_type, true_a: \
f"""
python code/advection_diffusion_inference.py -animal {animal} -ear {ear} -version {version} -sampler {sampler} -unknown_par_type {unknown_par_type} -unknown_par_value {unknown_par_value} -data_type {data_type} -inference_type {inference_type} -Ns {Ns} -Nb {Nb} -noise_level {noise_level} -num_CA {num_CA} -num_ST {num_ST} -true_a {true_a} -rbc {rbc} -NUTS_kwargs '{{"max_depth": {max_depth}, "step_size": 0.1}}' --data_grad  --sampler_callback  --adaptive"""

# Results for Figure 2, row 1.
# Tag to identify the saved files
file_tag_fig2_row1 = f"{animal}_{ear}_{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}"
# Command to run the inference script
command_fig2_row1 = command_function(inference_type, true_a)
# Run the command
submit(file_tag_fig2_row1, command_fig2_row1) # commented out to avoid running it automatically since it takes ~13 hrs to run

# Results for Figure 2, row 2.
true_a = -1.0 # True advection coefficient.
inference_type = "advection_diffusion" # Type of inference to perform, here we infer both advection and diffusion coefficients.
# Tag to identify the saved files
file_tag_fig2_row2 = f"{animal}_{ear}_{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}"
# Command to run the inference script
command_fig2_row2 = command_function(inference_type, true_a)
# Run the command
submit(file_tag_fig2_row2, command_fig2_row2) # commented out to avoid running it automatically since it takes ~13 hrs to run

# Results for Figure 2, row 3.
true_a = 0.5 # True advection coefficient.
# Tag to identify the saved files
file_tag_fig2_row3 = f"{animal}_{ear}_{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}" 
# Command to run the inference script
command_fig2_row3 = command_function(inference_type, true_a)
# Run the command
submit(file_tag_fig2_row3, command_fig2_row3) # commented out to avoid running it automatically since it takes ~13 hrs to run

# Results for Figure 2, row 4.
true_a = 2.0 # True advection coefficient.
# Tag to identify the saved files
file_tag_fig2_row4 = f"{animal}_{ear}_{sampler}_{unknown_par_type}_{unknown_par_value}_{data_type}_{inference_type}_{Ns}_{noise_level}_{version}__{num_ST}_{num_CA}_{true_a}_{rbc}"
# Command to run the inference script
command_fig2_row4 = command_function(inference_type, true_a)
# Run the command
submit(file_tag_fig2_row4, command_fig2_row4) # commented out to avoid running it automatically since it takes ~13 hrs to run