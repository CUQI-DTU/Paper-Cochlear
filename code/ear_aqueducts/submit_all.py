#%% Submit all the jobs to the cluster
# Usage: python3 submit_all.py

# Importing necessary libraries
import os
from job_submit import submit, create_command
from advection_diffusion_inference_utils import all_animals, all_ears, Args,\
    create_experiment_tag, create_args_list
version = 'v21May2024_temp'
Ns = 20
Nb = 10
noise_levels = ["fromDataVar", "fromDataAvg", "avgOverTime", 0.1, 0.2]
add_data_pts_list = [[]]

if version == 'v_April22_2024_':
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    noise_levels.remove("avgOverTime")
    num_ST_list = [0, 4]
    sampler = 'NUTS'
    data_type = 'real'
    unknown_par_types = ['constant']
    unknown_par_values = [[100.0]]

elif version == 'v06May2024_a':
    # Array of all animals
    animals = [None]
    # Array of all ears
    ears = [None]
    num_ST_list = [4]
    sampler = 'NUTS'
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['sampleMean']
    unknown_par_values = ['m1:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm1:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm2:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm2:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm3:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm3:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm4:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm4:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm6:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm6:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4']

elif version == 'v20May2024_const_b':
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [4]
    sampler = 'NUTS'
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['constant']
    unknown_par_values = [[100.0]]

elif version == 'v21May2024_temp':
    # Array of all animals
    noise_levels = [ 0.1, 0.2, "fromDataAvg"]
    animals = [None]
    # Array of all ears
    ears = [None]
    num_ST_list = [4]
    add_data_pts_list = [[400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]]
    sampler = 'MH'#'NUTS'
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['sampleMean']
    unknown_par_values = ['m1:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm1:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm2:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm2:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm3:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm3:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm4:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm4:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm6:l:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4',
                          'm6:r:NUTS:constant:100.0:real:heterogeneous:1000:0.1:v:April22:2024:a::4:5@results4']

# Main command to run the job
main_command = "python advection_diffusion_inference.py"
arg_list = create_args_list(animals, ears, noise_levels, num_ST_list, add_data_pts_list, unknown_par_types, unknown_par_values, data_type, version, sampler, Ns, Nb)
for args in arg_list:
    cmd = create_command(main_command, args)
    print()
    print(cmd)
    tag = create_experiment_tag(args)
    print(tag)
    submit(tag, cmd)