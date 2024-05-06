#%% Submit all the jobs to the cluster
# Usage: python3 submit_all.py

# Importing necessary libraries
import os
from job_submit import submit, create_command
from advection_diffusion_inference_utils import all_animals, all_ears, Args,\
    create_experiment_tag
version = 'v06May2024_a'
Ns = 20
Nb = 10
noise_levels = ["fromDataVar", "fromDataAvg", "avgOverTime", 0.1, 0.2]

if version == 'v_April22_2024_':
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    noise_levels.remove("avgOverTime")
    num_ST_list = [0, 4]
    version = 'v_April22_2024_'
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
    version = 'v06May2024_a'
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
    

# Main command to run the job
main_command = "python advection_diffusion_inference.py"

# Loop over all animals, ears, noise levels and num_ST
for animal in animals:
    for ear in ears:
        for noise_level in noise_levels:
            for num_ST in num_ST_list:
                for unknown_par_type in unknown_par_types:
                    for unknown_par_value in unknown_par_values:
                        args = Args()
                        args.animal = animal if animal is not None else unknown_par_value.split(':')[0]
                        args.ear = ear if ear is not None else unknown_par_value.split(':')[1]
                        args.version = version
                        args.sampler = sampler
                        args.data_type = data_type
                        args.Ns = Ns
                        args.Nb = Nb
                        args.noise_level = noise_level
                        args.num_ST = num_ST
                        args.inference_type = 'heterogeneous'
                        args.unknown_par_type = unknown_par_type
                        args.unknown_par_value = unknown_par_value

                        cmd = create_command(main_command, args)
                        print(cmd)
                        print()
                        tag = create_experiment_tag(args)
                        print(tag)
                        submit(tag, cmd)