# Submit all the jobs to the cluster
# Usage: python3 submit_all.py

# Importing necessary libraries
import os
from job_submit import submit, create_command
from advection_diffusion_inference_utils import all_animals, all_ears, Args,\
    create_experiment_tag

# Array of all animals
animals = all_animals()
# Array of all ears
ears = all_ears()
noise_levels = ["from_data_var", "from_data_avg"]
num_ST_list = [0, 8]
version = 'v_April20_2024'
sampler = 'MH'
Ns = 20
Nb = 10

# Main command to run the job
main_command = "python advection_diffusion_inference.py"

# Loop over all animals, ears, noise levels and num_ST
for animal in animals:
    for ear in ears:
        for noise_level in noise_levels:
            for num_ST in num_ST_list:
                args = Args()
                args.animal = animal
                args.ear = ear
                args.version = version
                args.sampler = sampler
                args.data_type = 'real'
                args.Ns = Ns
                args.Nb = Nb
                args.noise_level = noise_level
                args.num_ST = num_ST

                cmd = create_command(main_command, args)
                print(cmd)
                tag = create_experiment_tag(args)
                submit(tag, cmd)