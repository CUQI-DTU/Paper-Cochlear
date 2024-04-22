# Submit all the jobs to the cluster
# Usage: python3 submit_all.py

# Importing necessary libraries
import os
from job_submit import submit, create_command
from advection_diffusion_inference_utils import all_animals, all_ears, Args,\
    create_experiment_tag

# Array of all animals
animals = all_animals()[:2]
# Array of all ears
ears = all_ears()
noise_levels = ["from_data_var", "from_data_avg", 0.1, 0.2][:2]
num_ST_list = [0, 4]
version = 'v_April22_2024_b'
sampler = 'NUTS'
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
                args.inference_type = 'heterogeneous'

                cmd = create_command(main_command, args)
                print(cmd)
                tag = create_experiment_tag(args)
                submit(tag, cmd)