#%% Submit all the jobs to the cluster
# Usage: python3 submit_all.py

# Importing necessary libraries
import os
from job_submit import submit, create_command
from advection_diffusion_inference_utils import all_animals, all_ears, Args,\
    create_experiment_tag, create_args_list
#version = "v16Aug2024_synth_large_a_repeat_sept8_fix_geom"
#version = "v12Sep2024_no_Gibbs_real"
#version = "v14septCASynthAdvDiff"
#version = "v14septCASynthDiff"
#version = "v14septCARealAdvDiff"
#version = "v14septCARealDiff"
#version = "v14septCASTSynthDiff"
#version = "v14septCASTRealDiff"
#version = "v13octCARealDiffGibbs"


#version = "paperV2CASynthDiff"
#version = "paperV2CARealDiff"  
#version = "paperV2CASTSynthDiff"
#version = "paperV2CASTRealDiff"
#version = "paperV2CASynthAdvDiff"
#version = "paperV2CARealAdvDiff"
#version = "paperV2CARealDiff_CArbc_clip"
#version = "paperV2CARealAdvDiff_CArbc_clip"
#version = "paperV2CARealDiff_CArbc_clip_grad_data_temp_Nov15"

# V3
#version = "paperV3CARealDiff"
#version = "paperV3CASTRealDiff"
#version = "paperV3CARealAdvDiff"
# paperV4CARealDiff_GMRF_gibbs_scale_all_pixel3 IS NOT GMRF



#version = "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all"

#version = "paperV4CARealAdvDiff"
#version = "paperV4CARealDiffPixel"
#version = "paperV4CARealAdvDiffPixel"

#Ns_s = [1000]
#Nb_s = [10]

version = "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff1_zerou0_update_hp"
#version = "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff2_zerou0_update_hp"
#version = "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff3_zerou0_update_hp"

#version = "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff1_zerou0_update_hp"
#version = "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff2_zerou0_update_hp"
#version = "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff3_zerou0_update_hp"

noise_levels = ["fromDataVar", "fromDataAvg", "avgOverTime", 0.1, 0.2]
add_data_pts_list = [[]]
inference_type = 'heterogeneous'

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

elif version == 'v21May2024_c':
    # Array of all animals
    noise_levels = [ 0.1, 0.2, "fromDataAvg"]
    animals = [None]
    # Array of all ears
    ears = [None]
    num_ST_list = [4]
    add_data_pts_list = [[400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]]
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

elif version == 'v16Aug2024_synth_small_a':
    # Array of all animals
    animals =[all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = 'NUTS'
    Ns = 200
    Nb = 20
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = 0.245 # funval

elif version == 'v16Aug2024_synth_large_a':
    # Array of all animals
    animals =[all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = 'NUTS'
    Ns = 200
    Nb = 20
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = 0.9 # funval

elif version == 'v16Aug2024_real':
    # Array of all animals
    animals =all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 200000
    #Nb = 20000
    # opt 2
    sampler = 'NUTS'
    Ns = 200
    Nb = 20
    data_type = 'real'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = 100 # funval

elif version == "v16Aug2024_synth_large_a_repeat_sept8_fix_geom":
    # Array of all animals
    animals =[all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = 'NUTS'
    Ns = 200
    Nb = 20
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = 0.9 # funval

if version == "v12Sep2024_no_Gibbs_synth":
    # Array of all animals
    animals =[all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = ['NUTS', 'MH', 'MH']
    Ns = [1000, 5000000, 10000000] # try 10000000 for MH
    Nb = [20, 500000, 1000000]
    rbc = ['zero', 'fromData']
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = [0.1, 0.9] # funval

if version == "v12Sep2024_no_Gibbs_real":
    # Array of all animals
    animals =all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]
    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = ['NUTS', 'MH', 'MH']
    Ns = [1000, 5000000, 10000000] # try 10000000 for MH
    Nb = [20, 500000, 1000000]
    rbc = ['zero', 'fromData']
    data_type = 'real'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = 'advection_diffusion'
    true_a = [0.1] # funval (value not used)

if version == "v14septCASynthAdvDiff":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [3000] # try 10000000 for MH
    Nb = [20]
    rbc = ['zero']
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = ['advection_diffusion']
    true_a = [0.1, 0.9] # funval 
    noise_levels = ["fromDataAvg"]

if version == "v14septCARealAdvDiff":
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [3000] # try 10000000 for MH
    Nb = [20]
    rbc = ['zero']
    data_type = 'real'
    unknown_par_types = ['custom_1']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type =['advection_diffusion']
    true_a = [0.1] # funval 
    noise_levels = ["fromDataAvg"]

if version == "v14septCASynthDiff":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]

    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = ['NUTS']
    Ns = [3000] # try 10000000 for MH
    Nb = [20]
    rbc = ['zero']
    data_type = 'syntheticFromDiffusion'
    unknown_par_types = ['constant', 'smooth']
    unknown_par_values = [[100.0], [200, 50]] # this value is not used in the code supposedly
    inference_type = ['constant', 'heterogeneous']
    true_a = [0.1] # funval (value not used)
    noise_levels = ["fromDataAvg"]

if version == "v14septCARealDiff":
# case

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = ['NUTS']
    Ns = [3000] # try 10000000 for MH
    Nb = [20]
    rbc = ['zero']
    data_type = 'real'
    unknown_par_types = ['constant']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = ['heterogeneous']
    true_a = [0.1] # funval (value not used)
    noise_levels = ["fromDataAvg"]

if version == "v14septCASTSynthDiff":

    noise_levels = ["fromDataAvg"]
    # Array of all animals
    animals = [None]
    # Array of all ears
    ears = [None]
    num_ST_list = [4]
    sampler = ['NUTS']
    Ns = [3000]
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['zero'] 
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
    unknown_par_types = ['sampleMean']*len(unknown_par_values)

if version == "v14septCASTRealDiff":

    noise_levels = ["fromDataAvg"]
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [4]
    sampler = ['NUTS']
    Ns = [3000]
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['zero'] 
    unknown_par_types = ['constant']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly

if version == "v13octCARealDiffGibbs":
# case

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    # opt 1
    #sampler = 'MH'
    #Ns = 300000
    #Nb = 20000
    # opt 2
    sampler = ['NUTSWithGibbs']
    Ns = [900] # try 500
    Nb = [100] # try 500
    rbc = ['zero']
    data_type = 'real'
    unknown_par_types = ['constant']
    unknown_par_values = [[100.0]] # this value is not used in the code supposedly
    inference_type = ['heterogeneous']
    true_a = [0.1] # funval (value not used)
    noise_levels = ["fromDataAvg"] # this noise level will not be used here



if version == "paperV2CASynthDiff":
    raise ValueError("This version is not supported yet")


if version == "paperV2CARealDiff":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [1]

    sampler = ['NUTSWithGibbs']
    Ns = [5000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromData']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":7}

if version == "paperV2CASTSynthDiff":
    raise ValueError("This version is not supported yet")

if version == "paperV2CASTRealDiff":
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [4]
    sampler = ['NUTSWithGibbs']
    Ns = [5000]
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromData']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":7}

if version == "paperV2CASynthAdvDiff":
    raise ValueError("This version is not supported yet")

if version == "paperV2CARealAdvDiff":
    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [1]

    sampler = ['NUTSWithGibbs']
    Ns = [5000] # try 10000000 for MH
    Nb = [0]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type =['advection_diffusion']
    rbc = ['fromData']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here 
    NUTS_kwargs = {"max_depth":7, "step_size": 0.25}


if version == "paperV3CARealDiff":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTSWithGibbs']
    Ns = [5000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True

if version == "paperV3CASTRealDiff":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [4]

    sampler = ['NUTSWithGibbs']
    Ns = [5000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True


if version == "paperV3CARealAdvDiff":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTSWithGibbs']
    Ns = [5000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["fromDataAvg"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True

if version == "paperV2CARealDiff_CArbc_clip_grad_data_temp_Nov15":
    # Array of all animals
    animals = ['m2']
    # Array of all ears
    ears = ['l']
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [5] # try 10000000 for MH
    Nb = [2]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = [0.2] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = True

if version == "paperV4CARealDiff_Gauess_gibbs_scale_all7":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [2000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = [0.2] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CARealAdvDiff_Gauess_gibbs_scale_all7":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [2000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = [0.2] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CARealAdvDiff":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [1000] # try 10000000 for MH
    Nb = [20]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ["estimated"] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = False

if version == "paperV4CARealDiffPixel":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [2] # try 10000000 for MH
    Nb = [1]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = [0.3] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = True

if version == "paperV4CARealAdvDiffPixel":

    # Array of all animals
    animals = all_animals()
    # Array of all ears
    ears = all_ears()
    num_ST_list = [0]

    sampler = ['NUTS']
    Ns = [2] # try 10000000 for MH
    Nb = [1]
    data_type = 'real'
    true_a = [0.1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['constant'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = [0.3] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":5}
    data_grad = True
    u0_from_data = True
    sampler_callback = True
    pixel_data = True

if version == "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff1_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[1]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff1.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff2_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[2]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff2.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CASynthDiff_Gauess_gibbs_scale_all_diff3_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[1]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.1] # funval (value not used)
    inference_type = ['heterogeneous']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff3.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff1_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[1]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.5, 2, -1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff1.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff2_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[2]]
    # Array of all ears
    ears = [all_ears()[0]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.5, 2, -1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff2.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

if version == "paperV4CASynthAdvDiff_Gauess_gibbs_scale_all_diff3_zerou0_update_hp":
    # Array of all animals
    animals = [all_animals()[0]]
    # Array of all ears
    ears = [all_ears()[1]]
    num_ST_list = [0]


    sampler = ['NUTSWithGibbs']
    Ns = [200] # try 10000000 for MH
    Nb = [20]
    data_type = 'syntheticFromDiffusion'
    true_a = [0.5, 2, -1] # funval (value not used)
    inference_type = ['advection_diffusion']
    rbc = ['fromDataClip']
    unknown_par_types = ['synth_diff3.npz'] # this value is not used in this case
    unknown_par_values = [[100.0]] # this value is not used in this case
    noise_levels = ['std_0.1'] # this noise level will not be used here
    NUTS_kwargs = {"max_depth":10, "step_size": 0.1}
    data_grad = True
    u0_from_data = False
    sampler_callback = True
    pixel_data = False
    adaptive = True

# Main command to run the job
main_command = "python advection_diffusion_inference.py"
arg_list = create_args_list(animals, ears, noise_levels, num_ST_list, add_data_pts_list, unknown_par_types, unknown_par_values, data_type, version, sampler, Ns, Nb, inference_type, true_a, rbc, NUTS_kwargs, data_grad, u0_from_data, sampler_callback, pixel_data, adaptive)
print("length of arg_list: ", len(arg_list))
for args in arg_list:
    cmd = create_command(main_command, args)
    print()
    print(cmd)
    tag = create_experiment_tag(args)
    print(tag)
    submit(tag, cmd)