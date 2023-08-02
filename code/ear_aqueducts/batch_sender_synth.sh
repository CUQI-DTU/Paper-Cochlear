#!/bin/bash


./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_b' -sampler 'MH' -unknown_par_type 'constant' -unknown_par_value 400 -data_type 'synthetic' -inference_type 'both' -Ns_const 100 -Ns_var 100 -noise_level 0.1
./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_b' -sampler 'MH' -unknown_par_type 'constant' -unknown_par_value 100 -data_type 'synthetic' -inference_type 'both' -Ns_const 100 -Ns_var 100 -noise_level 0.1
