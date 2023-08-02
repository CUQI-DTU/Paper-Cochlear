#!/bin/bash


./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_c' -sampler 'NUTS' -unknown_par_type 'constant' -unknown_par_value 400 -data_type 'synthetic' -inference_type 'both' -Ns_const 1000 -Ns_var 1000 -noise_level 0.1
./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_c' -sampler 'NUTS' -unknown_par_type 'constant' -unknown_par_value 100 -data_type 'synthetic' -inference_type 'both' -Ns_const 1000 -Ns_var 1000 -noise_level 0.1
./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_c' -sampler 'NUTS' -unknown_par_type 'constant' -unknown_par_value 2000 -data_type 'synthetic' -inference_type 'both' -Ns_const 1000 -Ns_var 1000 -noise_level 0.1
./sender_synth.sh -animal 'm1' -ear 'l' -version 'v_aug2_c' -sampler 'NUTS' -unknown_par_type 'smooth' -unknown_par_value 400 1200 -data_type 'synthetic' -inference_type 'both' -Ns_const 1000 -Ns_var 1000 -noise_level 0.1

