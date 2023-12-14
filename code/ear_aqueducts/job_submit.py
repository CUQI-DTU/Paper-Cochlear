import os

#from skimage.metrics import structural_similarity as ssim
 

#folder_reconstruction = "Output"
#folder_output_examples = "gbar/results"
 
def submit(jobid,cmd):
    id = str(jobid)
    jobname = 'job_' + id
    memcore = 7000
    maxmem = 8000
    email = 'amaal@dtu.dk'
    ncores = 10
 
    # begin str for jobscript
    strcmd = '#!/bin/sh\n'
    strcmd += '#BSUB -J ' + jobname + '\n'
    strcmd += '#BSUB -q hpc\n'
    strcmd += '#BSUB -n ' + str(ncores) + '\n'
    strcmd += '#BSUB -R "span[hosts=1]"\n'
    strcmd += '#BSUB -R "rusage[mem=' + str(memcore) + 'MB]"\n'
    strcmd += '#BSUB -M ' + str(maxmem) + 'MB\n'
    strcmd += '#BSUB -W 2:00\n'
    strcmd += '#BSUB -u ' + email + '\n'
    strcmd += '#BSUB -N \n'
    strcmd += '#BSUB -o hpc/output/output_' + id + '.out\n'
    strcmd += '#BSUB -e hpc/error/error_' + id + '.err\n'
    strcmd += 'module load python3/3.10.2\n'
    strcmd += 'source ../../../../BE_collab/bin/activate\n'
    strcmd += cmd
 
    jobscript = 'hpc/submit_'+ jobname + '.sh'
    f = open(jobscript, 'w')
    f.write(strcmd)
    f.close()
    os.system('bsub < ' + jobscript)
 
if __name__ == "__main__":
    animal = 'm1'
    ear = 'l'
    version = 'v_dec1_a_temp9'
    sampler = 'NUTS'
    unknown_par_type = 'smooth'
    unknown_par_value = [400, 1200]
    unknown_par_value_str1 = str(unknown_par_value[0])+' '+str(unknown_par_value[1]) 
    unknown_par_value_str2 = str(unknown_par_value[0])+'_'+str(unknown_par_value[1])
    data_type = 'real'
    inference_type = 'both'
    Ns_const = 20
    Ns_var = 20
    noise_level = 0.1
    data_pts_type = 'CA'
    cmd = "python3 demo_aqueduct.py -animal "+animal+" -ear "+ear+" -version "+version+" -sampler "+sampler+" -unknown_par_type "+unknown_par_type+" -unknown_par_value "+unknown_par_value_str1+" -data_type "+data_type+" -inference_type "+inference_type+" -Ns_const "+str(Ns_const)+" -Ns_var "+str(Ns_var)+" -noise_level "+str(noise_level)+" -data_pts_type "+data_pts_type

    print(cmd)

    tag = animal+'_'+ear+'_'+sampler+'_'+unknown_par_type+'_'+unknown_par_value_str2+'_'+data_type+'_'+inference_type+'_'+str(Ns_const)+'_'+str(Ns_var)+'_'+str(noise_level)+'_'+version+'_'+data_pts_type

    submit(tag,cmd)
