import os
from advection_diffusion_inference_utils import Args, create_experiment_tag
 
def submit(jobid,cmd):
    id = str(jobid)
    jobname = 'job_' + id
    memcore = 7000
    maxmem = 8000
    email = 'amaal@dtu.dk'
    ncores = 1
 
    # begin str for jobscript
    strcmd = '#!/bin/sh\n'
    strcmd += '#BSUB -J ' + jobname + '\n'
    strcmd += '#BSUB -q compute\n'
    strcmd += '#BSUB -n ' + str(ncores) + '\n'
    strcmd += '#BSUB -R "span[hosts=1]"\n'
    strcmd += '#BSUB -R "rusage[mem=' + str(memcore) + 'MB]"\n'
    strcmd += '#BSUB -M ' + str(maxmem) + 'MB\n'
    strcmd += '#BSUB -W 52:00\n'
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

def create_command(main_command, args):
    if isinstance(args.unknown_par_value, list):
        if len(args.unknown_par_value)>=2:
            unknown_par_value_str = str(args.unknown_par_value[0])+' '+str(args.unknown_par_value[1])
        elif len(args.unknown_par_value)==1:
            unknown_par_value_str = str(args.unknown_par_value[0])
        else:
            raise Exception
    else:
        unknown_par_value_str = str(args.unknown_par_value)

    if isinstance(args.add_data_pts, list) and len(args.add_data_pts)==0:
        add_data_pts_str = ' '
    elif isinstance(args.add_data_pts, list) and len(args.add_data_pts)>0:
        add_data_pts_str = ' '.join([str(i) for i in args.add_data_pts])
    else:
        raise Exception("Unknown args.add_data_pts type")
    if args.NUTS_kwargs is not None:
        # replace " with ' in the string
        NUTS_kwargs_str = str(args.NUTS_kwargs).replace("'", '"')

    cmd = main_command+" -animal "+args.animal+" -ear "+args.ear+" -version "+args.version+" -sampler "+args.sampler+" -unknown_par_type "+args.unknown_par_type+" -unknown_par_value "+unknown_par_value_str+" -data_type "+args.data_type+" -inference_type "+args.inference_type+" -Ns "+str(args.Ns)+" -Nb "+str(args.Nb)+" -noise_level "+str(args.noise_level)+" -add_data_pts "+ add_data_pts_str + " -num_CA "+str(args.num_CA)+" -num_ST "+str(args.num_ST) + " -true_a " + str(args.true_a) + " -rbc " + args.rbc + " -NUTS_kwargs " + "\'"+NUTS_kwargs_str+"\'" + " -data_grad " + str(args.data_grad) + " -u0_from_data " + str(args.u0_from_data) + " -sampler_callback " + str(args.sampler_callback) + " -pixel_data " + str(args.pixel_data) 
    return cmd
 
if __name__ == "__main__":
    args = Args()
    args.animal = 'm1'
    args.ear = 'l'
    args.version = 'v_dec1_a_temp9'
    args.sampler = 'NUTS'
    args.unknown_par_type = 'smooth'
    args.unknown_par_value = [400, 1200]
    args.data_type = 'real'
    args.inference_type = 'both'
    args.Ns = 20
    args.Nb = 10
    args.noise_level = 0.1
    args.num_ST = 0
    main_command = "python3 demo_aqueduct.py"

    cmd = create_command(main_command, args)
    print(cmd)

    tag = create_experiment_tag(args)

    submit(tag,cmd)