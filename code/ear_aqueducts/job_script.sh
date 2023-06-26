#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J Job_arg1arg2
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 1GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 03:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u amaal@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_arg1arg2_%J.out 
#BSUB -e Output_arg1arg2_%J.err 

# here follow the commands you want to execute with input.in as the input file
module load python3/3.10.2 
numpy/1.22.2-python-3.10.2-openblas-0.3.19 
module load scipy/1.7.3-python-3.10.2
###pip3 install --upgrade pip
###pip3 install -e /zhome/0f/0/161811/tools/CUQIpy

python3 demo_aqueduct.py arg1 arg2
