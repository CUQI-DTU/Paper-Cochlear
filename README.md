# Collab-BrainEfflux


## To run on DTU HPC
  1. log in to compute machine
  2. `module load python3/3.10.2` then create a python environment (e.g. `python3 -m venv BE_collab`)
  3. pip install cuqipy (possibly from source)
  4. clone this repo
  5. update `code/ear_aqueduct/job_submit.py` with your email, python environemt name, and the parameters you need.
  6. run `python3 job_submit.py`
