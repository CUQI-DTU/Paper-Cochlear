#!/bin/bash
sed -e "s/arg1/$1/g"  -e "s/arg2/$2/g" -e "s/arg3/$3/g" -e "s/arg4/$4/g" < job_script.sh | bsub
