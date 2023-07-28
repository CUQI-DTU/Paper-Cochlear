#!/bin/bash
RESULT_VERSION="v7"
SAMPLER="MH"

./sender.sh 'm1' 'l' $RESULT_VERSION $SAMPLER
./sender.sh 'm1' 'r' $RESULT_VERSION $SAMPLER
./sender.sh 'm2' 'l' $RESULT_VERSION $SAMPLER
./sender.sh 'm2' 'r' $RESULT_VERSION $SAMPLER
./sender.sh 'm3' 'l' $RESULT_VERSION $SAMPLER
./sender.sh 'm3' 'r' $RESULT_VERSION $SAMPLER
./sender.sh 'm4' 'l' $RESULT_VERSION $SAMPLER
./sender.sh 'm4' 'r' $RESULT_VERSION $SAMPLER
./sender.sh 'm6' 'l' $RESULT_VERSION $SAMPLER
./sender.sh 'm6' 'r' $RESULT_VERSION $SAMPLER
