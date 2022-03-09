#!/bin/bash

set -m

function exitall() {
   echo 'Terminated'
   pkill -f run_mpc # kill all processes containing -f "NAME"
   # NOTE: copy and paste above command in shell if does not work
}

trap exitall SIGINT SIGTERM 

for i in {1..10}
do
   for exp in EulerAccurate MidpointAccurate MidpointInaccurate MultistageHypersolver
   do
      echo "Running experiment $exp number $i"
      (/bin/python3.8 02_run_mpc.py --number $i --experiment $exp & ) # use multiprocessing
   done
done





