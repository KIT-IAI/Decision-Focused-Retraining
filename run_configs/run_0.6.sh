#!/bin/bash

for i in {101..300}

do
  python src/main_retraining_Pinball.py -id $i  -run_name "r_0.6" -quantile 0.6
  python src/main_optimisation.py -id $i  -run_name "r_0.6"
  

done