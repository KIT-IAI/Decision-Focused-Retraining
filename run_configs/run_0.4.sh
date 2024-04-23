#!/bin/bash

for i in {101..300}

do
  python src/main_retraining_Pinball.py -id $i  -run_name "r_0.4" -quantile 0.4
  python src/main_optimisation.py -id $i  -run_name "r_0.4"
  

done