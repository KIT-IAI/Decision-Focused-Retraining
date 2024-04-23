#!/bin/bash

for i in {101..300}

do
  python src/main_retraining_MSE.py -id $i  -run_name "r_MSE" 
  python src/main_optimisation.py -id $i  -run_name "r_MSE"
  

done