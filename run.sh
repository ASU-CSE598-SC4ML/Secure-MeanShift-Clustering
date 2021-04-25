#!/bin/bash

n_point=100
n_dust=8
n_center=8
radius=0.1

for iter_time in 0
do
    for n_point in 1000
    do
        python main.py --n_point=${n_point} --n_dust=${n_dust} --n_center=${n_center} --radius=${radius}
    done
done
#When finished, open a new terminal and type: "nvvp &" to open nvidia visual profiler. Open generated sql files in the GUI.
