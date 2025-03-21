#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1 
#SBATCH --ntasks=8   

module purge
module load foss/2022a
module load Python/3.11.3
module load GCC/11.3.0
module load mpi4py/3.1.4-Python-3.11.3

srun -n 8 python3 case2.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s