#!/bin/bash -l
#
# Multithreading example job script for MPCDF Raven.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# parallel Matlab, Julia, and similar cases.
#
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH -J eppRF300
#BATCH  --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72   # assign all the cores to that first task to make room for multithreading
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=none
#SBATCH --mail-user=gholamia@mpip-mainz.mpg.de

module purge
module load intel/21.6.0 impi/2021.6
module load anaconda/3/2021.11
module load boost-mpi/1.79 fftw-mpi/3.3.10 hdf5-mpi/1.12.2
module load gromacs/2018.8

# set number of OMP threads *per process*
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset I_MPI_PMI_LIBRARY 
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0   # the option -ppn only works if you set this before

# 0 in the following command line means starting a simulation
mpirun -n 27 ./run.py 0

