#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run ESPR++ with MPI on Mogon II.
#
# This script requests 64 cores on two node. The job
# will have access to all the memory in the nodes.  
#-----------------------------------------------------------------

#SBATCH -J HPX_SOC02         # Job name
#SBATCH -o md.soc02.%j.out   # Specify stdout output file (%j expands to jobId)
#SBATCH -e md.soc02.%j.err   # Print out error msg
#SBATCH -p parallel       # Queue name
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=skylake
#SBATCH -t 24:00:00       # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A m2_trr146

date
echo -e "=============================================================="
scontrol show jobid -dd ${SLURM_JOB_ID}
echo -e "=============================================================="

#if [ -z "$LD_LIBRARY_PATH" ]; then
#  export LD_LIBRARY_PATH="/home/zhexu/Program/scafacos/b04/local04/lib"
#else
#  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/zhexu/Program/scafacos/b04/local04/lib"
#fi

nsub_total=32
n_per_core=$1
let "nsub_total=nsub_total*n_per_core"
#echo ${nsub_total}
#exit 0
# Load all necessary modules if needed
module load lang/Python/3.8.6-GCCcore-10.2.0
module load mpi/OpenMPI/4.1.1-GCC-11.2.0
module load data/HDF5/1.12.1-gompi-2021b
module load tools/Ninja/1.10.2-GCCcore-11.2.0
module load devel/CMake/3.22.1-GCCcore-11.2.0
module load tools/VTune
module load lib/jemalloc/5.2.1-GCCcore-11.2.0
export BOOST_BUILD=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_79_gcc11.2/build
export BOOST_BUILD_COMMON=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_79_gcc11.2/
export BOOST_ROOT=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_79_gcc11.2/install
export spdlog_DIR=/home/zhexu/scratch/build-spdlog-gcc11.2/local/lib64/cmake/spdlog/
export fmt_DIR=/home/zhexu/scratch/build-fmt-gcc11.2/local/lib64/cmake/fmt/
export HPX_DIR=/lustre/miifs01/project/m2_zdvresearch/zhexu/build-hpx-gcc11.2-b1.79
export MPICXX=mpic++

export ESPPATH=/lustre/miifs01/project/m2_zdvresearch/zhexu/build-epp-hpx-gcc11.2/
source /home/zhexu/bin/esprc
unset ESPPATH

# Launch the executable
srun /home/zhexu/scratch/build-epp-hpx-gcc11.2/john_examples/more_main --ghostCommunication_impl_john_opt=false --isteps=10000 --numSubs=$nsub_total --hpx:threads 16
#srun python test2.py


