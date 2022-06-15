#!/bin/bash 

#SBATCH --workdir /data/skloth/sim/ionic_liquids_MPIP/AA/new_sim/T300_nvt
#SBATCH --job-name AA_C4_PF6_300nvt
#SBATCH --partition=gpu-nodes 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8 

module purge 
module load gromacs/2018.3-CUDA-8.0 

WORK_DIR="/data/skloth/sim/ionic_liquids_MPIP/AA/new_sim/T300_nvt"
mkdir $WORK_DIR/out

export GMX_MAXCONSTRWARN=-1; 
gmx grompp \
    -f $WORK_DIR/parameters.mdp \
    -o $WORK_DIR/run.tpr \
    -c $WORK_DIR/start.gro \
    -r $WORK_DIR/start.gro \
    -p $WORK_DIR/topology.top \
    -po $WORK_DIR/out/mdout.mdp \
    -maxwarn 5 -quiet yes 

gmx mdrun \
    -s $WORK_DIR/run.tpr \
    -o $WORK_DIR/out/traj.trr \
    -c $WORK_DIR/out/out.gro \
    -x $WORK_DIR/out/traj_full.xtc \
    -e $WORK_DIR/out/energy.edr \
    -g $WORK_DIR/out/log.log \
    -cpo $WORK_DIR/out/state.cpt \
    -cpi $WORK_DIR/out/state.cpt \
    -cpt 5 \
    -v -nt 8 -quiet yes
