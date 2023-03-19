source $HOME/bin/intel.2020
source $HOME/bin/cmk.load
mload lib/jemalloc/5.2.1-GCCcore-9.3.0 system/hwloc/2.2.0-GCCcore-9.3.0
export HPX_DIR=/lustre/miifs01/project/m2_zdvresearch/zhexu/build-hpx-1.5.1-intel2020/local/lib64/cmake/HPX/
ESPP_EVAL_DIR="/lustre/miifs01/project/m2_zdvresearch/zhexu/build-h4e-intel2020-jv/evaluation"
export PYTHONPATH="${PYTHONPATH}:${ESPP_EVAL_DIR}"

dtmp=$(date "+%N")
DATEF=$(date "+%y%m%d-%H%M%S.")${dtmp: -3}
echo DATEF=${DATEF}
OUT_BASE="${0##*/}-${DATEF}"
echo OUT_BASE=${OUT_BASE}
SECONDS=0

###################################################################################################

NUM_TRIALS=3

echo '================================================================================'
echo '--------------------------------------------------------------------------------'

CMD_PRE="srun --nodes=1"
SIM_ARGS=""
OUT_ARGS="--outBase ${OUT_BASE}"

CMD_SLURM_PRE_MPI="--ntasks-per-node=32 --cpus-per-task=2"
CMD_SLURM_PRE_HPX="--ntasks-per-node=2 --cpus-per-task=32"

###################################################################################################

function run_cmd() {
    EXE_ARGS="--runType ${RUN_TYPE} --numSubs=${NUM_SUBS} --hpx:threads=${NUM_THREADS}  --trial ${TRIAL}"
    CMD="${CMD_PRE} -n ${NUM_RANKS} ${CMD_SLURM_PRE} python lj.py ${SIM_ARGS} ${EXE_ARGS} ${OUT_ARGS}"
    echo CMD=\"${CMD}\"
    echo
#    eval $CMD
    echo
    echo '--------------------------------------------------------------------------------'
}

function run_with_num_ranks() {
    ###############################################################################################
    ### hpx_off
    RUN_TYPE="hpx_off"
    NUM_RANKS=${NUM_RANKS_MPI}
    CMD_SLURM_PRE=${CMD_SLURM_PRE_MPI}
    NUM_THREADS=1
    NUM_SUBS=1

    run_cmd
exit 0
    ###############################################################################################
    ### hpx_on
    RUN_TYPE="hpx_on"
    NUM_RANKS=${NUM_RANKS_HPX}
    CMD_SLURM_PRE=${CMD_SLURM_PRE_HPX}
    NUM_THREADS=16

    for NUM_SUBS in ${NUM_SUBS_LIST[@]}
    do
        run_cmd
    done
}

function run_single_trial() {
    ###############################################################################################
    ## single socket
    NUM_RANKS_MPI=16
    NUM_RANKS_HPX=1
    NUM_SUBS_LIST=(16 32 64 128 256 512)

    run_with_num_ranks

    ###############################################################################################
    ## single node (2 sockets)
    NUM_RANKS_MPI=32
    NUM_RANKS_HPX=2
    NUM_SUBS_LIST=(16 32 64 128 256)

    run_with_num_ranks

    ###############################################################################################
}

for (( TRIAL = 0; TRIAL < $NUM_TRIALS; TRIAL++ ))
do
    echo '================================================================================'
    echo
    echo TRIAL=${TRIAL}
    echo
    run_single_trial
    echo
done
###################################################################################################

echo
echo '================================================================================'
echo
echo SECONDS=$SECONDS
echo
