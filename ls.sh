WORKFLOW_TOP_DIR="/home/vance/espressopp/hpx4espp-paper-workflow"
ESPP_VEC_SOURCE_DIR="${WORKFLOW_TOP_DIR}/espressopp/vec/boost-1.72.0-SciPy-bundle-2020.03-intel-2020a-Python-3.8.2/SOURCE_ME.sh"
ESPP_HPX_SOURCE_DIR="${WORKFLOW_TOP_DIR}/espressopp/hpx4espp/hpx-1.5.1-boost-1.72.0-SciPy-bundle-2020.03-intel-2020a-Python-3.8.2/SOURCE_ME.sh"
ESPP_HPX_NO_VEC_SOURCE_DIR="${WORKFLOW_TOP_DIR}/espressopp/hpx4espp/hpx-1.5.1-boost-1.72.0-SciPy-bundle-2020.03-intel-2020a-Python-3.8.2-no_vec/SOURCE_ME.sh"
ESPP_EVAL_DIR="${WORKFLOW_TOP_DIR}/evaluation"

CMD_PRE="srun --nodes=1"
CMD_MPI_PRE="--ntasks-per-node=32 --cpus-per-task=2"

DATEF=$(date "+%Y-%m-%d--%H-%M-%S-%N")
echo DATEF=${DATEF}
OUT_BASE="${0##*/}-${DATEF}"
echo OUT_BASE=${OUT_BASE}

OUT_ARGS="--outBase ${OUT_BASE}"

SIM_TYPE_LIST=("lj" "pm")

RUN_TYPE_LIST=("orig" "soa" "vec")
SOURCES_LIST=(${ESPP_HPX_SOURCE_DIR} ${ESPP_HPX_NO_VEC_SOURCE_DIR} ${ESPP_HPX_SOURCE_DIR})

MPI_NP_LIST=(32 16 8 4 2 1)

# loop through number of mpi procs
for MPI_NP in "${MPI_NP_LIST[@]}";
do
    # loop through simulation types (lj and pm)
    for SIM_TYPE in "${SIM_TYPE_LIST[@]}";
    do
        SIM_ARGS="--simulation ${SIM_TYPE}"

        # loop through run types ("orig" "soa" "vec")
        for (( i = 0; i < ${#RUN_TYPE_LIST[@]}; i++ )); do
            echo '================================================================================'
            SOURCE_ME=${SOURCES_LIST[$i]}
            echo SOURCE_ME=$SOURCE_ME
            source $SOURCE_ME
            export PYTHONPATH="${PYTHONPATH}:${ESPP_EVAL_DIR}"
            echo '--------------------------------------------------------------------------------'

            RUN_TYPE=${RUN_TYPE_LIST[$i]}
            EXE_ARGS="--runType ${RUN_TYPE}"

            CMD="${CMD_PRE} -n ${MPI_NP} ${CMD_MPI_PRE} python lj_pm.py ${SIM_ARGS} ${EXE_ARGS} ${OUT_ARGS}"

            echo CMD=\"${CMD}\"
            echo
            eval $CMD
            echo

        done
    done
done