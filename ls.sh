ESPP_EVAL_DIR="/home/zhexu/scratch/epp_job/eval_hpx/eval_short/"
module load data/h5py/2.10.0-intel-2020a-Python-3.8.2
module load numlib/FFTW/3.3.8-intel-2020a
module load numlib/GSL/2.6-iccifort-2020.1.217
export BOOST_BUILD=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_72_intel2020/build
export BOOST_BUILD_COMMON=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_72_intel2020/
export BOOST_ROOT=/lustre/miifs01/project/m2_zdvresearch/zhexu/local.boost_1_72_intel2020/install
export ESPPATH=/home/zhexu/scratch/build-epp-jv-intel2020
source /home/zhexu/bin/esprc

dtmp=$(date "+%N")
DATEF=$(date "+%y%m%d-%H%M%S")
DATEF=${DATEF}${dtmp: -3}
echo DATEF=${DATEF}
OUT_BASE="${0##*/}-${DATEF}"
echo OUT_BASE=${OUT_BASE}

CMD_PRE="srun --nodes=1"
CMD_MPI_PRE="-p parallel -A m2_zdvresearch -C skylake --ntasks-per-node=32 --cpus-per-task=2"

OUT_ARGS="--outBase ${OUT_BASE}"

SIM_TYPE_LIST=("lj") # "pm")

RUN_TYPE_LIST=("orig" "soa" "vec")
#SOURCES_LIST=(${ESPP_HPX_SOURCE_DIR} ${ESPP_HPX_NO_VEC_SOURCE_DIR} ${ESPP_HPX_SOURCE_DIR})

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
            #echo '================================================================================'
            #SOURCE_ME=${SOURCES_LIST[$i]}
            #echo SOURCE_ME=$SOURCE_ME
            #source $SOURCE_ME
            #export PYTHONPATH="${PYTHONPATH}:${ESPP_EVAL_DIR}"
            #echo '--------------------------------------------------------------------------------'

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
