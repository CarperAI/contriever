#!/bin/bash
#SBATCH --job-name="contriever"
#SBATCH --partition=compute-od-gpu
#SBATCH --cpus-per-task=5
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=/home/guac/contriever/logtrain/%A
#SBATCH --mem=450GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append

port=$(shuf -i 15000-16000 -n 1)
# NOTE: TDIR must point to the directory specified in `tokenization_pile_script.sh`
TRAIN_PATH=/fsx/carper/contriever
OUTPUT_DIR=$TRAIN_PATH/checkpoint/pile/$name
DATA_DIR=$TRAIN_PATH/encoded-data/bert-base-uncased
TRAIN_DATASETS="00"
# NOTE: Uncomment the line below to use the full dataset
#TRAIN_DATASETS=""
#for i in 0{0..9} {10..29} ; do
#    TRAIN_DATASETS+="${TDIR}/pile/{i}"
#done

rmin=0.05
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
mo=bert-base-uncased
mp=none

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

source $TRAIN_PATH/.env/bin/activate
cd $TRAIN_PATH

srun python train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAIN_DATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --moco_queue $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim shampoo \
        --per_gpu_batch_size 64 \
        --output_dir  $OUTPUT_DIR \
        --main_port $port \
