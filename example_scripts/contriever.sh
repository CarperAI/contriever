#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --job-name=contriever
#SBATCH --output=/private/home/gizacard/contriever/logtrain/%A
#SBATCH --partition=learnlab
#SBATCH --mem=450GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append


port=$(shuf -i 15000-16000 -n 1)
# NOTE: TDIR must point to the directory specified in `tokenization_pile_script.sh`
TDIR="/fsx/carper/contriever/encoded-data/bert-base-uncased"
# If you want to run it on one shard of the training set, specify one dir:
# `TRAINDATASETS="${TDIR}/NN"`
# where "NN" can be anything from `00-29`.
TRAINDATASETS=""
for i in 0{0..9} {10..29} ; do
    TRAINDATASETS+="${TDIR}/pile/{i}"
done


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

# TODO: Rename environment path to your `contriever` python env.
srun ~gizacard/anaconda3/envs/contriever/bin/python3 train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --moco_queue $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size 64 \
        --output_dir /checkpoint/gizacard/contriever/xling/$name \
        --main_port $port \

