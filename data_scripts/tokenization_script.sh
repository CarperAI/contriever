#!/bin/bash

# Usage: sh tokenization_script.sh -n {input_file} -o {out_dir}
while getopts n:o: flag
do
    case "${flag}" in
        n) input_file=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done
NSPLIT=128 #Must be larger than the number of processes used during training
FILENAME=$(basename "$input_file")
INFILE="${input_file}"
TOKENIZER=bert-base-uncased
#TOKENIZER=bert-base-multilingual-cased
SPLITDIR=./tmp-tokenization-${TOKENIZER}-${FILENAME}/
OUTDIR=${output_dir}/encoded-data/${TOKENIZER}/pile/"$(basename "$FILENAME" | sed 's/\(.*\)\..*/\1/')"
NPROCESS=8

mkdir -p ${SPLITDIR}
echo ${INFILE}
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}

pids=()

for ((i=0;i<$NSPLIT;i++)); do
    num=$(printf "%03d\n" $i);
    FILE=${SPLITDIR}${num};
    #we used --normalize_text as an additional option for mContriever
    python3 preprocess.py --tokenizer ${TOKENIZER} --datapath ${FILE} --outdir ${OUTDIR} &
    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done

echo ${SPLITDIR}

rm -r ${SPLITDIR}
