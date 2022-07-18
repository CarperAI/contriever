#!/bin/bash

# Usage: 
# sh tokenization_pile_script.sh -i {input_dir} -o {output_dir} -s {"train"|"val"|"test"}
# -i: Input directory containing the uncompressed pile (`-s`) file(s) to be tokenized.
#     The pile directory is assumed to be strutctured as: 
#     train /
#       {00-29}.jsonl
#     val.jsonl
#     test.jsonl
# -o: The ouput directory where the tokenized files will be stored.
# -s: The split to be tokenized {"train"|"val"|"test"}.
while getopts i:o:s: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_dir=${OPTARG};;
        s) split=${OPTARG};;
    esac
done

# Tokenize a split of `The Pile`
dir="$(dirname -- "$0")"
if [ "$split" == "train" ]; then
    # for i in 0{0..9} {10..29} ; do
    for FILE in $input_dir/train/*; do
        sh $dir/tokenization_script.sh -n $FILE -o $output_dir
    done
elif [ "$split" == "val" ]; then
    sh $dir/tokenization_script.sh -n "${input_dir}/val.jsonl" -o $output_dir
elif [ "$split" == "test" ]; then
    sh $dir/tokenization_script.sh -n "${input_dir}/test.jsonl" -o $output_dir
fi

# Example:
# sh tokenization_pile_script.sh -i /fsx/pile_raw -o /fsx/carper/contriever -s train
