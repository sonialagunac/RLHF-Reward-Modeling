#!/bin/bash

for i in {0..19}
# for i in {0..4}
do
    echo "Submitting batch index $i"
    sbatch generate_labels_concepts_llama.sh $i 20
done
