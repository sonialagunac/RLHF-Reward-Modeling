#!/bin/bash

num_batches=10 # Change according to your needs
for batch_index in $(seq 0 $(($num_batches - 1)))
do
  sbatch stage2prep_datagen.sh $batch_index $num_batches
done