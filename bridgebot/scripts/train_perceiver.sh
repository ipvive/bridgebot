#!/bin/bash

for x in scaffold/perceiver/config/c0base.py; do
    python3 scaffold/perceiver/experiment.py \
	    --output_base_dir=gs://njt-serene-epsilon/scaffold_perceiver_experiment_out \
	    --config="$x"
done
