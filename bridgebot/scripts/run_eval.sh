#!/bin/bash

configpath="config/2/2.1.smp-2dyn.json"

for modelname in 600000 700000 800000 900000; do
    python3 ./app.py \
        --bridgebot_config_file=$configpath \
        --model_dir=gs://njt-serene-epsilon/models/rl/epoch-1/smp-keep/$modelname/ \
        --do_eval \
        --learn_supervised=false \
        --beam_num_workers=1 \
        --beam_direct_run_mode=multi_processing \
        --replay_buffer_address=10.112.2.3:10000 \
        --tfrecord_temp_dir=gs://njt-serene-epsilon/scratch/rl-tfrecord-examples/$modelname/eval
done

