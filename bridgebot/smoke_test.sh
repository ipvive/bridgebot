../bazel-bin/bridgebot/replay_buffer/linux_amd64_stripped/replay_buffer
python3 selfplay_test.py
python3 train.py --mubert_config_file=mubert_config_tiny.json --output_dir=/home/njt/models/scratch --do_train --learn_supervised=true
python3 train.py --mubert_config_file=mubert_config_tiny.json --output_dir=/home/njt/models/scratch --do_train --learn_supervised=false
python3 app.py --run_simulate --model_dir=/home/njt/models/scratch --mubert_config_file=mubert_config_tiny.json -num_parallel_simulations=6 --batch_size=3

git clone -b training_pipeline --single-branch --depth 1 git@github.com:bridgebot.git
pip3 install -r requirements.txt
python3 train.py --mubert_config_file=mubert_config_tiny.json --output_dir=/home/njt/models/scratch --do_train --learn_supervised=true --beam_staging_location=/tmp --beam_temp_location=/tmp --supervised_dataset_location=/tmp/foo/bar --train_input_files=gs://njt-serene-epsilon/tfrecord-lin/00.rec --eval_input_files=gs://njt-serene-epsilon/tfrecord-lin/90.rec


python3 train.py --mubert_config_file=mubert_config_tiny.json --output_dir=models/scratch --do_train --learn_supervised=false --beam_num_workers=1 --beam_direct_run_mode=multi_processing --sample_batch_size=1000 --beam_runner=dataflow --beam_temp_location=gs://njt-serene-epsilon/beam-temp --replay_buffer_address=10.128.0.7:10000 --tfrecord_temp_dir=gs://njt-serene-epsilon/RL-temp

# prepare SL dataset
python3 app.py --mubert_config_file=mubert_config_tiny.json --model_dir=gs://njt-serene-epsilon/models/scrach --do_train --learn_supervised=true --beam_num_workers=19 --train_input_files=gs://njt-serene-epsilon/tfrecord-lin/[0-8]?.rec --eval_input_files=gs://njt-serene-epsilon/tfrecord-lin/9?.rec --beam_temp_location=gs://njt-serene-epsilon/beam-temp --beam_staging_location=gs://njt-serene-epsilon/staging --supervised_dataset_location=gs://njt-serene-epsilon/sl-dataset-v2 --beam_runner=dataflow --beam_job_name=prepare-sl-dataset-v2

# local prepare SL dataset
python3 app.py --mubert_config_file=mubert_config_tiny.json --model_dir=gs://njt-serene-epsilon/models/scrach --do_train --learn_supervised=true --beam_num_workers=2 --train_input_files=gs://njt-serene-epsilon/tfrecord-lin/88.rec --eval_input_files=gs://njt-serene-epsilon/tfrecord-lin/99.rec --beam_temp_location=gs://njt-serene-epsilon/beam-temp --supervised_dataset_location=gs://njt-serene-epsilon/sl-dataset-scratch --beam_direct_run_mode=multi_processing

# local top
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=jargon/uncased_L-2_H-128_A-2/vocab.txt --model_dir=gs://njt-serene-epsilon/models/scratch --do_train --beam_num_workers=1 --top_dataset_location=gs://njt-serene-epsilon/top-dataset 

# dataflow top datset variant 1
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/scratch --do_train --beam_num_workers=19 --train_lin_files=gs://njt-serene-epsilon/tfrecord-lin/[0-8]?.rec --corpus_files=gs://njt-serene-epsilon/corpus/* --eval_lin_files=gs://njt-serene-epsilon/tfrecord-lin/9?.rec --beam_temp_location=gs://njt-serene-epsilon/beam-temp --beam_staging_location=gs://njt-serene-epsilon/staging --top_dataset_location=gs://njt-serene-epsilon/top-dataset --beam_runner=dataflow --beam_job_name=prepare-top-dataset 

# dataflow top dataset variant 2
comment_regex="[1-7]@[hHsScCdDnN]"
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/scratch --do_train --beam_num_workers=19 --train_lin_files=gs://njt-serene-epsilon/tfrecord-lin/[0-8]?.rec --corpus_files=gs://njt-serene-epsilon/lin_text_corpus --eval_lin_files=gs://njt-serene-epsilon/tfrecord-lin/9?.rec --beam_temp_location=gs://njt-serene-epsilon/beam-temp --beam_staging_location=gs://njt-serene-epsilon/staging --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-bids-redacted --beam_runner=dataflow --beam_job_name=prepare-top-dataset-lin-bids-redacted --comment_regex="$comment_regex" --num_merge_keys=10000 --redact_commentator_names

# local top dataset test
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/scratch --do_train --train_lin_files=gs://njt-serene-epsilon/tfrecord-lin/[0-8]?.rec --corpus_files=gs://njt-serene-epsilon/lin_text_corpus --eval_lin_files=gs://njt-serene-epsilon/tfrecord-lin/9?.rec --top_dataset_location=scratch

# train top on TPU for top_io loded model
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --init_checkpoint=gs://njt-serene-epsilon/models/top.small.0/pretrained --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-bids --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-bids --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=1000000 --log_dir=logs

# train top on TPU
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-bids --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-bids --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=1000000 --log_dir=logs

# quick run top flags with TPU
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --init_checkpoint=gs://njt-serene-epsilon/models/top.small.0/pretrained --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-selected-nnnn --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-selected --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=5000 --log_dir=logs --freeze_player=false --freeze_jargon=false
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --init_checkpoint=gs://njt-serene-epsilon/models/top.small.0/pretrained --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-selected-yyyy --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-selected --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=5000 --log_dir=logs --freeze_player=true --freeze_jargon=true --use_jargon_sequence_out=true --use_player_sequence_out=true
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --init_checkpoint=gs://njt-serene-epsilon/models/top.small.0/pretrained --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-selected-nyyy --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-selected --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=5000 --log_dir=logs --freeze_player=true --freeze_jargon=true --use_jargon_sequence_out=true --use_player_sequence_out=false

# run top eval
python3 top_app.py --bridgebot_config=bridgebot_config_small.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/top.small.0/lin-bids --do_eval --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-bids

# run SL with tpu.
python3 app.py --mubert_config_file=mubert_config_small.json --model_dir=gs://njt-serene-epsilon/models/small.3 --do_train --learn_supervised=true --beam_num_workers=1 --use_tpu=true --tpu_name=njt --gcp_zone=us-central1-b --supervised_dataset_location=gs://njt-serene-epsilon/sl-dataset

# run RL with tpu.
# ctpu up -name njt -machine-type n1-highmem-4 -preemptible
  # [1] replay buffer
bazel build bridgebot/replay_buffer/...
bazel-bin/bridgebot/replay_buffer/linux_amd64_stripped/replay_buffer -persist_prefix=/home/njt/replay_buffer/random/record
  # [2] populate buffer
cd bridgebot
python3 replay_buffer/test/stress.py
  # [3] train
cd bridgebot
python3 app.py --bridgebot_config_file=bridgebot_config_small.json --model_dir=gs://njt-serene-epsilon/models/rl-tpu-small --init_checkpoint=gs://njt-serene-epsilon/models/sl-small --do_train --learn_supervised=false --beam_num_workers=1 --beam_direct_run_mode=multi_processing --use_tpu=true --tpu_name=selfplay --gcp_zone=us-central1-b --tfrecord_temp_dir=gs://njt-serene-epsilon/rl-input-data
  # [4] tensorboard
tensorboard --logdir gs://njt-serene-epsilon/models/tpu.rl.random

# run selfplay with tpu
python3 app.py --run_simulate --model_dir=gs://njt-serene-epsilon/models/rl-tpu-small --bridgebot_config_file=bridgebot_config_small.json --num_parallel_simulations=2048 --simulation_batch_size=1024 --batch_size=1024 --use_tpu=true --tpu_name=selfplay --gcp_zone=us-central1-b 
