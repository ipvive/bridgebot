# build dataset
comment_regex="[1-7]@[hHsScCdDnN]"
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/scratch --do_train --beam_num_workers=1 --train_lin_files=gs://njt-serene-epsilon/tfrecord-lin/[0-8]?.rec --corpus_files=gs://njt-serene-epsilon/lin_text_corpus --eval_lin_files=gs://njt-serene-epsilon/tfrecord-lin/9?.rec --beam_temp_location=gs://njt-serene-epsilon/beam-temp --beam_staging_location=gs://njt-serene-epsilon/staging --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-just-bids --beam_runner=dataflow --beam_job_name=prepare-top-dataset-lin-just-bids --comment_regex="$comment_regex" --num_merge_keys=10000 --num_train_steps=1

# train on TPU
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_l-8_h-512_a-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/top.tiny.just-bids --do_train --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-just-bids --use_tpu --tpu_name=top --gcp_zone=us-central1-b --num_train_steps=1000000 --freeze_player=False --freeze_jargon=False --iterations_per_loop=20000 --save_checkpoints_steps=20000

# eval
python3 top_app.py --bridgebot_config=bridgebot_config_tiny.json --jargon_vocab_file=gs://njt-serene-epsilon/jargon/uncased_L-8_H-512_A-8/vocab.txt --model_dir=gs://njt-serene-epsilon/models/top.tiny.just-bids --do_eval --top_dataset_location=gs://njt-serene-epsilon/top-dataset-lin-just-bids
