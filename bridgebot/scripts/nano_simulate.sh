python app.py \
	--debug \
	--run_simulate \
	--model_dir=gs://njt-serene-epsilon/models/rl-tpu-nano \
	--bridgebot_config_file=bridgebot_config_nano.json \
	--num_parallel_simulations=1 \
	--num_parallel_inferences=1 \
	--num_simulations_per_move=1 \
	--simulation_batch_size=1 \
	--new_boards_prefix=./data/nano_boards/boards
