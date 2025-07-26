python3 app.py \
 --do_train \
 --learn_supervised=true \
 --bridgebot_config_file=bridgebot_config_small.json \
 --model_dir=gs://njt-serene-epsilon/models/sl-small \
 --supervised_dataset_location=gs://njt-serene-epsilon/sl-dataset-v2
