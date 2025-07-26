ctpu up -name train-2-1-smp-2dyn --tf-version=2.3.1 -vm-only -machine-type n1-standard-16
ctpu pause -name train-2-1-smp-2dyn
ctpu up -name train-2-1-smp-2dyn --tf-version=2.3.1 -preemptible -machine-type n1-standard-16
cat > ~/CONFIG.sh <<'EOF'
export hostname=train-2-1-smp-2dyn
export modelname=2.1.smp-2dyn
export configpath=config/2/2.1.smp-2dyn.json
EOF

ssh-keyscan github.com > github-key
actualfp=$(ssh-keygen -lf github-key | sed 's/.*SHA256://;s/ .*//')
expectfp=$(curl -s https://api.github.com/meta | grep RSA | sed 's/.* "//;s/".*//')
if [ "$actualfp" = "$expectfp" ]
then
	cat github-key >> ~/.ssh/known_hosts
else
	echo "POTENTIAL MITM ATTACK."
	sleep 3600
fi

git clone git@github.com:ipvive/bridgebot.git
cd bridgebot
git checkout selfplay-v1
git pull
sudo pip3 install --upgrade pip
pip3 install -r requirements.txt
mkdir -p ~/data/screenlog
../bazelisk build ...
gsutil -m cp -r gs://njt-serene-epsilon/boards/v0/boards-0c.uniform-goodscores ~/data/

screen -S rb-eval -L -Logfile ~/data/screenlog/rb-eval
cd ~/bridgebot
source ~/CONFIG.sh
../bazel-bin/bridgebot/replay_buffer/replay_buffer_/replay_buffer \
	-serve_boards=../../data/boards-0c.uniform-goodscores/boards-00000? \
    -new_boards=/tmp/scratch-eval-N \
	-listen $hostname:10001

screen -S rb-train -L -Logfile ~/data/screenlog/rb-train
cd ~/bridgebot
source ~/CONFIG.sh
../bazel-bin/bridgebot/replay_buffer/replay_buffer_/replay_buffer \
	-serve_boards=../../data/boards-0c.uniform-goodscores/boards-* \
    -new_boards=/tmp/scratch-train-N \
	-listen $hostname:10000

screen -S train -L -Logfile ~/data/screenlog/train
cd ~/bridgebot
source ~/CONFIG.sh
python3 app.py \
	--bridgebot_config_file=$configpath \
	--model_dir=gs://njt-serene-epsilon/models/sl/0/$modelname/ \
	--do_train \
	--learn_supervised=false \
	--beam_num_workers=15 \
	--beam_direct_run_mode=multi_processing \
	--use_tpu=true \
	--tpu_name=$hostname \
	--gcp_zone=us-central1-b \
	--replay_buffer_address=$hostname:10000 \
	--tfrecord_temp_dir=gs://njt-serene-epsilon/scratch/rl-tfrecord-examples/$modelname \
	--sample_batch_size=5000 \
    --num_train_steps=1000000

screen -S eval -L -Logfile ~/data/screenlog/eval
cd ~/bridgebot
cat > do_eval.sh <<EOF
source ~/CONFIG.sh
last_ckpt=x
while true
do
	cur_ckpt=$(gsutil cat gs://njt-serene-epsilon/models/sl/0/$modelname/checkpoint | head -1 | sed 's/"$//;s/.*"//')
	if [ $cur_ckpt != $last_ckpt ]
	then
		last_ckpt=$cur_ckpt
		python3 app.py \
			--bridgebot_config_file=$configpath \
			--model_dir=gs://njt-serene-epsilon/models/sl/0/$modelname/ \
			--do_eval \
			--learn_supervised=false \
			--beam_num_workers=1 \
			--beam_direct_run_mode=multi_processing \
			--replay_buffer_address=$hostname:10001 \
			--tfrecord_temp_dir=gs://njt-serene-epsilon/scratch/rl-tfrecord-examples/$modelname/eval
	fi
	sleep 60
done
EOF
bash do_eval.sh

tail +1f data/screenlog/train | egrep -v 'Running|[wW]orker|function|refresh'

screen -S tensorboard
source ~/CONFIG.sh
tensorboard --logdir gs://njt-serene-epsilon/models/sl/0/$modelname/

source ~/CONFIG.sh
capture_tpu_profile --tpu $hostname \
    --logdir gs://njt-serene-epsilon/models/sl/0/$modelname/ \
    --duration_ms 62000
