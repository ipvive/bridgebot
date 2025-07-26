ctpu up -name sim-0 --tf-version=2.3.1 -vm-only -machine-type n1-standard-16
cat > CONFIG.sh <<'EOF'
export branchname=selfplay-v1b
export hostname=sim-0
export modelname=fake
export epochname=0b.uniform-goodscores
export datadir=/home/njt/data
export epochdir=$datadir/boards-$epochname
export screenlogdir=$datadir/screenlog
EOF
source ~/CONFIG.sh

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
git checkout $branchname
pip3 install --upgrade pip
pip3 install -r requirements.txt
../bazelisk build ...
mkdir -p $screenlogdir
mkdir -p $epochdir

screen -S rb -L -Logfile $screenlogdir/rb
source ~/CONFIG.sh
../bazel-bin/bridgebot/replay_buffer/linux_amd64_stripped/replay_buffer \
       	-persist_prefix=$epochdir/boards

screen -S pipe -L -Logfile $screenlogdir/pipe
source ~/CONFIG.sh
../bazel-bin/bridgebot/inference_pipe/linux_amd64_stripped/inference_pipe

screen -S fakeinfer -L -Logfile $screenlogdir/fakeinfer
source ~/CONFIG.sh
../bazel-bin/bridgebot/inference_pipe/fakeinfer/linux_amd64_stripped/fakeinfer \
	-batch_size 1000

screen -S sim -L -Logfile $screenlogdir/sim
source ~/CONFIG.sh
ps auxww | grep simulate.py | awk '{print $2}' | xargs kill
for i in $(seq 400)
do
	python3 simulate.py --num_simulations_per_move=1 &
done
