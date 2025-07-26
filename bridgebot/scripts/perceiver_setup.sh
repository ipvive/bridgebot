#!/bin/bash

export PYTHONPATH=`pwd`
python3 -m pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install jaxline==0.0.4 dm-haiku==0.0.4 optax einops tensorflow-datasets apache-beam
# sudo pip3 uninstall tb-nightly

mkdir -p $HOME/tensorflow_datasets
gsutil -m cp -r gs://njt-serene-epsilon/fsa_builder $HOME/tensorflow_datasets/
cd bridge/fastgame
python3 setup.py install --user
