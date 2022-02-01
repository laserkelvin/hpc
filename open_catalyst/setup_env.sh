#!/bin/bash

# assume you have a working anaconda installation
# without the `ocp-models` environment spec
conda create -n ocp-models python=3.8
conda activate ocp-models

conda install -c intel \
	mkl \
	mkl-service \
	impi-devel \
	numpy

pip install black \
	scipy \
	ase \
	pymatgen==2020.12.31 \
	tensorboard \
	tqdm \
	demjson \
	Pillow \
	wandb \
	lmdb \
	submitit \
	psutil

# specific wheels for important stuff
pip install torch -f https://download.pytorch.org/whl/cpu/torch-1.10.0%2Bcpu-cp38-cp38-linux_x86_64.whl
pip install intel_extension_for_pytorch=1.10.100 -f https://software.intel.com/ipex-whl-stable

# this may give troubles
pip install torch-sparse torch-scatter torch-geometric
# install the OCP modules
pip install -e .

