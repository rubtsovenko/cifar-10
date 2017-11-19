#!/bin/bash
conda create -n cifar10_env python=3.6
source activate cifar10_env
pip install -r requirements.txt
python -m ipykernel install --user --name cifar10_kernel --display-name "cifar10_kernel"
