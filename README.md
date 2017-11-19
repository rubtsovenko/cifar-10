# CIFAR-10

## Running a model
The easiest way to start working with this model is to install anaconda and execute provided .sh file:
```bash
python CifarLoader.py

./create_environment.sh
```

This script creates a virtual environment cifar10_env and a kernel cifar10_kernel for a jupyter notebook.
Then you should just run jupyter and open  Run_model.ipynb with created cifar10_kernel.

## Training simple baseline
```bash
source activate cifar10_env
python train.py --run_name=run_1 --ckpt=0 --trunk=net_2 --num_epochs=2 --optimizer=momentum --learning_rate=1e-3 --weight_decay=0.004 --keep_prob=0.75 --tf_random_seed=1 --np_random_seed=1
#python train.py --run_name=run_1 --ckpt=90 --trunk=net_2 --num_epochs=10 --optimizer=momentum --learning_rate=1e-4 --weight_decay=0.004 --keep_prob=0.75 --tf_random_seed=1 --np_random_seed=1
#python predict.py --run_name=run_1 --ckpt=100 --trunk=net_2 --optimizer=momentum --predict_train=True
```
This model gives the following results:
Train Accuracy: 1.00
Test Accuracy:  0.77