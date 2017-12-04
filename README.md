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
run.py
#python predict.py --run_name=run_1 --ckpt=100 --trunk=net_2 --optimizer=momentum --predict_train=True
```
This model gives the following results:
Train Accuracy: 1.00
Test Accuracy:  0.77

## Improve performance with resnet20
Accuracy 0.90 was obtained with an original resnet20 with momentum optimizer - 0.1 for 50 epochs, then 0.01 for 10, 
batch_size = 256, decay_bn = 0.9, shortcut - projection. I faced troubles with training a net with adam
```bash

```


To obtain a reproducible results one has to use specified seed and only one thread to fetch data form files
```bash
python run.py --run_mode=train --run_name=resnet20_reproducible --trunk=resnet20 --optimizer=momentum --learning_rate=0.1 --train_batch_size=128 --decay_bn=0.9 --num_threads=1 --weight_decay=0.0001 --random_seed_tf=1

```
