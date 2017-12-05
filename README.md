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
python run.py --run_mode=train --run_name=baseline --ckpt=0 --trunk=net_2 --num_epochs=70 --optimizer=momentum --learning_rate=1e-3 --train_batch_size=128 --decay_bn=0.9 --weight_decay=0.004 --tf_random_seed=1
python run.py --run_mode=train --run_name=baseline --ckpt=70 --trunk=net_2 --num_epochs=30 --optimizer=momentum --learning_rate=1e-4 --train_batch_size=128 --decay_bn=0.9 --weight_decay=0.004 --tf_random_seed=1
python run.py --run_mode=predict --run_name=baseline --ckpt=100 --trunk=net_2
```
This model gives the following results:
Train Accuracy: 1.00
Test Accuracy:  0.77

## Improve performance with resnet20
Accuracy 0.90 was obtained with an original resnet20 with momentum optimizer - 0.1 for 50 epochs, then 0.01 for 30, 
batch_size = 128, decay_bn = 0.9, shortcut - projection, weight_decay wasn't used, also I investigated the influence of
a batch_size, there is no difference between 128 and 256, but lower leads to worse result, batch norm momentum has a little 
influence, 0.999 tends to lead to non stable results, training loss could have dropped at some point, the 0.9 gave fastest
convergence. I faced troubles with training a net with adam, it also wasn't stable and lead to overfitting. Data augmentation
helps a lot to prevent overfitting but train accuracy couldn't reach 100% accuracy.
```bash
source activate cifar10_env
python run.py --run_mode=train --run_name=resnet20 --ckpt=0 --trunk=resnet20 --num_epochs=50 --optimizer=momentum --learning_rate=0.1 --train_batch_size=128 --decay_bn=0.9 --weight_decay=0.0 --random_seed_tf=1
python run.py --run_mode=train --run_name=resnet20 --ckpt=50 --trunk=resnet20 --num_epochs=30 --optimizer=momentum --learning_rate=0.01 --train_batch_size=128 --decay_bn=0.9 --weight_decay=0.0 --random_seed_tf=1
python run.py --run_mode=predict --run_name=resnet20 --ckpt=80 --trunk=resnet20
```
This model gives the following results:
Train Accuracy: 0.956
Test Accuracy:  0.904