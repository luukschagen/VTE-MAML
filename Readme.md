# Variational Task Encoders for Model Agnostic Meta Learning

This repository contains the code for the Master's Thesis by Luuk Schagen in fulfillment of the requirements for the degree of Master of Science in Data Science & Entrepreneurship 
at the Jheronimus Academy of Data Science.

## Abstract

*This work proposes an extension to recent model agnostic meta-learning methods to improve reasoning about uncertainty under multimodal task distributions. 
Meta-learning concerns learning over a dataset of many similar tasks, so that experience from previous tasks can be generalized to new tasks, so that they 
can be learned with few new data points. In practice, this often means that there is uncertainty about the identity of the newly learned tasks, especially 
when this task is not completely exchangeable with the original task distribution. Furthermore, it can be difficult in practice to accurately define which 
tasks can be considered similar enough to the original task distribution to effectively meta-learn from. This introduces additional uncertainty in the 
predictions on new tasks, as now the usefulness of new information to the problem is itself uncertain. This thesis proposes a method that aims to solve these 
issues by introducing methods of variational inference, without making strong assumptions about the underlying task distribution. This is achieved by modelling 
the identity of the task as a latent random variable, which modulates the fine-tuning of a meta-learned neural network. In doing so, the algorithm can meaningfully 
reason about the identity of new tasks, resulting in better calibrated uncertainty measures and more reasonable behavior for tasks lying outside the original 
task distribution. This is achieved with minimal detriment to model performance.*

## Data used

This repository contains code for three benchmark experiments that were used to evaluate the algorithm. A synthetic regression experiment, a synthetic classification
experiment, and the MiniImagenet Benchmark. The synthetic benchmarks are generated on the fly while the training is in progress, so they require no additional data.
The process by which they're generated is outlined in the Thesis, and can be inspected in the `datasets` folder of this repository

For training and testing on the MiniImagenet dataset the image files need to be downloaded. The .csv files in the `datasets/MiniImagenet` folder represent
the canonical train-validation-test-split used in previous literature. In order for the code to run, all 60,000 MiniImagenet images should be placed in the `images` folder
within the abovementioned folder. These can be acquired by from this [direct download link](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view),
or by following [these instructions](https://github.com/yaoyao-liu/mini-imagenet-tools).

## Model testing instructions

This repository contains the trained models for each of the benchmarks. The code for loading these models and applying them on the tests can be found in the `Experiments`
folder, in the form of three Jupyter Notebooks. This includes the benchmarks for MAML, Multimodal-MAML and the newly introduced VTE-MAML on the abovementioned datasets.
Trained models for PLATIPUS are not available, as this was not reimplemented, and the original implementation was used, which I was given acces to, 
courtesy of Chelsea Finn and Kelvin Xu.

Instead of loading these trained models, the training procedure can be rerun by running the commands outlined below.

## Model Training Instructions

Listed below are the exact commands required for training all models contained within this repository. This excludes the benchmarks trained on the PLATIPUS
model, which was not reimplemented in this work, but used in its original implementation, which I was given access to, courtesy of Chelsea Finn and Kelvin Xu.

### Regression

#### 5 shot 5 modes

##### MAML

`python main.py regression maml --config model_configs/regression.json --task_num 12500 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 5`

##### MMAML

`python main.py regression mmaml --config model_configs/regression.json --task_num 12500 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 5 --output_dimensions 100 100 100 --decoder_layers 3`

##### VTE-MAML

`python main.py regression vte-maml --config model_configs/regression.json --task_num 12500 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.0001 --inner_updates 5 --validation_updates 10 --modes 5 --output_dimensions 100 100 100 --decoder_layers 3`

#### 5 shot 3 modes

##### MAML

`python main.py regression maml --config model_configs/regression.json --task_num 7500 --meta_batch_size 75 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 3`

##### MMAML

`python main.py regression mmaml --config model_configs/regression.json --task_num 7500 --meta_batch_size 75 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 3 --output_dimensions 100 100 100 --decoder_layers 3`

##### VTE-MAML

`python main.py regression vte-maml --config model_configs/regression.json --task_num 7500 --meta_batch_size 75 --k_shot 5 --k_query 15 --inner_lr 0.001 --outer_lr 0.0001 --inner_updates 5 --validation_updates 10 --modes 3 --output_dimensions 100 100 100 --decoder_layers 3`

### Classification

#### 3 shot 3 modes

##### MAML

`python main.py classification maml --config model_configs/classification.json --task_num 12500 --meta_batch_size 125 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 3 --report_accuracy`

##### MMAML

`python main.py classification mmaml --config model_configs/classification.json --task_num 12500 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 3 --output_dimensions 100 100 100 --decoder_layers 3 --report_accuracy`

##### VTE-MAML

`python main.py classification vte-maml --config model_configs/classification.json --task_num 12500 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 3  --output_dimensions 100 100 100 --decoder_layers 3 --report_accuracy`

#### 3 shot 2 modes

##### MAML

`python main.py classification maml --config model_configs/classification.json --task_num 12500 --meta_batch_size 125 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 2`

##### MMAML

`python main.py classification mmaml --config model_configs/classification --task_num 12500 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 2 --output_dimensions 100 100 100 --decoder_layers 3`

##### VTE-MAML

`python main.py classification vte-maml --config model_configs/regression.json --task_num 12500 --k_shot 6 --k_query 6 --inner_lr 0.001 --outer_lr 0.001 --inner_updates 5 --validation_updates 10 --modes 2  --output_dimensions 100 100 100 --decoder_layers 3`

### MiniImagenet

#### 1 shot 5 way image classification
##### MAML
`python main.py imagenet maml --config model_configs/imagenet.json --task_num 1000 --k_shot 1 --k_query 15 --validation_tasks 20 --validation_samples 100 --inner_lr 0.01 --outer_lr 0.005 --inner_updates 5 --validation_updates 10 --loss crossentropy --device cuda --epochs 600 --meta_batch_size 4 --report_accuracy --n_way 5`

##### MMAML
`python main.py imagenet mmaml --config model_configs/imagenet.json --task_num 1000 --k_shot 1 --k_query 15 --validation_tasks 20 --validation_samples 100 --inner_lr 0.01 --outer_lr 0.005 --inner_updates 5 --validation_updates 10 --loss crossentropy --device cuda --epochs 600 --meta_batch_size 4 --report_accuracy --n_way 5 --convolutional_layers 32 64 128 256 --output_dimensions 32 32 32 32 --decoder_layers 3`

##### VTE-MAML
`python main.py imagenet vte-maml --config model_configs/imagenet.json --task_num 1000 --k_shot 1 --k_query 15 --validation_tasks 20 --validation_samples 100 --inner_lr 0.001 --outer_lr 0.0005 --inner_updates 5 --validation_updates 10 --loss crossentropy --device cuda --epochs 600 --meta_batch_size 4 --report_accuracy --n_way 5 --convolutional_layers 32 64 128 256 --output_dimensions 32 32 32 32 --decoder_layers 3`


## Credits and Thanks
The code in this repository is made available under MIT License (see `LICENSE.md`). 

The implementation of MAML and its variations in this repository owes much to the [PyTorch implementation of MAML](https://github.com/dragen1860/MAML-Pytorch)
by Jackie Loong.

