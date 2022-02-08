# Embed2Sym

This repo contains the code for the Embed2Sym framework 
and experiments presented in the paper "Embed2Sym - Scalable Neuro-Symbolic Reasoning via Clustered Embeddings".

## Requirements

The code was tested on Ubuntu 18.04, Python 3.8 and TensorFlow 2.7.0.<br />

## Installing

Simply clone the repo and install the dependencies appearing in the requirements.txt file.


## Running the Experiments

The run.py script can be used to execute the experiments appearing in the paper. 

* MNIST Addition

```
python run.py mnist_addition N
```
where N should be replaced with one of 1, 2, 3, 4 and 15.

* CIFAR-10 Addition

```
python run.py cifar10_addition 1
```

* Member

```
python run.py member N
```
where N should be replaced with one of 3, 4, 5 and 20.

Note that while the current script only runs these specific experiments, 
the framework is general and can easily be employed for many neuro-symbolic tasks.
In the future we will add code and instructions here for running Embed2Sym on your own tasks.  

## Author

Embed2Sym was developed by Yaniv Aspis.<br /> 
Email: yaniv.aspis17@imperial.ac.uk
