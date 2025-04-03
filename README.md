# Cahn-Hilliard System Equations Simulation to Find Exact Solution by PINN(Prerequisite Experiments for On-Going Paper Project)



This folder is code for the paper simulating Cahn-Hilliard equation by using PINN. Governing equation of Cahn-Hilliard equation is represented in the paper [1][WZJ20](See below detail for this paper) in the reference below.

You can download the paper via: [[arXiv]](https://arxiv.org/abs/2007.04542).


[comment]: # ([[ResearchGate]])

## Problem Summary

This project presents methods to improve the performance of Physics Informed Neural Networks (PINNs). Phase field model of Cahn-Hilliard type equations is the one of the most popular methods to describe interfacial dynamic problems. To design this euqation system,
the deep neural network is designed to perform as an automatic numerical solver as physics informed neural network (PINN).

![](assets/example.jpg)

## Main strategies in the paper

- **Adaptive sampling** : By making two groups of collocation points for training, this decrease the loss in very efficient way. For the first group, sample the points of training on the domain uniformly. And for the second group, Latin hypercube sampling is used for picking up the training points and then we choose only highest loss points to train by ranking the loss values for each points.
- **Time adaptive sampling 1** : For sampling on time t domain, we use starting time 0 and the extending end-times in {.1, .2, .3, ..., .9, 1} for each 30000 iterations of neural network training.
- **Time adaptive sampling 2** : By spliting all the domain with unit .25, we make time domain [0, .25], [.25, .5],[.5, .75], [.75, 1.0] separately for neural network training. For doing this, from second net training, we use the final training expected outputs velocity u of the previous net training as the initial condition.
- *Minibatch* : By using the mini-batch structure of neural network, we can decrease the whole iterations(epochs in the paper) efficiently.
  
## Folders in this repository

- 'AC_1D_gamma2_4' : Reconstruction of the neural network in the Fig3.5 in [1].
- 'devel_AC_gamma2_4_time_adap1': This is advanced experiments based on adaptive sampling and time-adaptive1 methods on neural network training. The experiments use the mini-batch and new sampling method to achieve successful results from the failure on the previous experiments in Fig3.6 in [1].

```math
\begin{array}{c}
    u_t - 0.0001 u_{xx} +4 u^3 -4 u = 0,\
    u(0,x) = x^2 sin(2 \pi x),\
    u(t, -1) = u(t,1),\
    u_x(t, -1) = u_x(t,1) .
\end{array}
```



** Challenges**:

- Stiffness due to small interfacial width requires an unconditionally stable numerical scheme.
- Nonlinearity of the system complicates proving the unique solvability of the numerical scheme.


## Requirement

- Python 3.6
- PyTorch 2.3.0
- NumPy 1.22.4
- ‎Matplotlib 3.5.3

## Preparation

### Clone

```bash
git clone https://github.com/woooojng/PINN_Cahn-Hilliard_system.git
```

[comment]: # (%### Create an anaconda environment [Optional]:)


[comment]: # (### Download the pretrained embeddings:)


## Usage

### Train the model at clonned directory in terminal:

```bash
python3 train.py
```

### Show help message and exit:

```bash
python3 train.py -h
```

## File Specifications

- **Building_net.py**: Neural Network Architecture for layer setting and forward step with x, y, t input/ velocity u, pressure P, density rho viscoscity mu output variables.
- **Conditions.py**: For the equations in [1], the equations on left/right wall and top/bottom outer boundary of domain with MSE function are defined. Also, for the Cahn-Hillard PDE equations in [1], MSE loss function associated with governing equations is defined.
- **Train.py**: Neural network training function starting from Initial condition train running and then running for total loss summing with all loss functions.


## Reference

[comment]: # (If this work is helpful, please cite as:)

<a id="1">[1]</a> 
Wight, C. L. & Zhao, J. Solving Allen–Cahn and
Cahn–Hilliard equations using the adaptive physics
informed neural networks. Preprint at arXiv
https://arXiv.org/abs/2007.04542 (2020).



[comment]: # (## Acknowledgments)

[comment]: # (This work is supported partly by the National Natural Science Foundation)

## Contact

wki1 [AT] iu [DOT] edu

[comment]: # (## License)

[comment]: # (MIT)
