# Cahn-Hilliard_equations

# Physics Informed Neural Network(PINN) for Cahn-Hilliard-Navier-Stokes equation


This folder is code for the paper simulating Cahn-Hilliard equation by using PINN. Governing equation of Cahn-Hilliard equation is represented in the paper [1][WZJ20](See below detail for this paper) in the reference below.

You can download the paper via: [[arXiv]](https://arxiv.org/abs/2007.04542).


[comment]: # ([[ResearchGate]])

## Problem Summary

This project presents methods to improve the performance of Physics Informed Neural Networks (PINNs). Phase field model of Cahn-Hilliard type equations is the one of the most popular methods to describe interfacial dynamic problems. To design this euqation system,
the deep neural network is designed to perform as an automatic numerical solver as physics informed neural network (PINN).

![](assets/example.jpg)

**Main strategies in the paper**():

- The authors develop a second-order in time numerical scheme using convex-splitting for the Cahn-Hilliard equation and pressure-projection for the Navier-Stokes equation.
- This scheme is unconditionally stable and uniquely solvable at each time step, ensuring robustness in simulations.
- The weak coupling in the scheme allows for efficient computation using a Picard iteration method.
- This work is significant as it provides a stable and accurate method for simulating the CHNS model, which is crucial for understanding the dynamics of fluid interfaces in various applications.
  
## Editing

The Physics-Informed Neural Network (PINN) merges neural networks (NN) with partial differential equations (PDEs), enabling direct solutions to intricate physical problems without strict dependence on labeled data. This cutting-edge approach synthesizes PDE principles with NN architecture to accurately predict system behavior, proving invaluable across diverse scientific and engineering domains.

This project introduces strategies to enhance PINN approximating capabilities. Key aspects of the paper include:

- The Cahn-Hilliard equation describes the evolution of the phase field variable $\phi$, representing the fluid interface.
- The Navier-Stokes equation governs the fluid flow, introducing complexity due to the coupling between velocity, pressure, and the phase field.

This project introduces strategies to enhance PINN approximating capabilities. Specifically, the study focuses on applying these enhanced PINNs to solve complex problems related to rising bubble systems with diffuse boundaries  as time goes by, employing a time-adaptive approach in conjunction with the level set method. By simulating using the PINN framework and comparing outcomes with existing results, the research aims to assess qualitative patterns and identify potential novel insights. Furthermore, utilizing existing data to validate accuracy and refine the network through the PINN procedure may reveal convergence among different methods and shed light on the true behavior of the system. Additionally, exploring the Deep Ritz method, which shares similarities with PINNs, could provide deeper insights into the underlying energy minimization associated with the problem when compared against PINN outcomes.

In this notebook, we are going to combine different two networks to apply the adaptive time marching strategy in the paper [1]. To describe in detail for this strategy, by discretizing time domain evenly, one network simulate PDE system on a time period and the next similar but different network simulate successively from the ending point of the former network. Therefore, as the name of itself, this strategy is applying simulation adaptively by marching on discretized time domain.

On our simulation, the first network on time $\[0, .1\]$ is made with the loss function added up for the loss terms coming from initial configuration equations, boundary condition equations and the following NS PDE equations.(See [1])

```math
\begin{array}{c}
    \rho (x) \left( \frac{\partial u}{\partial t} + u \cdot \nabla u \right) = - \nabla p +  \eta({\phi }) \Delta u + \rho (x) g,\
    \nabla \cdot u = 0,\
    [u]|_\Gamma = 0,\
    [pI + \eta (\nabla u + (\nabla u)^T)]|_\Gamma \cdot n = \sigma_{DA} \kappa n .
\end{array}
```



**Challenges**:

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
git clone https://github.com/hiyouga/RepWalk.git](https://github.com/woooojng/Bubble_PINN.git
```

[comment]: # (%### Create an anaconda environment [Optional]:)


[comment]: # (### Download the pretrained embeddings:)


## Usage

### Train the model at clonned `Sharp_interface_Bubble_PINN_ver1` directory in terminal:

```bash
python3 train.py
```

### Show help message and exit:

```bash
python3 train.py -h
```

## File Specifications ; Edit

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
