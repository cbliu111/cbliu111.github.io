---
title: "Coarse-graining, renormalization, multiscale modeling, sparsity, thermodynamics"
date: 2025-10-01
categories:
  - blog
tags:
  - theory
  - emergence
math: true
---

## The world is simple at every level

Many interesting observations:
- thermodynamics
- mean-field methods
- renormalization
- effectiveness of tensor network representation
- coarse-graining dynamics in stochastic processes
- emergence of simple dynamical behavior from complicate systems
- validation of variational principles
- divide and conquer for NP-hard problems
- compression is intelligence
- manifold assumption for real-world data
- neural networks are sparse learners: activated neurons are sparse, self-attention is sparse, low rank property of Hessian matrix

It seems **a reduced solution is already satisfactory enough**. 

## Information loss from low-level to high-level representation of world

I feel there is a multi-level representation of the world, the higher level representations are more vague, while the lower level representations are much accurate, containing more information. 

From the lower level to higher level, information are gradually lost. But the lost information seem to do not influence with the prediction of macroscopic behaviors in the higher level. Those lost information are corresponding to the fast fluctuated motions, which are less relevant to the behavior in the higher level representations. 

For thermodynamics, we may just discard information in the low level, and gradually advanced to the high level. From microscope to macroscope, there may have many intermediate levels, instead of just a vague definition of microscope level and a macroscope level. The introduction of ensemble, or probability can be helpful in identifying the less relevant information in the system. 

For mean-field, the averaging process discards fluctuated motions, and also the information related to these motions, or distributions, or interactions, or something else. 

For machine learning, the distribution of real-world data may not be continuous at all, but the machine learning actually discards the details of the sharp and ruggedness of the real-world distribution, and approximate the distribution with a smoothed distribution function. One example is the learning of quantum system, using POVM, each coordinate of the classical analog corresponds to a different dimension in the Hilbert space. Therefore, the corresponding distribution in the classical analog of the quantum states are not smooth at all. However, the machine learning methods can also approximate it with a smooth distribution function. 

Coarse-graining dynamics of stochastic dynamical system: using mean-field randomization, we can obtain a coarse-grained dynamics through time-averaging of the reaction rates. The macroscope behavior of the dynamical system is still largely preserved. 

Manifold assumption of real-world dataset: the world is really run on a low dimensional manifold when considering some level of coarse-graining. 

Multiscale modeling and emergence of macroscope dynamics: it is surprising to find out the dynamics of the macroscope world still respects almost the same level of complexity in the microscope world, as contrast to the high complexity of these macroscope system. Although the low level dynamics is simple, but there is no guarantee that the composition of many low level dynamics is also simple. The **emergence of simplicity from complexity** indicates some kind of invariant picture in the design of world. 

## Clustering of freedom

Is **timescale separation** leading to the emergency of macroscopic dynamics? Not completely, same kind of timescale separation can produce different amount of information loss. 

How about **constant information loss**? 

The macroscopic dynamics is highly dependent on the choice of abstract representations of human. e.g. considering earth as a mass point is a higher level of abstraction than considering earth as a rigid body, rotation is not considered in mass point, but will be considered in rigid body. The description of mass point then will be much simpler than the rigid body. This emergence of simplicity is originated from the change of abstraction, or due to **elimination of freedom**. 

Elimination of some freedom will result in large amount of information loss. e.g. eliminating the spatial coordinate of a particle can provide significant information loss in Szilard engine, but eliminating the freedom of electrons will have nearly no influence on the macroscopic information. But the freedom of the particle is not the freedom of the electrons or the nuclear. The spatial coordinate of a particle can be viewed as composite freedom of all the electrons and the nuclear. This freedom is emerged from the collection of many low level freedoms. Therefore, the **clustering of freedoms projects the low level dynamics to the high level dynamics**.  

**Clustering freedom** should be governed by **minimum information loss** principle. 

Neural networks are sparse learners:
> LESS: selecting influential data for targeted instruction tunning, https://arxiv.org/abs/2402.04333

**information-constrained automatic model reduction**: unification of model reduction methods. 

low-dimensional representation of the learned representation of state distribution, **embedding the state space into a manifold** to reveal new patterns, phases and collective behaviors. 


## Understanding thermodynamics 

With clustering freedom, we may view statistical thermodynamics as an inference under constraints of physical laws and minimum information loss. 

### Gibbs: finite resolution of the measuring apparatus

For a deterministic trajectory, we usually assume the system state is a mathematical point in the phase space. But in practice, this point should be replaced with a small blob which encloses the state point of the system determined by the dynamical law. 

Assume the system is initially within a region of phase space $\Delta$, with probability mass constrained by a diameter $\delta$. Here $\delta$ measures the resolution limit of arbitrary measuring apparatus. 

Let the system evolve to time $t$, based on the dynamical mechanism, $\Delta(t)$ will spread out to a fractal structure will more and more finer detailed structure, but its volume will be kept according to Liouville theorem. Define the ''length'' $L$ as the minimum possible line to link all the state points started from $\Delta$, then $L(t) > L$.

Since there is no way to distinguish two point within $\delta$ distance, the fractal structure with finer detailed structure means the volume of $\Delta(t)$ becomes larger as $\Delta*(t) = L(t) * \delta > \Delta(t) = L * \delta$. 

And Boltzmann entropy is the logarithm of $\Delta*(t)$.

**criticism**: although we can not measure, but that does not limit our ability to predict in arbitrary accuracy, so information is not lost in our theory but in the verification of the theory result? What is the physical nature of this limited measure resolution? Quantum physics?

### Another: equilibrium is a macroscopic state

We may prove that a subset of the phase space will then contain a fixed ratio of this fractal structure after a sufficient long time that the state points of the system is mixed well with the rest of the phase space. Thus producing the fixed view of the system from the macrostate perspective.  

This fixation of macrostate is a result of the self-similarity of the fractal structure, for the increment of finer details does not change the value of the macrostate as an average of a subset of the phase space. 

Also, for system starting from two different initial distribution, with time passing, they will be mixed arbitrarily well, thus making the value of the macrostate the same for these two different initial conditions, since the subset for calculating this macrostate is the same one. 

In this way, we have proved the existence of a fixed equilibrium macrostate, and the convergence of any initial distribution to this equilibrium macrostate. 

Since any initial distribution is converged to this equilibrium macrostate, it must be a distribution that lost every information about the initial distribution. Therefore it is the distribution that maximized the entropy. And this equilibrium macrostate is served as the time limiting distribution of any other initial distribution. 

So information is not lost, we can still predict with arbitrary accuracy, but further information does not change the value of the macrostate. So that part of information can be ignored when anticipating the future behavior of the macrostates.  

**criticism**: how to explain the arrow of time?

### Time reversibility as a statistical result of large freedom

We usually associate time irreversibility with entropy, and view entropy as an objective physical quantity such that increase of entropy is associated with the irreversible evolution of the system state. While on the other hand the microscopic dynamical law follows the time reversible manner. So leads to the paradox of macro-irreversible and micro-reversible of the same system evolution process. 

I think we may view the macroscopic evolution of the system as reversible as well, it is just very unlikely due to the large freedom of the system. For small freedom system, we may expect the system to evolves to reduce the entropy. Thus the irreversibility is just a matter of statistical effect. 

Entropy just measure the increment of uncertainty of the system. Some may argue that we can always chose a system by applying time reverse action on the current state and then decrease the entropy. True, but that requires to obtain a large amount of information of the position and momenta of every particle at that moment. 

Thus time arrow simply describe our increment of uncertainty. If only looking at an equilibrium state, our uncertainty will not increase, then we will loss the direction of time. 

The constrains include the law of dynamics, the observed information and the environmental setting, e.g. like connected with a reservoir. 

The Boltzmann entropy is defined as

$S_B = k_B \ln W$

where $W$ is the amount of weights that assigned to all the microstates.

So the weak point about Boltzmann distribution is how to assign the weights to each microstate. 

The most fundamental nature of the Hamiltonian system and the quantum system is the ''diffusive'' property, which means the dynamical system tends to expand its phase space when evolving along with time. That is, the possible microstates in the future is more than the number in the past. That is the reason why Boltzmann entropy can explain both the classical and quantum system follow the maximum entropy principle. 

However, for a generalized dynamical system, the law of mechanisms does not generally ensure ''diffusive''. e.g. the 3X+1 problem, where almost all trajectories shrink to a 4-2-1 circle. Or the Game of Life problem, the evolution trajectories is actually dependent on the initial state, either be stable, or faded.

To extend statistical thermodynamic into general dynamical system, we see the nature of statistical thermodynamics is merely an inference of the distribution of system states given the constraints. 

e.g. microcanonical system. E, V, N is fixed, and we know the dynamical law is diffusive, such that the future distribution tends to be more flattened. Thus the real distribution of the future is the distribution that maximize the entropy. Since there are no constraints on the phase space, so the distribution is actually a uniform distribution within the fixed energy hypersurface. 
This is actually the true objective true distribution, and we learned this distribution through the principle of maximizing the entropy as a way to approach the real distribution, which is equivalent to maximize the relative entropy between the real distribution and our estimate. 

So we attribute entropy increase as a property of the dynamical law, which is not ensured in a generalized dynamical law. 

**One restriction of the dynamical system may be the concave or convex of the entropy.**






