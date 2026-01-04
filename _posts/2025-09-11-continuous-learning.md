---
title: "Brain-inspired continuous learning?"
date: 2025-09-11
categories:
  - blog
tags:
  - algorithm
  - continuous learning
math: true
---

### The purpose: understand why DNN can learn, and how to continue-learning

Human learns whenever they are interacting with the environment. However, machine learns only when training. It is crucial to teach the machine to learn in a continual way, adapting model parameters to incoming streams of data. However, training of deep neural networks (DNNs) suffers from catastrophic forgetting: when learning new set of parameters without revising the old samples, the model tends to forget the learned knowledge. Moreover, the neural network's weights will not be changed once training or fine-tuning is done. If neural network does not have the ability to adapt to new dataset or new distribution of data, the network will be abandoned. With the development of knowledge, or the invention of new language, the model is degraded, making the model vulnerable. On the contrary, biological neural network has the ability to continuous learning all the time, and the model can evolve to adapt to new inputs. Learning new knowledge will not affect old knowledge, but will modify or update old knowledge. The adapting to new knowledge also empower biological neural network the strong generalization ability. Learning more knowledge will enhance the performance on all other tasks and inputs. 

Training of MET suffers from catastrophic forgetting: when learning new set of parameters without revising the old samples, the model tends to forget the learned knowledge. However, the neural network's weights will not be changed once they are trained. If neural network does not have the ability to adapt to new dataset or new distribution of data, the network will be abandoned. With the development of knowledge, or the invention of new language, the model is degraded, making the model vulnerable. 

On the other hand, biological neural network has the ability to continuous learning all the time, and the model can evolve to adapt to new inputs. Learning new knowledge will not affect old knowledge, but will modify or update old knowledge. The adapting to new knowledge also empower biological neural network the strong generalization ability. Learning more knowledge will enhance the performance on all other tasks and inputs. 

Both bias and variance can decrease as the number of parameters grows. 
Deep neural networks is powerful than originally think \cite{nealModernTakeBiasVariance2019}. 

For DNNs to generalize well to out-of-distribution test data, many studies focused on the wide/flat local minima. This can be intuitively understood from the viewpoint of the geometry of the loss landscape. 

![flat local minima](figures/flat_minima.png). 

The loss landscape of a DNN is the non-convex high-dimensional surface formed by the loss function in the parameter space of the neural network. Non-convex refers to the presence of multiple local minima, saddle points, and flat regions. 

Outline of the work:
1. Training with SGD include noise and also largely follow a deterministic path (low-dimensional subspace)
2. Modeling of the training with stochastic dynamical system on a low-dimensional subspace
3. A stochastic dynamical system can be understood with landscape and flux framework. The landscape is well-connected, funneled and smooth if using correct network architecture and regularizations. Source of flux comes from the random sampling of items from dataset. From gradient-driven to flux-driven induces a phase transition. Over-parameterized network is a sloppy model, so can make contact with protein folding-evolution correspondence. 
4. Loss landscape has unique feature: the training does not converge to a stick point or conventional basins but to a well-connected low-loss subspace with complex geometry, we term it optima manifold. 
5. Why optima manifold? Permutation and shift symmetry of the parameter space. 
6. DNN is actually conceptual learner. Combination of interactive concepts produce combinatorial many information flows, which explains the power of expressivity of DNN. Also, permutation of parameters or shift of weights preserve concepts, thus give the symmetries of the parameter space. Generalization and continual learning are the two-side of the same coin: generalization use the memorized concepts to predict unseen data; continual learning aims to preserve the memorized concepts while learning new concepts for new tasks. Both of them demand sparse yet powerful concepts. There must have a universal relation between sparse concepts and presence of optima manifold. 

Inference process can be viewed as a discrete-time stochastic dynamical system, and learning (weight updating) can be viewed as a variational optimization process. 
Inference-learning correspondence as similar to dynamics-evolution correspondence of proteins. 
Computation network (made of computation graphs of all layers of a network, maybe very similar to tensor networks) and how data or information flow from the entrance of the network to the exit of the network. Network with different architecture may have very different computational network, and also very different patterns of information flow. 
The most important thing may not be how you calculate it, but instead how many times you calculate it. 
There may be a correlation between the gradient of a parameter and 



Shallow neurons process data and produce sensory information. Sensory information is then integrated and combined to form concepts (or representations, patterns). Interaction of low-order concepts generate high-order concepts hierarchically in deep neurons. 
At the end of computation, the collection of high-order concepts produce an output $\mathbf{y} = \mathbf{F}(\mathbf{x}; \mathbf{\theta})$, which is a vector (in the sense of one-hot encoding), to be used to calculate the loss (energy) $\mathcal{L} = E(\mathbf{y}, \mathbf{y}_{gt})$ with $\mathbf{y}_{gt}$ the label value. 
We define the native prediction (analogy to the native state of protein folding) of the network's output $\mathbf{y}$ to be the output of data $\mathbf{x}$ when network parameter $\mathbf{\theta}$ is frozen. $\mathbf{y}$ is then a random variable that fluctuates with randomly sampled $\mathbf{x}$. 
On the other hand, since $\mathcal{L}$ is also a function of network parameter $\mathbf{\theta}$, updating of $\mathbf{\theta}$ through SGD can also induce stochastic dynamics of $\mathbf{y}$. 
Is there a correlation between the fluctuations induced by random data and the fluctuations induced by SGD? Just like the dynamics-evolution correspondence of protein folding. 
We can define the joint landscape of $\mathbf{x}$ and $\mathbf{\theta}$, which can be a super-funneled landscape. 
How about use protein evolution algorithm to train neural networks?
We can define **conceptual frustration**: conflict of concepts, two compensate concepts are simultaneously activated, or no valid concepts are activated in computing $\mathbf{y}$. 
Then, we can propose the **minimal frustration principle**: conceptual frustration is always minimized for a well-generalized neural network. 
How to minimize conceptual frustration? The secrete is sparse representation of concepts. When sensory information is acquired, only a few neurons are activated to compute $\mathbf{y}$, such that minimized concepts are used to achieve a low loss. When number of activated concepts are small, there is smaller chance that they conflict with each other. By using sparse concepts, the conceptual frustration is minimized. 
Sparse concept neural network hence generalizes well since prediction of new data will be less confused. 
This sparsity also makes the loss landscape low-rank. 
SGD uses noise to minimize conceptual frustration. 
Why sparse representation is enough to achieve low loss? Because the smooth manifold conjecture of natural dataset. Only a few independent representations are enough for characterizing the feature space of natural data. 
Moreover, sparsity can also be increased when the neural network gets larger. So larger neural network has more sparse concepts, and hence smaller conceptual frustration. This is why large network behaves better than small network. 
Will the network architecture also impacts the sparsity of concepts? Maybe transformers are more capable of making concepts isolated than other architecture networks. Since the attention mechanism, the attention map computes the pairwise attention (or pairwise conceptual interactions) across the whole data. All the potential pairwise interaction is captured by attention mechanism. On the contrary, convolutional computations only capture a small region of the data. So we can think of transformer capturing the **global conceptual interaction**, while CNN only capture the **local conceptual interaction**.  
No higher order interactions are available in transformer. This may be more like the Feynman graph, where more precise computation demands higher order interactions. The fully-connected network calculates all the higher order interactions. However, the computational complexity is high. Furthermore, the embedding of data into a high dimensional space makes the concepts even more sparse. And computation of conceptual interactions are made possible by computing the cosine similarities. 
Each embedding is a concept, by computing similarities of concepts transformer makes the gathering of high-order concepts more effective. 

Robustness and plasticity: robustness is generalization, plasticity is continual learning. 
Proteins are good at both sides, also brains. 

Why minimal frustration? Because we design the network to be generalizable, which selects minimal frustration indirectly. So can we automatically design network with minimized frustration?

Sparse concepts may also have lower loss than dense concepts because they will have limited frustration in the low loss region. 

## Results

### Training as a stochastic dynamical system on a low-dimensional subspace

In deep neural network model, the model is parameterized by millions or trillions of numerical parameters $\omega_i (i = 1, 2, \ldots, N)$. These parameters consisted a non-convex loss function (energy) landscape. The learning (training) process of neural network is made possible by the stochastic gradient descent (SGD). 

And the gradient is calculated from some loss function, which is generally a relative-entropy between distributions (as indicated in previous section)
$$
\mathcal{L}(\mathbf{\omega}) = \int d\left[ p(\mathbf{x}), q_{\mathbf{\omega}}(\mathbf{x}) \right] d\mathbf{x} 
$$
where $d\left[ p(\mathbf{x}), q_{\mathbf{\omega}}(\mathbf{x}) \right]$ is a measure of distance between the ground truth probability distribution $p(\mathbf{x})$ and the modeled distribution $q_{\mathbf{\omega}}(\mathbf{x})$. If regressive task is involved, both $p(\mathbf{x}) = \delta(\mathbf{x})$ and $q_{\mathbf{\omega}}(\mathbf{x}) = \delta(\mathbf{y})$ are $\delta$ distributions, and the meansure $d$ characterizes the geometric distance between $\mathbf{x}$ and $\mathbf{y}$. 

The loss function is an integral functional valid given infinite available data samples from the data distribution. This integral is approximated by the dataset which is a collection of data samples from the data distribution. The collection of dataset is as much as approximating the integral computation with a Monte Carlo method. Given finite size of the dataset (samples from the real world) or utilizing minibatch training, the loss function is approximated by sampling 
$$
\mathcal{L}(\mathbf{\omega}) = \mathbb{E}_{\mathbf{x} \sim p} d\left[ p(\mathbf{x}), q_{\mathbf{\omega}}(\mathbf{x}) \right] 
$$
And the step of weight for iteration $t$ is  
$$
\Delta \omega_i (t) = -\alpha(t) \frac{\partial \mathcal{L}^{\mathbf{x}(t)}(\mathbf{\omega})}{\partial \omega_i}
$$
where $\omega_i (t)$, $\mathbf{x}(t)$, and $\alpha(t)$ are the weight, samples in minibatch and the learning rate for iteration $t$, respectively. The noise of gradients by using minibatch originated from the variance $\delta \mathcal{L} = \mathcal{L}^{\mathbf{x}} - \mathcal{L}$. 

By taking the continuous-time approximation (viewing iteration as continuous time), and keeping the first-order derivative term, a stochastic partial differential equation is obtained for SGD,
$$
\frac{\partial \mathbf{\omega}}{\partial t} = -\alpha(t) \frac{\partial \mathcal{L}}{\partial \mathbf{\omega}} + \mathbf{\eta}(\mathbf{\omega})
$$
by setting the iteration time for one minibatch $\Delta t = 1$. The continuous-time approximation is valid as long as the minibatch size is much smaller than the size of the dataset, i.e. training time of one epoch $T \gg \Delta t$. The noise term is $\mathbf{\eta} \equiv -\alpha(t) \nabla_{\mathbf{\omega}} \delta \mathcal{L}^{\mathbf{x}}(\mathbf{\omega})$, with zero mean $\mathbb{E}_{\mathbf{x} \sim p} \mathbf{\eta} = 0$, and time correlation $C_{ij} (\mathbf{\omega}) \equiv \mathbb{E}_{\mathbf{x} \sim p} \eta_i \eta_j = \alpha^2 \mathbb{E}_{\mathbf{x} \sim p} \frac{\partial \delta \mathcal{L}^{\mathbf{x}}}{\partial \omega_i} \frac{\partial \delta \mathcal{L}^{\mathbf{x}}}{\partial \omega_j}$. The noise depends explicitly on weight, indicating the noise is a kind of state-dependent multiplicative noise. 


#### Escape of saddle points

Saddle points also exist extensively in loss landscape, every critical point that is not a global minimum is a saddle point, and there exist bad saddle points \cite{kawaguchiDeepLearningPoor2016}. Proliferation of saddle points, instead of local minima, is crucial for non-convex optimization in high dimensional parameter spaces \cite{dauphinIdentifyingAttackingSaddle2014}. But saddle points are generally not concerns of training DNN. 

#### Noise of SGD

The escape of saddle points are largely attributed to the stochastic gradient descent (SGD) algorithm, where noise plays a crucial role in exploration of the parameter space for well-generalized model. 
Also, adding gradient noise during the training can avoid overfitting and also results in lower training loss \cite{neelakantanAddingGradientNoise2015}. 
Small batch size improves generalization performance. Small batch size increases noise, reduced computation accuracy of loss function, and has resulted in small weight-updating step size compared with large batch size (which can be modulated by increase learning rate) \cite{mastersRevisitingSmallBatch2018}. 
Gradient noise scale predicts the largest useful batch size across many domains. Noise scale increases as the loss decreases over a training run and depends on the model size primarily through improved model performance \cite{mccandlishEmpiricalModelLargeBatch2018}. 
SGD remains bounded and converges with probability 1 under a very broad range of step-size schedules. SGD avoids strict saddle points/manifolds with probability 1 for the entire spectrum of step-size policies considered. A cool-down phase with vanishing step-size leads to faster convergence \cite{mertikopoulosAlmostSureConvergence2020}.
DNN can converge for both sub-Gaussian and centered noise and integrable heavy-tailed noise \cite{scamanRobustnessAnalysisNonConvex2020}.
Confirmation of noise effects on enhancing generalization \cite{smithGeneralizationBenefitNoise2020}. Small batch size with large learning rate or large batch size with small learning rate may have the same effect: promote exploration of the parameter space after entering the optima manifold.
Symmetric noise to reach flat minima \cite{sungSSGDSymmetricalStochastic2020}. Does symmetric noise explore better?
SGD generalizes well than ADAM. Heavy-tails of gradient noise in these algorithms, and SGD has smaller escaping time than ADAM. Also, ADAM has anisotropic structure in gradient noise and lighter gradient noise trails. SGD is more locally unstable than ADAM at sharp minima defined as the minima whose local basins have small Radon measure, and can better escape from them to flatter ones with larger Radon measure \cite{zhouTheoreticallyUnderstandingWhy2020}. Evidence for heavy-tailed gradient noise assumption, using simply state-dependent Gaussian noise may be over-simplified. 
SGD favors flat minima exponentially more than sharp minima, while Gradient Descent (GD) with injected white noise favors flat minima only polynomially more than sharp minima. Either a small learning rate or large-batch training requires exponentially many iterations to escape from minima in terms of the ratio of the batch size and learning rate. Thus, large-batch training cannot search flat minima efficiently in a realistic computational time \cite{xieDiffusionTheoryDeep2021}.
Escape efficiency of SGD with Gaussian noise, by introducing the Large Deviation Theory for dynamical systems. Based on the theory, we prove that the fast escape form sharp minima, named exponential escape, occurs in a non-stationary setting, and that it holds not only for continuous SGD but also for discrete SGD \cite{ibayashiExponentialEscapeEfficiency2022}. SGD escapes the bad minima even before complete exploration of the local area. 
SGD escape sharp local minima under the presence of heavy-tailed gradient noise. As the learning rate decreases the dynamics of the heavy-tailed truncated SGD closely resemble those of a continuous-time Markov chain that never visits any sharp minima \cite{wangEliminatingSharpMinima2022}. The source of noise comes from the random sampling of items from dataset.
Evolutionary algorithms (EAs) inspired by the Gillespie-Orr Mutational Landscapes model for natural evolution is formally equivalent to SGD \cite{kucharavyEvolutionaryAlgorithmsLight2023}.
Training in the parameter space of DNN behaves like a stochastic dynamical behavior where distribution of PCA eigenvalues and the projected trajectories resemble a random walk with drift \cite{antogniniPCAHighDimensional2018}. 
Convergence of SGD to local minimizer is assured by using stable manifold theorem from dynamical system theory with random initialization, which makes DNN easily optimized using local updates \cite{leeGradientDescentOnly2016}\cite{soudryNoBadLocal2016}. 
From this viewpoint, SGD minimizes an average potential over the posterior distribution of weights along with an entropic regularization term. The behavior near critical points are not like Brownian, but resemble closed loops with deterministic components. This "out-of-equilibrium" behavior is a consequence of highly non-isotropic gradient noise is SGD. The rank of covariance matrix of mini-batch gradients is as small as 1% of its dimension. The training trajectories converge to a non-equilibrium steady-state distribution of local minima, also the support of this distribution has very low rank, indicating a low dimensional manifold topology structure \cite{chaudhariStochasticGradientDescent2018}. 
Statistical interaction between the individual entries of the network asymptotically vanishes, resemble a chaotic behavior of training \cite{bortoliQuantitativePropagationChaos2020}.
The training trajectories are strongly restricted within a low-rank space, i.e. loss surface has low intrinsic dimensionality \cite{liMeasuringIntrinsicDimension2018}. 
This can also be understood from the Hessian eigenvalue density \cite{ghorbaniInvestigationNeuralNet2019}, or visualize learning dynamics by calculating similarities between units from all layers \cite{giganteVisualizingPHATENeural2019}. 
Noise of SGD is not only state-dependent, but has a power-law dynamics. Stationary distribution is heavy-tailed. Mean escaping time is polynomial order of the barrier height of the basin, much faster than exponential order of state-dependent noise. So the learning tends to stop at flat minima with lower generalization error \cite{mengDynamicStochasticGradient2020}. 
SGD is a landscape-dependent annealing algorithm, effective temperature is high when the system is at sharp minima and decreases with the landscape flatness. This is the inverse variance-flatness relation. The reason for this inverse relationship is that the SGD noise strength and its correlation time depend inversely on the landscape flatness \cite{fengInverseVarianceFlatness2021}. Neural network may treat each data as a separate dataset, so the ordinary training may also be a sequential learning process. If treating the set of tasks as a whole dataset, then sequential learning is a learning with partial available data, the optima manifold is not complete. The inverse variance-flatness relation may only be held for some weights, but for other weights, their variance are free (not correlated with the loss function). In other word, this paper only revealed the partial story: the curvature or landscape side. Another side is the flux which is dependent on the parameters that are not strictly controlled by the inverse variance flatness relation. We should divide the set of weights into curvature weights and flux weights.
Only a few eigenvalues of Hession of the loss functions on the loss landscape are large. Indicating the training process only happens in low-dimensional subspace corresponding to the top eigenvalues of the Hessian \cite{fjellstromDeepLearningStochastic2022}. Will the number of large eigenvalues has an abrupt change during the training? Indicating the phase transition of the training.


#### Smooth manifold conjecture of dataset

Mixup training generalize well than empirical risk minimization method. 
Possibly due to increased continuity of dataset \cite{liangUnderstandingMixupTraining2018}.
Generalised Gauss-Newton matrix approximation of the Hessian. Maximal learning rates as a function of batch size. Hessian spectrum trained on Gaussian mixture data is similar to hessian spectrum trained on natural images \cite{granziolLearningRatesFunction2022}. Efficient way for computing Hessian. Natural images are having a regularized distribution, distributed on a low-rank manifold, which is the manifold conjecture.
Entropic regularization: loss function considers the contribution of adversarial samples that are drawn from a specially designed distribution in the data space that assigns high probability to points with high loss and in the immediate neighborhood of training samples \cite{jagatapAdversariallyRobustLearning2022}.
Class-imbalance dataset: network weights converges to a saddle point in the loss landscapes of minority classes \cite{rangwaniEscapingSaddlePoints2022}. Re-weighting of the dataset may be automatically done with curiosity mechanism.
Rashomon set is the set of models that perform approximately equally well on a given dataset. Rashomon ratio is the fraction of all models in a given hypothesis space that are in the Rashomon set. Noisier datasets lead to larger Rashomon ratios through the way that practitioners train models \cite{semenovaPathSimplerModels2023}.
**A continued distribution is enough to describe real-world data distributions.**

In MET, for time and rate parameters, only a small region around the sample of the training dataset has high precision.
In other words, the generalization ability decreases with the distance from the sample of the dataset, so the model has a limited generalization ability.
Some questions: 
- What is the relation between the generalization ability and the distance from the sample?
- Is generalization ability nothing but just smooth fitting of the dataset sample?
- Since MET use a learnable projection to embed the prompts, is the smooth fitting only limited to linear projections?
- Generalization is considered to be related to the flatten-minimum of the loss landscape, what is the relation between distance, prediction precision and the flatness of loss landscape?
- Maybe a more flat minimum can have larger distance under the same precision? 
- We can define intraplotation and outraplotation separately for the datasets, consider both states and prompts as two different datasets. And generalization can be discussed for these two datasets.
- We can also consider a quantum system, using POVM the classical corresponding distribution is very ruggedness, will the generalization ability be limited?

**Data continuity: adversarial attack destroyed the data continuity, so lead to failure of inference.**


### Landscape and flux perspective of DNN and phase transition in training

Fokker-Planck approach for SGD. Stationary state of the system in the long-time limit, exhibiting persistent currents in the space of network parameters. As in its physical analogues, the current is associated with an entropy production rate for any given training trajectory. The stationary distribution of these rates obeys the integral and detailed fluctuation theorems. Surprisingly, the effective loss landscape and diffusion matrix that determine the shape of the stationary distribution vary depending on the simple choice of minibatching done with or without replacement \cite{adhikariMachineLearningOut2023}.
Decomposing the stochastic dynamics into longitude and transverse parts. Replacing loss function with stochastic potentials resolve the inverse relationship between variance and flatness \cite{xiongStochasticGradientDescent2023}. Very limited results, and no improve algorithm on the learning process. But flatness of loss function has the specific meaning that related to generalization of model prediction ability, replacing loss function with some potential does not make any sense.

Convergence to limit cycles indicating existence of flux:
> [CS18] Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles For Deep Networks

Noise is essentical in training very deep neural networks:
> [NVL+15] Adding Gradient Noise Improves Learning for Very Deep Networks

![landscape](figures/landscape.png)

#### Loss landscape can be well-connected and smooth if using correct regularizations

The loss landscape is well-connected, facilitating the convergence of model to the optima manifold within practical time \cite{czarneckiDeepNeuralNetwork2020}.
The smoothness of loss landscape can be further increased by using skip connections, i.e. deep neural networks with skip connections are more like Go-model for protein folding \cite{liVisualizingLossLandscape2018}. 
Large-batch SGD tends to converge to sharp minima, smooth out sharp minima by perturbing multiple copies of the DNN by noise injection and averages these copies \cite{wenSmoothOutSmoothingOut2018}. 
One significant difference between DNN training and real-world dynamical system is the step-size of DNN training is adjustable by tuning the learning rate. In fact, control the ratio of batch size to learning rate to generalize well \cite{heControlBatchSize2019}. 
Decreasing learning rate acts as landscape stretching \cite{orvietoContinuoustimeModelsStochastic2019}. 
Learning rate actually generate a coarse-grained dynamics from the original learning dynamics. Small learning rate is fine-grained dynamics. From another viewpoint, adjusting learning rate is like rescaling the loss landscape. 
Rescaling induced equivalent relations for flatness/sharpness induce a quotient manifold structure in the parameter space. A Hessian-based measure of flatness that is invariant to rescaling. Large-batch SGD minima are indeed sharper than small-batch SGD minima \cite{rangamaniScaleInvariantFlatness2019}.
Computational efficient local extragradient method as a way of smoothing for distributed large-batch training \cite{linExtrapolationLargebatchTraining2020}. Seems to be a way to modulate the learning rate. 
Monotonic linear interpolation property: linear interpolation between initial neural network parameters and converged parameters after training with SGD leads to a monotonic decrease in the training objective \cite{lucasAnalyzingMonotonicLinear2021}. First observed by Goodfellow et al. 2014. Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544, 2014. This can be understood from the high connectivity property of the loss landscape. It may also suggest that the loss landscape is a very smooth landscape which has a direct path linking the initial and the converged parameters.
Smaller batch size leads to higher scores in a shorter training time, and argue that this is due to better regularization of the gradients during training \cite{atrioSmallBatchSizes2022}.
Forcing the learned reward function to be local Lipschitz-continuous is a sine qua non condition for the method to perform well \cite{blondeLipschitznessAllYou2022}. Continuous condition may be very important for neural networks.
Define geometric complexity of loss landscape by using a discrete Dirichlet energy. Many previously proposed regularization methods are all acting to control geometric complexity \cite{dherinWhyNeuralNetworks2022}.
Gradient descent produce sharp local minima than SGD. SGD can be viewed as GD with stochastic regularization \cite{geipingStochasticTrainingNot2022}.
Poor conditioning, i.e. bad model or hyperparameter choices leads to training instability \cite{gilmerLOSSCURVATUREPERSPECTIVE2022}.

Keeping constant learning rate is better for transformer:
> [SLA+19] Measuring the Effects of Data Parallelism on Neural Network Training

Training transformer is intrinsic different from training convolutional neural network:
> [LLG+20] Understanding the Difficulty of Training Transformers

Different models have different optimal batch size:
> [SLA+19] Measuring the Effects of Data Parallelism on Neural Network Training

#### Similarity between loss landscape and protein folding landscape, consider inference-training and function-evolution correspondence, super-funneled landscape

The geometry of the loss landscape is much like the folding landscape of protein. As a matter of fact, there is a connection between spin-glass theory and neural network learning dynamics: the similarity between loss function of the neural networks and Hamiltonian of the spherical spin-glass models \cite{choromanskaOpenProblemLandscape2015}. 
Falling in to the optima manifold is done within limited time-steps. SDE of learning dynamics, fast equilibrium conjecture. Intrinsic learning rate \cite{liReconcilingModernDeep2020}.
For data rich training, the loss landscape is funneled and easy to optimize. For data poor training, many minima with similar loss values exists and are separated by low barriers. Different from the hierarchical landscapes of structural glass formers \cite{verpoortArchetypalLandscapesDeep2020}.
Compute the entropy of typical solutions using replica trick, wide flat minima arise as complex extensive structures, from the coalescence of minima. High robust solutions are exponentially rare to low robust solutions, and concentrate in particular regions \cite{baldassiUnveilingStructureWide2021}. Possibly, the structure of the optima manifold is also very complex.
Chaotic geometry of the loss function for glassy dynamics exhibited by the Newman-Moore model due to the presence of fracton excitations, which causes mode collapse in neural annealing protocols \cite{inackNeuralAnnealingVisualization2022}.
Analogy to protein interaction, if we can define affinity (loss), can we also define specificity (some new name)?

#### Phase transition during training induced by gradient-driven to flux-driven transition

**Initial training has small noise level due to the large loss, so curvature dominates the convergence process. When loss decreased to a certain level, noise level increased. To keep the product of learning rate and noise within a limited level, learning rate should be decreased, or batch size should be increased.**

First gradient (curvature) then flux (noise), so there will be a change of phase. 

Reveal the phase transition by using entropy production rate as the order parameter. 

Training of deep neural network is very much like the folding of protein, except for deep neural network the steady-state is not unique. Therefore, deep neural network can be viewed as a special spin-glass. The glass transition in spin-glass system corresponds to the gradient to flux transition in deep neural network. 

Exponentially many configurations in every band of loss landscape. 

Funneled landscape, entropy is width, loss is depth. 

There may also be a Levinthal paradox in training deep neural networks.

First phase, driven by gradient and flux. Second phase, mainly flux. Using **entropy production rate as the order parameter**. 
Generalization gap. Start local SGD at the second phase of training to generalize well \cite{linDonUseLarge2020}. Utilizing the nature of phase transition in learning dynamics. 
Poor conditioning: failure to coordinate the optimization trajectory to avoid regions of high curvature and into flatter regions \cite{gilmerLossCurvaturePerspective2021}. 
Best test accuracy is obtained when: the loss landscape is globally well-connected; ensembles of trained models are more similar to each other; and models converge to locally smooth regions. Phases of learning (and consequent **double descent behavior**) \cite{yangTaxonomizingLocalGlobal2021}. Landscape connectivity (network architecture), data continuity (quality of data) and existence of optima manifold (good loss function) are the key factors for good generalization.
Low-curvature minima generalize better, SGD discourages curvature. Explicit SGD steady-state distribution showing that SGD optimizes an effective potential related to but different from train loss, and that SGD noise mediates a trade-off between low-loss versus low-curvature regions of this effective potential \cite{bradleyShiftcurvatureSGDGeneralization2022}.
SGD changes from **superdiffusion** to **subdiffusion** when navigating through the loss landscape, such learning dynamics happen ubiquitously in various neural networks, including vision transformers. Superdiffusion: intermittent, big jumps, allows extensive explore the loss landscape. This behavior is due to the interaction of SGD and fractal-like regions of the loss landscape \cite{chenAnomalousDiffusionDynamics2022}.
Suboptimal local minima are common for wide neural nets \cite{dingSuboptimalLocalMinima2022}. Will there be a smooth transition from large loss area to small loss area or abrupt transition?
Changes in model architecture (and its associate inductive bias) cause visible changes in decision boundaries, while multiple runs with the same architecture yield results with strong similarities, especially in the case of wide architectures. Decision boundary reproducibility depends strongly on model width. Near the threshold of interpolation, neural network decision boundaries become fragmented into many small decision regions, and these regions are non-reproducible. Meanwhile, very narrows and very wide networks have high levels of reproducibility in their decision boundaries with relatively few decision regions. **Double descent phenomena**: double descent is predominantly driven by the “unnecessary” oscillations resulting from model instability, and not by the error bubbles around mislabeled points \cite{somepalliCanNeuralNets2022}.
Loss spike, training becomes stable when it finds a flat region. But at a smaller-loss-as-sharper region, the training becomes unstable and loss exponentially increases once it is too sharp. Loss spike may facilitate condensation, i.e. input weights evolve towards the same, and hence improves generalization \cite{zhangLossSpikeTraining2023}. The existence of loss spike may be due to the rugged landscape region. So although the loss landscape is highly connected and can be smooth for some directions, on some other directions the loss landscape is very rugged.
Curvature-based loss increment ∆L tends to be small for a relatively flat test-loss landscape. Sample-wise loss, batch-wise loss, dataset-wise loss. Post-Training of Feature Extractor that updates the feature extractor part of an already-trained deep model to search a flatter minimum \cite{satoPoFPostTrainingFeature2022}.

Convergence rate, $k$ is epochs: 
- gradient descent: $\mathcal{O}(1/k)$
- mini-batch gradient descent: $\mathcal{O}(\sqrt{H/bk}+L^2/k^2)$
- stochastic gradient descent: $\mathcal{O}(1/\sqrt{k})$

Two phase phenomenon:
> [SJD+18] An Empirical Model of Large-Batch Training
[SLA+19] Measuring the Effects of Data Parallelism on Neural Network Training

Noise dominanted and curvature dominated phase:
> [ML18] Revisiting small batch training for deep neural networks
[SED20] On the Generalization Benefit of Noise in Stochastic Gradient Descent

Convergence rate and steady-state risk (divide driven force of learning into two conjugate parts?):
> [ZLN+19] Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model

Critical batch size and optimizers:
> [ZXZ+23] CowClip: Reducing CTR Prediction Model Training Time from 12 hours to 10 minutes on 1 GPU
[ZJC+21] A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes

### Optima manifold: well-connected low-loss subspace with complex geometry

However, instead of having a single folded state, there are exponentially many local minima with low loss values. 
For a single neuron with the logistic function as the transfer function having $d$ weights, the number of local minima for $n$ training examples may be $\lfloor n/d \rfloor ^d$ \cite{auerExponentiallyManyLocal1995}. 
There is an obvious gap between these local minimums and the other parameter state. 
For large-size decoupled networks, the lowest critical values of the random loss function forms a band, and number of local minima outside this band diminishes exponentially with the size of the network \cite{choromanskaLossSurfacesMultilayer2015}. 
Bulk of critical points of a deep neural network lies within a narrow band. 
This is in agreement with spin glasses theory that proves the existence of such a band when the dimension of the domain tends to infinity \cite{sagunExplorationsHighDimensional2015}. 
Every differentiable local minimum has almost the same loss, independent of the nature of the dataset, which is called the nonexistence of poor local minima \cite{kawaguchiDeepLearningPoor2016}\cite{soudryNoBadLocal2016}. 
No barriers in paths between two local minima for deep neural network, there exists a **single connected manifold of low loss** \cite{draxlerEssentiallyNoBarriers2018}. 
In fact, the local minimums are connected by simple curves over which training and test accuracy are nearly constant \cite{garipovLossSurfacesMode2018}. 
No bad local valley: in the sense that from any point in parameter space there exists a continuous path on which the cross-entropy loss (not only square loss) is non-increasing and gets arbitrarily close to zero \cite{nguyenLossLandscapeClass2018}. 
These local minimums form a very small subspace after a short period of training. 
The subspace is spanned by a few top eigenvectors of the Hessian (**equal to the number of classes in the dataset**), and is mostly preserved over long periods of training \cite{gur-ariGradientDescentHappens2018}. 
At the bottom of the loss landscape, there are no basins, but large connected components with no centralized point. Small and large batch size converge to different basins of attraction, but these basins are connected through flat paths and so belong to the same component \cite{sagunEmpiricalAnalysisHessian2018}. 
Model the loss landscape as a set of high dimensional wedges that together form a large-scale, interconnected structure and towards which optimization is drawn. Existence of low loss subspaces connecting a set of solutions \cite{fortLargeScaleStructure2019}.
The optima manifold are not symmetric. At a local minimum there exist many asymmetric directions such that the loss increases abruptly along one side, and slowly along the opposite side -- we formally define such minima as asymmetric valleys \cite{heAsymmetricValleysSharp2019}. 
The geometry of the subspace formed by local minima (termed optima manifold) may be a result of permutation and shift symmetry of the parameter space. 
Permutation symmetry of neurons gives rise to multiple equivalent (exponentially many) global minima of the loss function. 
Parameter space symmetries produces combinatorial many critical points \cite{breaWeightspaceSymmetryDeep2019}. 
Mathematical explanation of the existence of optima manifold, where loss function is almost constant, and the paths connect local minima can be chosen to be piece-wise linear, with as few as two segments \cite{kuditipudiExplainingLandscapeConnectivity2020}. 
Mode-connecting simplicial complexes that form multi-dimensional manifolds of low loss, connecting many independently trained models \cite{bentonLossSurfaceSimplexes2021}. A strong support for the existence of an optima manifold structure in the loss landscape. 
Learning a line of optima, instead of a single point. The midpoint of the line optima outperforms the standard training \cite{wortsmanLearningNeuralNetwork2021}. This is an evidence to show the connectivity of optima to form an optima manifold, and also the overlap of task-specific optima leads to the flat minima.
Hierarchical structure of the loss landscape of NNs, i.e., loss landscape of an NN contains all critical points of all the narrower NNs \cite{zhangEmbeddingPrincipleHierarchical2021}. Evidence of existence of exponentially many mapping trajectories in deep neural network, and deep neural network is a hierarchical stack of shadow neural networks. 
Low-rank matrix recovery: flat minima exactly recover the ground truth under standard statistical assumptions \cite{dingFlatMinimaGeneralize2023}.
The classical statistical learning theory implies that fitting too many parameters leads to overfitting and poor performance. But DNNs are sparse learners, they achieve plausible generalization abilities with implicit regularization induced by stochastic gradient descent and local geometry. Local geometry forces SGD to stay close to a low dimensional subspace and that this induces another form of implicit regularization and results in tighter bounds on the generalization error for deep neural networks \cite{imaizumiGeneralizationBoundsDeep2023}. 
Linear mode connectivity: two neural networks, trained similarly on the same data, will maintain loss when interpolated in the weight space. Distinct clusters of models which are linearly connected on the test loss surface, but are disconnected from models outside the cluster \cite{junejaLinearConnectivityReveals2023}. Models are clustered based on their generalization strategies, like having the same way of thought and solving problems.
Task-based flat minima: Improving Multi-task Learning \cite{phanImprovingMultitaskLearning2023}.
A degenerate Hessian implies locally flat regions. The landscape may be flat beyond the notion of wide basins. Training stops at a point that has a small gradient. The norm of the gradient is not zero, therefore it does not, technically speaking, converge to a critical point. There are still negative eigenvalues even when they are small in magnitude. \cite{sagunEigenvaluesHessianDeep2017}. The converged state is not a critical point or a conventional basin, but a manifold with complex geometry and low loss value. 
Over-parametrization and redundancy that are able to create large connected components at the bottom of the landscape. Small and large batch gradient descent appear to converge to different basins of attraction but we show that they are in fact connected through their flat region and so belong to the same basin (the same optima manifold) \cite{sagunEmpiricalAnalysisHessian2018a}.
Not only flatness but also closeness influence generalization \cite{chenRethinkingModelEnsemble2023}

**There are a wide flat minima region in the weight parameter space, optimizer should explore the region aggressively**. 

#### Symmetries of DNN

The existence of optima manifold can be understood from the perspective of DNN symmetry. 

Two local minima found by gradient-based methods end up lying on the same basin of the loss landscape after a proper permutation of weights is applied to one of the models \cite{akashWassersteinBarycenterbasedModel2022}. Permutation symmetry produces exponentially many identical local minima, this can also be explained by the permutation symmetry of mapping trajectories. The question is, what is the unique property of flat local minima? Are they invariant under the weight permutations?
Given permutation invariance (permutive symmetry), SGD solutions will likely have no barrier in the linear interpolation between them \cite{entezariRolePermutationInvariance2022}. That is, the loss landscape is highly connected in every level (value of loss).
The NN parameterization is invariant under two symmetries: permutation of the neurons and a continuous family of transformations of the scale of weight and bias parameters \cite{sahsShallowUnivariateReLU2022}.
Maximal number of condensed orientations in the initial training stage is twice the multiplicity of the activation function, where “multiplicity” indicates the multiple roots of activation function at origin \cite{zhouUnderstandingCondensationNeural2022}.
Scale invariant metric of flat minima, enjoys both direct theoretical connections and better empirical correlation to generalization error \cite{tsuzukuNormalizedFlatMinima2020}.

### Formation of information flow through interaction of sparse concepts

On the network perspective, training produces information flow that connecting dataset to results. The information flow is made possible by the interaction of sparse concepts. Phase transition corresponds to the low concepts to high concepts transition. Exploration of the complete parameter space is restricted to the optima manifold. Maybe we can establish a direct connection between information flow and region of optima manifold. Increase of the number of information flows restrict the optima manifold into a small subspace. Driven by flux, the optima manifold is still well-connected. 

Fisher Information of the weights to measure the effective connectivity between layers of a network during training. Counterintuitively, information rises rapidly in the early phases of training, and then decreases, preventing redistribution of information resources in a phenomenon we refer to as a loss of “Information Plasticity”. 
Evidence for phase transitions in learning dynamics \cite{achilleCRITICALLEARNINGPERIODS2019a}. 
Training and generalization is easier on clean and structured datasets and harder on noisy and unstructured datasets. Neural network lives in a low-dimensional information space (low rank Jacobian) \cite{oymakGeneralizationGuaranteesNeural2019}.
During the entire training process, feature distributions of differently initialized networks remain similar at each layer. Non-convex loss landscape can be reformulated as convex function with respect to the feature distributions in the hidden layers. Training follows a fixed trajectory in the feature distribution \cite{guHowCharacterizeLandscape2020}. This can be related to the permutation symmetry of the deep neural network. Also, the invariant of features, or the invariant of attention maps may indicate that although the ensemble of trained neural networks are just a set of equivalent mappings (in the sense of statistical).
Deep learning models are optimized for training data modularly, with different regions in the function space dedicated to fitting distinct types of sample information \cite{theunissenBenignInterpolationNoise2020}. Evidence for the existence of data-specific optima manifold. 
DNNs can be trained in low dimensional subspace: optimization in 40 dimensional spaces can achieve comparable performance as regular training over thousands or millions of parameters \cite{liLowDimensionalLandscape2021}. This has some similarity with the LoRA method, only a few crucial parameters should be adapted when learning a new task.
Using random matrix theory, quantify the weights eigenvalues. Firstly, initialized parameters have random-like empirical spectral density. Training starts to bleed out large eigenvalues gradually. Eventually leads to heavy-tailed eigenvalues, with most eigenvalues near zero \cite{martinImplicitSelfregularizationDeep2021}. The neural networks are becoming sparse during training, only a few parameters are important for the task.
Layer's output can be restricted to an eigenspace of its covariance matrix without performance loss. Layer saturation: the ratio between the eigenspace dimension and layer width \cite{richterFeatureSpaceSaturation2021}. 
Layerwise loss landscape analysis: eigenspectra of the Hessian at each layer. eigenspectra of the Hessian at each layer. Hessian eigenspectrum of middle layers of the deep neural network are observed to most similar to the overall Hessian eigenspectrum. Penalizing the trace of the Hessian at every layer indirectly forces Stochastic Gradient Descent to converge to flatter minima, which are shown to have better generalization performance \cite{sankarDeeperLookHessian2021}. Constrain layer-wise weights or weights along an inferencing trajectory?
Constructing multiple parameter modes and allocating tasks per mode, Mode-Optimized Task Allocation (MOTA). Training multiple modes in parallel, optimizes task allocation per mode \cite{dattaMultipleModesContinual2022}. Mode-specific local minima has a complicated topology structure in the loss landscape, the overlap of many modes gives the most generalized model. 
Batch-entropy quantifies the flow of information through each layer of a neural network. A positive batch-entropy is required for gradient descent-based training approaches to optimize a given loss function successfully. Expressivity of a network would grow exponentially with depth. Flow of information through neural networks \cite{peerImprovingTrainabilityDeep2022}. The expressivity can be quantified by the number of independent information paths, the collection of information paths form a path ensemble, which can be understood with thermodynamic properties.
Pre-training on ImageNet consistently removes the presence of barriers for ResNet architectures trained on CIFAR-10 data \cite{vlaarWhatCanLinear2022}. 
For transformers, sharpness does not correlate well with generalization but rather with some training parameters like the learning rate that can be positively or negatively correlated with generalization depending on the setup. In multiple cases, we observe a consistent negative correlation of sharpness with out-of-distribution error implying that sharper minima can generalize better. Right sharpness measure is highly data-dependent, and that we do not understand well this aspect for realistic data distributions \cite{andriushchenkoModernLookRelationship2023}. 
Layer Convergence Bias: shallower layers of DNNs tend to converge faster than the deeper layers. Flatter local minima of shallower layers make their gradients more stable and predictive, allowing for faster training. Another surprising result is that the shallower layers tend to learn the low-frequency components of the target function, while the deeper layers usually learn the high-frequency components. It is consistent with the recent discovery that DNNs learn lower frequency objects faster \cite{chenWHICHLAYERLEARNING2023}. Layer-wise analysis is also meaningful. Shallow layers may be more vague, while deep layers are more accurate. Learning is a gradual process, converge from shallow layers to deep layers, from less accurate to accurate, and from low frequency signal to high frequent signal.
Transformer's activation maps are sparse. By activation map we refer to the intermediate output of the multi-layer perceptrons (MLPs) after a ReLU activation function, and by “sparse” we mean that on average very few entries (e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input to MLP. Sparsity also emerges using training datasets with random labels, or with random inputs, or with infinite amount of data, demonstrating that sparsity is not a result of a specific family of datasets. Enforcing an even sparser activation via Top-k thresholding with a small value of k leading the model to have less sensitivity to noisy training data, more robustness to input corruptions, and better calibration for their prediction confidence \cite{liLazyNeuronPhenomenon2023}. Model sparsity, or sparse learner as transformer and other neural networks. Sparse means very little information flow is actually optimized through parameter updating.
DNN with strong generalization power usually learns simple concepts more quickly and encodes fewer complex concepts. Interactive concepts: visual concepts, or patches in vision models. The ways of interactions are exponentially many, so can be used to explain exponentially many related data that with the same concepts. Let a concept be frequently extracted by the DNN from training samples. If this concept is generalizable, then this concept is supposed to also frequently appear in testing samples. Otherwise, the concept is considered to have poor generalization power. Complex concepts are usually over-fitted. Simple concepts refer to concepts of low orders. Order of concepts is defined by number of input variables, $\text{order}(S) = |S|$. It is concept that memorized by a DNN to represent the interaction between input variables. It is also concepts that are encoded by the DNN. Concepts are also sparse. Inference score $\nu(\mathbf{x})$ on the sample $\mathbf{x}$ can be mathematically disentangled into numerical effects of different interactions, 
$$
\nu(\mathbf{x}) = \sum_{S \subseteq N} I(S | \mathbf{x})
$$
What is the nature of concepts? Maybe the concepts is just the information flows. What is the nature of computation? Maybe the actual type of compute is not important, but what is important is the interaction of information. Two numbers add up or multiplied, then they are interacted. But multiplication is stronger than add them up. DNNs may be understood from the hierarchy of concepts. Concepts in the shallow layer are combined to form concepts in the deep layers. Deep concepts are activated only when all shallow concepts are all presented in the input data. Maybe we can follow the information flow through the interaction of concepts. Deep concepts also have higher orders, because their orders are the add-up of shallow concepts. How to relate the interaction of concepts to generality of DNN? Forming low-order concepts first then high concepts making the DNN more likely to have permutive symmetry, and permutive symmetry is less sensitive to parameter shuffle. So flatness is only a phenomenon, the real reason for generalization is the permutive of concepts, i.e. the exponentially increased crossovers of information flows \cite{zhouConceptLevelExplanationGeneralization2023}.
Introducing an identity block, defined as $\phi_{id}(x) = x$, after each block in the original model, ensuring that the expanded model maintains the same output after expansion \cite{wuLLaMAProProgressive2024}. This is very similar to the base expansion technique in quantum physics. Can we describe the information flow on the concept basis, and viewing each layer as a hierarchical expansion of the representative base (concept base)?

### Identify the activity patterns of neural network

The collective behavior of neural network can be quantified by the correlation matrix. 

![correlation_matrices](figures/correlation_matrices.png)

To compute shuffled correlations, we circularly shifted each neuron’s activity in time by a random number of bins (at least ±1000) and correlated all the shifted traces with all the original traces. Spread was larger than would be expected by chance:

![test_correlation_against_null_hypothesis](figures/test_correlation_against_null_hypothesis.png)


Conjecture: Just like the brain, different categories of input data may produce different activity patterns. And it is the collective behavior of the network that responsible for the learning capability. The parameters should be sorted vertically by first PC weighting or by manifold embedding algorithm. It is also horizontally sorted by the data category. The diagram shows the high-dimensional population activity for different input data. Can be viewed like a stream of data. 

![parameter_activity](figures/parameter_activity.png)

The computational activity is highly correlated within the same data category, but rapidly drop to zero for different categories. Autocorrelation of PCA directions are shown below. The result demonstrates a task-specific structured activity, i.e. population computational activity is highly correlated for the same category of data. 

![autocorrelation_between_data_categories](figures/autocorrelation_between_data_categories.png)

Analysis the dimensionality of the computational activity for streamed data input with SVCA, and analysis the number of reliable variance. The magnitude of reliable spontaneous variance was distributed across dimensions according to a power law of exponent 1.0 seen for stimulus responses in brain: Power-law scaling of dimensionality \cite{stringerHighdimensionalGeometryPopulation}. The power-law distribution of reliable variance indicates a high-dimensional signal. 

![power_law_means_high_dimensional_signal](figures/power_law_means_high_dimensional_signal.png)


Since the connectivity of the neural network is available, we can also follow the computational path from the data to final output. The computational path is connected through the weight parameters. 

![node_computational_network](figures/node_computational_network.png)


Define computational activity as a correspondence of neuron activity. 

Since the output of computational network node is the direct result of the participation of the parameters of this node, we use the computational activity of node and computational activity of the parameters interchangeably. The same activity value is assigned to the parameter set that belong to the same node. 

![computational_path_ensemble](figures/computational_path_ensemble.png)

Conjecture: Although there will be combinatorial many possible computational paths, the number of activated computational paths is rather small. 


The patterns of computational paths can be embedded into a low-dimensional manifold, just like the neuron manifold. 

![manifold_embedding](figures/manifold_embedding.png)

Conjecture: training from different initial state may result into the same pattern of computational activity, and the same low-dimensional manifold. 

Conjecture: redundant of network weight parameters produces extra symmetries leading to the same computational manifold. The demanded computational resources are also high when using redundant parameters. However, the redundant is crucial since it facilitate the training of the network. 

Input of different categories of data produces different kind of computational paths. The prediction of neural network corresponds to the behavior of animal which links data to predictions. We can establish the correspondence through the reduced rank regression technique: find the correspondence coordinate between reliable variance and principal components of data and activity. We can also try to predict the activity patterns from the task, which will confirm the close connection between task and activity patterns. 

![reduced_rank_regression_technique_seek_for_correspondence](figures/reduced_rank_regression_technique_seek_for_correspondence.png)

Since natural data are distributed on a low-dimensional manifold, the linkage of distribution of data to predictions are also a low rank map. The map can be represented within a low-dimensional function space, which defines the prediction manifold. 

![neural_behavior_correspondence](figures/neural_behavior_correspondence.png)

Conjecture: there will be a correlational correspondence between computational manifold and prediction manifold. 

We then define the concept as the cluster of computational paths. Then concepts are directly comes from the combination of numbers and the mappings that connect these numbers. 

The constructive (destructive) interference of concepts: joint of two concepts in high level (deep layer) can result in stronger (weaker) activation. 


Before learning, the eigenspectrum was uniformly distributed. During training, the eigenspectrum decayed so that to minimize the frustration and the activity was reduced to a low-dimensional manifold. 

![power_law_eigenspectrum](figures/power_law_eigenspectrum.png)


In brain, sensory inputs is added to the original activity through extra dimensions. The orthogonal dimension here effectively avoid the interference of ongoing signal and external stimuli \cite{stringerSpontaneousBehaviorsDrive2019}. This orthogonal fusion of information is effective for avoiding destructive interference of concepts. 

If input data is pure noise, there will also be spontaneous behavior. 

ANN captures the low-dimensional manifold just like the human brain, so machine can at least be as smart as human brain. 

### Applications

#### Boundary of generalization

Input of random data produces probability flux, the learning is driven both by gradients and flux. Both gradient and flux drive the model to be settled at a connected cluster of local minima, the volume of this cluster may be directly related to the ability of generalization.

Generalization may be understood through two-side way. 
- Essential condition: distribution of data is continuous, therefore if we have learned to identify a data point, any data points that are very near this data point can also be identified. 
- Sufficient condition: loss landscape has a flat local minima, therefore the neural network can map the complete support set of data distribution into this flat local minima

Continuous data distribution --> loss landscape flat local minima

For quantum system, using povm to represent quantum state, using reinforcement learning to increase learning efficiency.

Low-loss region with in-distribution data, yield significantly different loss landscapes with out-of-distribution data \cite{fangRevisitingDeepEnsemble2023}. Generalization has a limit, and the limit is determined by the similarity of test data to train data. If test data is very different from train data, there will not generalize well. The minima of test data will also not be the same as the train data, but very close to the train data. So if the minima of train data is flat, that means the test data will also have a low loss.

#### Continue-learning

Five different mechanisms designed to mitigate catastrophic forgetting in neural networks: regularization, ensembling, rehearsal, dual-memory, and sparse-coding \cite{kemkerMeasuringCatastrophicForgetting2017}.
Inspired by synaptic consolidation in neuroscience, a weight-regularization method that protects the weights important for previous tasks enables state-of-the-art results on multiple reinforcement learning problems experienced sequentially \cite{kirkpatrickOvercomingCatastrophicForgetting2017}. 
Learn to forget for new tasks in meta-learning \cite{baikLearningForgetMetaLearning2020}. 
Similar to biological neural networks. Resolve conflicts in neural network for new and old tasks is crucial.
A set of binary masks for important weight parameters of DNN, has almost the same performance with fine-tune to adapt to new task \cite{zhaoMaskingEfficientAlternative2020}. Using weight occupancy for continuous learning and neural network structure evolution. 
"Sensitivity-stability" dilemma: trained neural networks have catastrophic forgetting phenomenon. Flattening Sharpness for Dynamic Gradient Projection Memory: calculate importance of basis related to old task, less important bases can be dynamically released to improve the sensitivity of new skill learning \cite{dengFlatteningSharpnessDynamic2021}. Old concepts should be removed if not adapted to new tasks. 
Low-Rand Adaptation (LoRA), freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. LoRA can reduce the trainable parameters by 10,000 times and GPU memory 3 times \cite{huLoRALowRankAdaptation2021}.
Furlanello et al. show that multiple rounds of distillation between models with the same architecture (termed self-distillation) can surprisingly improve the performance of the student. Flat minima also contribute to continuous learning \cite{boschiniClassIncrementalContinualLearning2022}. Class-Incremental Continual Learning: is forgetting necessary? Explain model capacity, continuous learning and catastrophical forgetting with the occupancy of mapping trajectories. 
Multi-task learning, the objective function is a weighted sum of all the tasks. Random grouping task training, shared parameter space \cite{pascalImprovedOptimizationStrategies2022}. For conventional single-task training, the objective function is also a weighted sum of data-specified terms.
Deep, nonlinear ANNs can learn new information by interleaving only a subset of old items that share substantial representational similarity with the new information. Similarity-weighted interleaved learning. Data efficiency and speedup in learning new items are increased roughly proportionally to the number of nonoverlapping classes stored in the network. Training with similarity-weighted interleaving of old items with new ones allows deep networks to learn new items rapidly without forgetting. The cosine similarity between the average per-class activation vectors for existing- and new-class items for a target hidden layer. A new item is called consistent if it could be added to the previously learned classes without requiring large changes to the network \cite{saxenaLearningDeepNeural2022}.  Corner-stone data that construct the old memory. If new data is not overlapped with old memory, learning can speed up. Maybe for interleaved data class, the neural network just need more training to classify the boundaries of data manifold. What is the fundamental reason for the interleaving learning algorithm? Can we explain these from the basis of loss landscape perspective? If new items and old items are overlapped, then the information flows may be overlapped completely or just in the shallow layers. The whole story of deep learning is about controlling the information flow which is determined by the architecture of the neural network. Training is a variational inference procedure to obtain the correct information flow. Information flow is in fact another name of dynamics, where information cluster and spread across the state space along with time. Also see the Hypothesis for more details. 
Model ensemble can strongly improve the transferability in adversarial attacks, flatness of loss landscape and the closeness to the local optimum of each model \cite{chenRethinkingModelEnsemble2023}. An ensemble perspective of training, think about many parallel models and describe the behavior of training from probabilistic perspective.
Continual learning as ensuring a proper stability-plasticity trade-off and an adequate intra/inter-task generalizability in the context of resource efficiency \cite{wangComprehensiveSurveyContinual2023}.
Inspired by the generative nature of the hippocampus as a short-term memory system in primate brain, we propose the Deep Generative Replay, a novel framework with a cooperative dual model architecture consisting of a deep generative model (“generator”) and a task solving model (“solver”) \cite{shinContinualLearningDeep2017}.
Synaptic Intelligence: consolidating changes in parameters that were important for solving past tasks, while allowing only the unimportant parameters to learn to solve future tasks \cite{zenkeContinualLearningSynaptic2017}. 
Internal or hidden representations are replayed that are generated by the network’s own, context-modulated feedback connections \cite{vandevenBraininspiredReplayContinual2020}. 
Identifying the most relevant data from these extensive datasets to effectively develop specific capabilities. Low-rank gradiEnt Similarity Search for instruction data selection: first constructs a highly reusable and transferable gradient datastore with low-dimensional gradient features and then selects examples based on their similarity to few-shot examples embodying a specific capability \cite{xiaLESSSelectingInfluential2024}. This is another way of calculating the similarities of input data. 

**Can we add time (or other additional scene labels) to the memories? Just like the MET model, adding labels to the memories can solve the conflict of old and new memories.**

In training neural networks on a dataset with many categories, there is an optimal local minima for each category. The neural network will try to fit to each category whenever it can. So the nn will always attempt to be overfitting, as driven by the loss function. We can call the local minima obtained for each category as **overfit local minima**. But for other categories, the overfit local minima is not the same as yet still near the overfit local minima of the original category. Therefore, a flat local minimum ensures almost the same loss for all the categories. 
So we have the hypothesis: **a well-designed neural network can always compress overfit local minima within a public region**. 

If we consider the neural network itself as a generalized dynamical system, the data is the initial state and the output of the neural network is the final state. Then the loss function is a functional of the final state. Each final state corresponds to a loss function value. The dynamical properties of the neural network then always ensures all the initial states converges to a final state with small divergence to a reference ground truth. 

The target of continuous learning is to learn a series of tasks

$\{ p_{\eta_1}(x), p_{\eta_2}(x), \ldots \}$

Each task is governed by a loss function

$\mathcal{L} = \mathcal{E} (p_\theta(x), p_\eta(x))$

Each loss function defines a task specified local minima, whose position in the parameter space is specified by the architecture of the model, the formula of the loss function and the protocol of training.
During training, the loss function is estimated by samples from the dataset, with the sampling distribution $p_s$.
$p_s$ is usually uniform for supervised learning, and can be determined by the model for cases such as the reinforcement learning.
The number of samples for one epoch training is the batch size.
The loss function is 

$\mathcal{L} = E_{x \sim p_s} \{\mathcal{E}(p_\theta(x), p_\eta(x))\}$

If x is sampled from a composed dataset with many tasks.
The composed data distribution is 

$p_c(x) = \sum_i w_i p_{s_i}(x)$

where $w_i$ is the appearance weight of the i-th task.
The composed loss is

$\mathcal{L} = E_{x \sim p_c} \{\mathcal{E}(p_\theta(x), p_\eta(x)\} = \sum_i w_i \mathcal{L}_i$

which is

$\sum_i w_i E_{x \sim p_{s_i}} \{\mathcal{E}(p_\theta(x), p_{\eta_i}(x)\}$

The gradient is

$\nabla_\theta \mathcal{L} = \sum_i w_i \nabla_\theta \mathcal{L}_i$

The loss landscape of the composed loss function is then the weighted add up of the individual loss landscape for each task.
The add up is much like the mixture of probability densities, resulting in **flatten region** around the local minima of each task in the parameter space.
This is the **origin of flatten minima**.
The final trained model will reach some compensate state that respect all the tasks.
If the model is powerful enough, it may find a flat minima that circumvent the current task specified local minima, where the model performs better than single task training.
This has some **similarity to protein folding**.
We may also expect a funneled loss landscape.
The training of the model follows a glass transition, reaching some region that is strongly restricted by the loss function.
When increase learning rate, decrease batch size, reduce dataset size, more noise is introduced to the loss landscape.
And the stability of the glass state decreases.
If the model size increases, becoming wider or deeper, the bottom of the loss landscape also enlarges, allowing for learning diverse tasks. 

If the model is large enough, the composed loss landscape flatten the transition region between task specified local minima, creating a flatten minima which exhibits generalicity for all the tasks.

So the question for continuous learning is, how do we estimate the composed gradient without data from previous task?

If there is only one variable dimension, two distinct tasks will have high transition barrier.
If adding additional variable dimension, the two task local minima will degenerate into one local minima.
It is like the relation between marginal and joint distribution.
The problem is to estimate the joint state provided the marginal information.

Some parameters are fixed for fine tune and biological neural network tends to update weights in energy efficient way 
> Inferring neural activity before plasticity as a foundation for learning beyond, nature neuroscience

so maybe we need to update the weights through intelligent way, to give specific learning rate to each weight of the model, mimiking the updating strategy of biological neural network.


$\frac{1}{\sum_i n_i} \{\sum_i n_i \mathcal{E}(p_\theta(x), p_{\eta_i}(x)) \}$

**Assign an occupy probability $p_o$ to every weight. If $p_o$ is 1, the weight is very important to current task and is not change for other tasks. If $p_o$ is not zero, the weight will be updated with probability (1-$p_o$). In this way, we can also modify the network topology according to $p_o$. $p_o$ also quantify the balance of memory and forgetting.**

The occupy probability has already been tried, which is categorized as regularization-based method for alleviate catastrophic interference. Some representative studies: 
Elastic Weight Consolidation (EWC) \cite{kirkpatrickOvercomingCatastrophicForgetting2017}, Learning without Forgetting \cite{liLearningForgetting2017}, Synaptic Intelligence \cite{zenkeContinualLearningSynaptic2017}. All these methods measure the importance of each parameter and adding a regularization term that penalizes changes in the most relevant parameters or mapping function of the network. These methods suffer when there is a need to learn many new classes incrementally \cite{kemkerMeasuringCatastrophicForgetting2017}\cite{vandevenBraininspiredReplayContinual2020}. 
Another category: replay-based approaches \cite{vandevenBraininspiredReplayContinual2020}\cite{shinContinualLearningDeep2017}. This is very like the distillation method that I used in the MET article. 
Replay-based methods solved the problem of not having access to the old data. 
Problem is shifted toward implementing an improved generator???
If we are using a generative model, then it is not a problem at all!!! Generative models can dream, which is a feed-back replay for learning new items. Generative replay. 
Another problem is new items must be interleaved with all old items. 
Computing similarity of new items with representative old items using each class's average activation helps reduce the computational burden. 
If the problem of learning new item is the similarity with old items, then maybe it is a boundary problem. The network must be adapted to the new boundaries of new items with old items. Then maybe the average similarity is not the best choice for the representative old items. We may need an item-wise identification of the dataset, or we can just call this property the "curiosity". Meaning something that is highly different to the old memory. 

The regularization-based approach and the replay-based approach may be just the two-side of a same coin. Both these methods try to establish new information flow while keeping the old information flow unchanged or have minimum changes during training. 
What if we treat sleeping as a knowledge distilling procedure? Dreams is the generative replay of long-term memory. What we see is what we dream, meaning only similar data items is recalled. These data items and the new short memory items consists of the new dataset for training. The network (which is a copy of yesterday's network) distills knowledge from both the long-term memory and the new short-memory items. The updating of weight can follow the synaptic importance regularization, which follows a lazy updating manner. 
We can consider the whole training procedure of a network also as a memory distilling process: random sampling of items from dataset is acting like generative replay, and batch average of items is acting like regularization for consolidating the long-term and short-term memory. 
We may prove the generalization is an essential property of network. And the concerto of long-term and short-term memory is the reason for generalization. 
We should go deep into the neural network, to see how the network learns new classes. 
Transfer-learning approaches freeze initial layers of the networks during new-class learning, and forgetting first happens at the top layers to output layers. From information flow perspective, if many patterns of flow have already established, then learning new item might be faster, since there is a high probability of new information being consistent across many more dimensions with the existing classes and features. That is, little effort is needed for updating the weights. 
Curiosity, or the driving force to learn new knowledge, may significantly increase the generalization ability, and also improve the understanding of old knowledge. That is, network trained on many classes with always generalize well than a network trained on few classes. The network pretrained with large dataset will also have better performance, and it will also learn more quickly than a network trained with small dataset. 
Reactivation probability proportional to similarity, as shown previously in simulations of hippocampal attractor dynamics \cite{wilsonReactivationHippocampalEnsemble1994}. 
Mismatch vs. poor similarity vs. surprise, as described in the Adaptive Resonance Theory \cite{grossbergHowDoesBrain1980}.
Replay dynamics and memory consolidation may be the key to understand neural network learning, to achieve lifelong learning in neural networks \cite{saxenaLearningDeepNeural2022}.

Another option:
Perhaps we should fully explore the weight parameter space for one task, save many models, and match these models with the new task. In fully explore the parameter space, the public region may be covered, which allows the memory of the old task when training new task. However, this method can be very inefficient, especially for large models. 

Human: "ju yi fan san"?

Curiosity?
curiosity: pay attention to something not seen before, or has very deep impressions.
learning should be based on some cornerstone data, which are very representative and sparse. only a few data is enough
LLM now has memory better than anyone in the world. the only difference is we have curiocity
similarity-weighted interleaved learning
> \cite{saxenaLearningDeepNeural2022}

[LoRA + MoE](https://zhuanlan.zhihu.com/p/685580458?utm_campaign=shareopn&utm_medium=social&utm_oi=52760531697664&utm_psn=1749690755726213120&utm_source=wechat_session)
https://arxiv.org/abs/2312.09979

Block expansion for solving catastrophic forgetting:
> LLaMA Pro: Progressive LLaMA with Block Expansion
Paper: https://arxiv.org/abs/2401.02415
Github: https://github.com/TencentARC/LLaMA-pro
[block expansion](https://www.zhihu.com/question/640869050/answer/3373620546?utm_psn=1734995754715004928)


ALRS, slow decay of learning rate for more generalization:
> Bootstrap Generalization Ability from Loss Landscape Perspective
[slowly decay to flat region](https://www.zhihu.com/question/638766873/answer/3358801861?utm_psn=1734998124115066880)


## Conclusion

Some claims:
- data continuity: adversarial attack destroyed the data continuity, so lead to failure of inference.
- loss landscape is highly connected: from any random initial configuration to this optima manifold can be achieved with a very limited training epochs \cite{gur-ariGradientDescentHappens2018}.
- Training different data leads to merge of steady-states: Data specific steady-state states is distributed in a tiny subspace of the parameter space. The support of the steady-state distribution is a manifold of many modes. These modes are connected by low barrier bridges. The shape of this manifold is more like a ring in the high-dimensional parameter space. We can term this structure optima manifold.
- small loss constrained the subspace of loss landscape into a small optima manifold, Volume of the optima manifold is small, compared to the size of the parameter space.
- common optima manifolds are exponentially rare.
- loss landscape is funneled and has very complex yet highly connected bottom topology. There is no unique folded state.
- Flux driven by the input data creates a series of overlapped basins, forming a connected optima manifold. 


**For every continuous distribution, there exists an optima manifold**.

There are many local minima for each partition of the dataset, so the overlay of these local minima indicate the global minima for all the partitions of the dataset. **Is the spatial distribution of local minima the most important?**

The cognition of brain is thought to be built on the network of neurons. Instead of relying on single neurons, the brain is believed to represent concepts and memories to through patterns of activity across large networks of neurons. The collective behavior of neurons allow for more flexible and robust representations that can adapt to new information and experiences. The information pass through the brain with a hierarchical manner, with lower-level sensory information being combined and integrated into higher-level representations, patterns or concepts. Handling sensory information is done through a modularity strategy, that different regions of the brain are specialized for processing different types of information, such as visual, auditory, or linguistic input. This bottom-up process is accompanied by a top-down processing, where prior knowledge and expectations influence the interpretation of new information. 


## Methods

Non-linear dimensionality reduction of the jump and retrain trajectories via PHATE, to visualize loss landscape geometry and topology \cite{horoiExploringGeometryTopology2022}.
Layer-wise conditioning analysis: explore loss landscape w.r.t. each layer independently \cite{LayerwiseConditioningAnalysis2020}.
Visualize the geometry of equivalences for neural networks \cite{lengyelGENNIVisualisingGeometry2020}.
Visualize the geometry of loss landscapes to understand generalization. 
Landscape is visualized using t-SNE embedding \cite{huangUnderstandingGeneralizationVisualizations2020}. Using UMAP may be better.
Attention layers can perform convolution and, indeed, they often learn to do so in practice. A multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer \cite{cordonnierRELATIONSHIPSELFATTENTIONCONVOLUTIONAL2020a}. Tools for visualize self-attention map.
Sharpness aware training to find more generalizable flat minima \cite{abbasSharpMAMLSharpnessAwareModelAgnostic2022}. Is flat minima unique in multi-task training?
Sharpness-aware minimisation by considering the information geometry of the model parameter space, replacing the Euclidean balls with ellipsoids induced by the Fisher information \cite{kimFisherSAMInformation2022}.
Reduce computational efficiency and scale up batch size for sharpness-aware minimization \cite{liuEfficientScalableSharpnessAware2022}. 
Improve sharpness-aware minimization by randomly smooth the loss landscape \cite{liuRandomSharpnessAwareMinimization2022}.
Stochastic weight averaging (SWA): SWA averages the checkpoints along the SGD trajectory using a designated learning rate schedule for better exploration of the parameter space \cite{luImprovingGeneralizationPretrained2022}. Saving many checkpoints for one training task reveals information about the loss landscape, which can guide further exploration of the training. 
Uses the Fisher information to identity the important parameters and formulates a Fisher mask to obtain the sparse perturbation \cite{zhongImprovingSharpnessAwareMinimization2022}. Using Fisher information matrix to seek for the important parameters. 
Selection of inherited models by injecting random noises. Flatness, or robustness can be quantified by loss drop with invariant injected noise. 
Evidence for the merge of local minima is the reason for existence of optima manifold \cite{duNoisebasedSelectionRobust2020}.

Attention landscape: plot the collection of attention maps.

### Hessian matrix

The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function.

$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

Positive definite Hessian: If the Hessian matrix is positive definite at a point, it means that the loss function has a local minimum at that point. The eigenvalues of the Hessian matrix are all positive, indicating that the curvature is positive in all directions.
Negative definite Hessian: If the Hessian matrix is negative definite at a point, it means that the loss function has a local maximum at that point. The eigenvalues of the Hessian matrix are all negative, indicating that the curvature is negative in all directions.
Indefinite Hessian: If the Hessian matrix has both positive and negative eigenvalues, it means that the point is a saddle point in the loss landscape. The curvature is positive in some directions and negative in others.
Computing the full Hessian matrix can be computationally expensive, especially for high-dimensional models. Approximations like the Gauss-Newton matrix or the Fisher information matrix are often used instead.

When the Hessian matrix is low rank, it means that the matrix has a large number of zero eigenvalues or eigenvalues that are close to zero. In other words, the Hessian matrix is not full rank, and its rank is much smaller than its dimensions.
Flat directions in the loss landscape: A low-rank Hessian indicates that there are many flat directions or plateaus in the loss landscape. In these flat directions, the curvature of the loss function is close to zero, meaning that the loss function does not change significantly when moving along these directions.
A low-rank Hessian can make the optimization problem ill-conditioned. Ill-conditioning means that small changes in the input (model parameters) can lead to large changes in the output (loss function). This can make the optimization process sensitive to numerical instabilities and slow down convergence.
Redundant or correlated parameters: A low-rank Hessian may suggest that some of the model parameters are redundant or highly correlated. This can happen when there are more parameters than necessary to capture the underlying patterns in the data, leading to over-parameterization.
Challenges in second-order optimization methods: Second-order optimization methods, such as Newton's method, rely on the inverse of the Hessian matrix to determine the update direction and step size. When the Hessian is low rank, the inverse may not exist or may be numerically unstable, making it difficult to apply these methods directly.

### Fisher information matrix

The time correlation of noise is very much similar to the Fisher information matrix:

$I_{ij} = \mathbb{E}\_{\mathbf{x} \sim p} \frac{\partial \log q_{\mathbf{\omega}}(\mathbf{x})}{\partial \omega_i} \frac{\partial \log q_{\mathbf{\omega}}(\mathbf{x})}{\partial \omega_j}$

### SVCA and PCA 

We can identify the dimension of computational activity through Shared Variance Components Analysis (SVCA), an analogy of Principle Components Analysis (PCA). SVCA is designed to identify shared variance among multiple datasets, which can represent different categories of data. This shared variance is more likely to reflect underlying intrinsic mechanisms that are consistent across datasets. In contrast, PCA identifies the main sources of variance within a single dataset, which may not necessarily be mechanical meaningful or consistent across different datasets. SVCA is more robust to noise and artifacts compared to PCA. By focusing on the shared variance across datasets, SVCA can effectively filter out dataset-specific noise or artifacts, whereas PCA may be more sensitive to these factors. The components identified by SVCA are often more interpretable in terms of their biological significance, as they represent patterns of neural activity that are consistent across datasets. PCA components, on the other hand, may be harder to interpret biologically, as they are derived from a single dataset and may not generalize to other datasets. SVCA can be applied to multiple datasets with different dimensions, such as different numbers of neurons or time points. This flexibility allows for the integration of data from various sources and modalities. PCA, in contrast, is typically applied to a single dataset with fixed dimensions. SVCA can be used to test specific hypotheses about the shared variance between datasets, such as the presence of a common neural mechanism or the effect of an experimental manipulation. PCA is more data-driven and exploratory, making it less suitable for hypothesis testing. SVCA has gained popularity due to its ability to identify biologically relevant and consistent patterns of neural activity across multiple datasets.



