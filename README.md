# complex-system-model-of-bipolar-disorder

The scripts in this repository are suplamentary to our paper "Toward a complex system
understanding of bipolar disorder: A chaotic model of abnormal circadian activity rhythms in euthymic bipolar disorder" published in Australian & New Zealand Journal of Psychiatry. If this code is used please cite the paper:

@article{hadaeghi2016toward,
  title={Toward a complex system understanding of bipolar disorder: a chaotic model of abnormal circadian activity rhythms in euthymic bipolar disorder},
  author={Hadaeghi, Fatemeh and Hashemi Golpayegani, Mohammad Reza and Jafari, Sajad and Murray, Greg},
  journal={Australian \& New Zealand Journal of Psychiatry},
  volume={50},
  number={8},
  pages={783--792},
  year={2016},
  publisher={SAGE Publications Sage UK: London, England}
}

# Model 
Fig 1. depicts a schematic representation of the proposed complex system. Cortical cognitive processing and subcortical circadian pacemaker are building blocks of the system. The primary cognitive network illustrates the interaction between two excitatory and inhibitory pathways
spatially spread over the frontal cortex and varying on a daily timescale. As a result of interactions among these neurons, various complex behaviors can emerge in the system’s long-term dynamics. To investigate the role of weighted excitatory and inhibitory units in the cognitive
process, we chose to mathematically represent the activation functions of each neuronal population via a hyperbolic tangent functions. Multiplied by the coefficients, A and B, the role of each population can be strengthened or weakened. It is important to note that the positive sign for
the output of E(x) in Figure 1 represents its excitatory effect, while the negative sign for the output of I(x) represents its inhibitory role. Therefore, the performance of this network can be mathematically integrated in the following recurrent map in which iteration number, n, captures the
low to high transitions across a daily cycle

![](https://github.com/m00rcheh/complex-system-model-of-bipolar-disorder/blob/main/Eq1.png)

In this formalism, the coefficients, A, B, w1 and w2 , symbolize the weights in brain synapses which are associated with the release of diverse neurotransmitters. The influence of excitatory and inhibitory brain actions can be adjusted through the values of A and B. Perturbation in the normal values of these coefficients represents abnormal quantities/activities of inhibitory neuro-
transmitters (e.g. GABA or dopamine) and excitatory neurotransmitters (glutamate or norepinephrine). Various complex behaviors can be captured by change in these coefficients (see bifurcation diagrams bellow). We assume that normal circadian activity has a trajectory close to the edge of chaos in the
periodic window of a nonlinear dynamical system. Therefore, using the products of this network, at a given point of time, an error estimator unit calculates the deviation of the trajectory from the edge of chaos and sends this signal to regulate the parameters in the circadian pacemaker block. As stated in equation (2), working at the edge of a period-p window, at each time instance, absolute difference between the current sample and the p-th sample ahead provides the error signal, e(n)

![](https://github.com/m00rcheh/complex-system-model-of-bipolar-disorder/blob/main/Eq2.png)

where n1 and n2 are, respectively, the first and the last time points of an arbitrary time window.
The proposed circadian pacemaker includes two populations of neurons. Regarding the output of the error estimator, the primary cortical network regulates the duty time and the amplitude of coming pulse generated by the secondary subcortical network, which works at the timescale of hours. To model the adaptive pulse generator unit, we chose to use a version of the Rulkov recurrent map which has been proposed to mathematically realize the action potential restitution curve in ventricular cardiac cells (Rulkov, 2007).

# Prerequisites
In order to run the scripts, Python 3.0, Numpy>=1.13.3, and matplotlib>=3.0.3 have to be installed on your local machine. 

# Codes
- Bifurcation.py returns two matrices and plots the bifurcation diagrams of the cognitive block of the model by varying the inhibitory (A), and excitatory (B) parameters. 

![](https://github.com/m00rcheh/complex-system-model-of-bipolar-disorder/blob/main/BifurcationDiagram.png)

- DailyCycle.py returns three time-series corresponding to normal, depressive, and manic states. 

