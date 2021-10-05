# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: ToolkitEnv2
#     language: python
#     name: toolkitenv2
# ---

# %% [markdown]
# # SNU
#
# ## Spiking Neural Unit concept
#
# Biological neurons receive input spikes, which are integrated into the membrane potential $V_m$ of the soma and lead to the emission of output spikes through the axon when $V_m$ crosses the spiking threshold $V_{\text{th}}$. These dynamics are often modelled using RC circuits. SNU models the spiking neural dynamics in the form of two ANN neurons, performing the integration and emission of output spikes. 
#
# ![image.png](attachment:image.png)
#
#
# We introduce a novel high-level abstraction of the LIF dynamics into an ANN unit, which we call a Spiking Neural Unit (SNU). The SNU comprises two ANN neurons as subunits: $N_1$, which models the membrane potential accumulation dynamics, and $N_2$, which implements the spike emission, as illustrated above. The integration dynamics of the membrane-potential state variable are realized through a single self-looping connection to $N_1$ in the accumulation stage. The spike emission is realized through a neuron $N_2$ with a step activation function. Simultaneously, an activation of $N_2$ controls the resetting of the state variable by gating the self-looping connection at $N_1$. Thus, the SNU — a discrete-time abstraction of an LIF neuron — represents a construct that is directly implementable as a neural unit in ANN frameworks, where it may be scaled to deep architectures. 
#
# Following the ANN convention the formulas that govern the computations occurring in a layer of SNUs are as follows:
#
# $$
# s_t = g(W x_t + l(\tau) \odot s_{t-1} \odot (1-y_{t-1}) \\
# y_t = h(s_t + b),
# $$
#
# where $s_t$ is the vector of internal state variables calculated by the $N_1$ subunits, $y_t$ is the output vector calculated by the $N_2$ subunits, $g$ is the input activation function, $h$ is the output activation function, and $\odot$ denotes point-wise vector multiplication. 
# The neuron $N_2$ with a step activation function reproduces the spiking behaviour, whereas the same neuron with a sigmoid activation function generalizes the neural dynamics beyond the spiking case.
# An illustration of an SNU using the convention of standard ANNs is presented below. This depiction enables the unique features of the SNU to be identified, viz., a non-linear transformation $g$ within the internal state loop, a parametrized state loop connection drawn in bold, a bias of the state output connection to the output activation function $h$ drawn in bold, and a direct reset gate $(1−y)$ controlled by the output $y$.
#
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ## Results on common benchmarks
#
# We have compared SNUs and common deep learning units, such as LSTMs and GRUs, on tasks including image classification, language modelling, music prediction and weather prediction. Importantly, we demonstrated that although SNUs have the lowest number of parameters, they can surpass the accuracy of these units and provide a significant speed up. Our results established the SNN state-of-the-art with the best accuracy of 99.53% +/- 0.03%, for the handwritten digit classification task based on the rate-coded MNIST dataset using a convolutional neural network. Moreover, we even demonstrated the first-of-a-kind temporal Generative Adversarial Network (GAN) based on an SNN. The figures below illustrate the performance of SNU-based networks in the various tasks.
#
# ![image-3.png](attachment:image.png)

# %% [markdown]
# ## Further details
#
# More scientific details can be found in [Nature MI paper](https://www.nature.com/articles/s42256-020-0187-0).
# To build on top of SNU dynamics in your applications, please explore the code examples.
