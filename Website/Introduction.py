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
# # Introduction
#
# The astonishing progress in artificial intelligence in recent years has been driven by the insights on how the neural networks in the brain work. However, apart from the networked structure of modern ANNs (Artificial Neural Networks), biology exhibits much more versatility in terms of dynamics, connectivity patterns and learning capabilities.

# %% [markdown]
# ## Why this Neuro-inspired AI Toolkit?
#
# To improve the efficiency, accuracy and to push forward the limits of modern AI, we focus on the application of biologically-inspired insights to practical Machine Learning tasks.
#
# ### For researchers
#
# On one hand, existing ANN frameworks, such as TensorFlow or PyTorch, largely focus their functionality on the state-of-the-art recurrent neural units (LSTMs, GRUs) and have limited capabilities to support biological features. On the other hand, existing SNN frameworks, such as Brian2 and Nengo, include abundance of biological features that model low-level neurosciencific insights. However, their purpose is understanding and fitting biological measurements. In consequence, often it is unclear how these insights could translate to improved performance on machine learning tasks. This Toolkit fits in-between: it provides a framework to seamlessly integrate neural networks that incorporate core biologically-inspired dynamics with the training and assessment procedures of ML benchmarks. Moreover, it can be easily extended to include novel biologically-inspired features.
#
# ### For engineers
#
# Neuro-inspired AI Toolkit's modules can be integrated into typical AI frameworks. It provides means to easily incorporate qualitatively new dynamics into your machine learning models, that enable efficient solutions of existing tasks (e.g. SNU unit can be up to 8x faster than LSTM) or enable solutions of completely new tasks. Code changes are minimal and typically involve just subsituting the units in the network definitions or the optimizer. See examples section for code details.

# %% [markdown]
# ## Neuro-inspired AI
#
# Spiking Neural Networks (SNNs) incorporating biologically-plausible neurons hold great promise because of their unique temporal dynamics and energy efficiency. However, developments in SNNs have been proceeding separately from Artificial Neural Networks (ANNs), which limited the adoption of deep learning insights to SNNs.

# %% [markdown]
# ### Modelling SNN dynamics with SNUs
#
# In [Nature MI paper](https://www.nature.com/articles/s42256-020-0187-0) we show an alternative perspective on the spiking neuron that incorporates its neural dynamics into a recurrent ANN unit called a Spiking Neural Unit (SNU). SNUs may operate as SNNs, using a step function activation, or as ANNs, using continuous activations. We demonstrate the advantages of SNU dynamics through simulations on multiple tasks and obtain accuracies comparable to, or better than, those of ANNs. The SNU concept enables an efficient implementation with in-memory acceleration for both training and inference. We experimentally demonstrate its efficacy for a music-prediction task in a first-ever in-memory-based SNN accelerator prototype using 52,800 phase-change memory  devices. Our results open up a new avenue for a broad adoption of biologically-inspired neural dynamics in challenging applications and acceleration with neuromorphic hardware. 
#
# ![image2.png](attachment:image.png)
#
# See more in SNU section and in [S.Woźniak et al., Nature MI, 2020](https://www.nature.com/articles/s42256-020-0187-0).
