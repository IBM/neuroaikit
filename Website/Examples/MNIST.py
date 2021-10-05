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
# | [![View in NeuroAIKit Documentation](../_static/toolkit_32px.png)](https://ibm.github.io/neuroaikit/Examples/MNIST.html) |  [![View source on GitHub](../_static/github_32px.png)](https://github.com/IBM/neuroaikit/blob/master/Website/Applications/MNIST.ipynb) |
# | - | - |
# | View in NeuroAIKit Documentation | View source on GitHub |

# %% [markdown]
# # Image recognition (MNIST)
#
# This tutorial shows how to:
#
# 1. Transform static images from MNIST dataset into a temporal stream of spikes (rate coding).
# 1. Build a conventional LSTM network and a spiking network.
# 1. Train and evaluate the accuracy and the speed of both models.

# %% [markdown]
# ## Prepare the input data

# %% [markdown]
# Import common and TensorFlow-specific modules:

# %%
import neuroaikit as ai
import neuroaikit.tf as aitf

# %% [markdown]
# and other modules:

# %%
import time
import tensorflow as tf

# %% [markdown]
# Load MNIST dataset:

# %%
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

# %% [markdown]
# This is a static dataset comprising 60000 training images of size 28x28:

# %%
train_x.shape

# %% [markdown]
# Let's use the `transform_rate` function to convert the grayscale pixel values into spike rates over `Ns` time steps:

# %%
Ns = 20
max_rate_spikes = 6

# %%
# This may take several seconds to execute:
train_x = ai.utils.transform_rate(train_x.reshape(train_x.shape[0], -1) / 255.0, Ns, max_rate_spikes)
test_x = ai.utils.transform_rate(test_x.reshape(test_x.shape[0], -1) / 255.0, Ns, max_rate_spikes)
train_y = tf.keras.utils.to_categorical(train_y) # convert the labels to one-hot representation
test_y = tf.keras.utils.to_categorical(test_y)

# %% [markdown]
# This is now a temporal dataset comprising 60000 images of 28x28=784, coded over `Ns` timesteps:

# %%
train_x.shape

# %% [markdown]
# Let's visualize the spikes for a few inputs starting from input line 180:

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
plt.scatter(*np.where(train_x[0,:,180:190]), marker='|')
plt.title('Spike trains')
plt.xlabel('Timestep')
plt.xlim([-0.5,20.5])
plt.ylabel('Input index')

# %% [markdown]
# The spiking rate corresponds to the grayscale value of the pixels, so the brighter the pixel, the higher the number of spikes. Instead of spike trains, we can also plot 2D images for each timestep:

# %%
steps = 5 #number of timesteps to plot
plt.matshow(np.hstack(train_x[0,0:steps,:].reshape([steps,28,28])), cmap='binary')
plt.axis('off')

# %% [markdown]
# ## Build the model

# %% [markdown]
# Let's process the inputs using standard LSTMs and also SNNs, with two hidden layers with 250 units each:

# %%
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.InputLayer(input_shape=[None, 28*28]))
lstm_model.add(tf.keras.layers.LSTM(250, return_sequences=True))
lstm_model.add(tf.keras.layers.LSTM(250, return_sequences=True))
lstm_model.add(tf.keras.layers.LSTM(10, return_sequences=True))
lstm_model.add(tf.keras.layers.GlobalAveragePooling1D())  # calculate averaged spiking rate

# %% [markdown]
# Using SNUs instead of LSTMs involves changing the type of the layer, optionally setting custom configuration (here `decay` and `g` function), that can be shortened using `config` dictionary:

# %%
# Full syntax:
#  model.add(aitf.layers.SNU(250, decay=0.9, g=aitf.activations.leaky_rel, return_sequences=True))
# Shortened syntax with config dictionary:
#  model.add(aitf.layers.SNU(250, **config, return_sequences=True))
config = {'decay': 0.9, 'g': aitf.activations.leaky_rel}

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[None, 28*28]))
model.add(aitf.layers.SNU(250, **config, return_sequences=True))
model.add(aitf.layers.SNU(250, **config, return_sequences=True))
model.add(aitf.layers.SNU(10, **config, return_sequences=True))
model.add(tf.keras.layers.GlobalAveragePooling1D())  # calculate averaged spiking rate

# %% [markdown]
# ## Train

# %% [markdown]
# We train LSTM and SNN for a few epochs only, to quickly run this code also on CPU, but consider running for more epochs to get better accuracy:

# %%
epochs = 3

# %%
# This will take ~20 minutes on CPU
time_start = time.time()
lstm_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5), loss='mse', metrics=['accuracy'])
lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=15)
loss, acc = lstm_model.evaluate(test_x, test_y)
print("LSTM Loss {}, Accuracy {}".format(loss, acc))
print('Finished. Total time: {0:.1f} [s]'.format(time.time() - time_start))

# %%
# This will take ~5 minutes on CPU
time_start = time.time()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5), loss='mse', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=epochs, batch_size=15)
loss, acc = model.evaluate(test_x, test_y)
print("SNN Loss {}, Accuracy {}".format(loss, acc))
print('Finished. Total time: {0:.1f} [s]'.format(time.time() - time_start))

# %% [markdown]
# Comparing LSTM vs. SNN model:
#
# * SNN is faster to train and execute by ~4x
# * SNN achieves higher accuracy
