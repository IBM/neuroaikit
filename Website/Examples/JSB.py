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
# | [![View in Neuro-inspired AI Toolkit Documentation](../_static/toolkit_32px.png)](https://ibm.github.io/neuroaikit/Examples/JSB.html) |  [![View source on GitHub](../_static/github_32px.png)](https://github.com/IBM/neuroaikit/blob/main/Website/Examples/JSB.ipynb) |
# | - | - |
# | View in Neuro-inspired AI Toolkit Documentation | View source on GitHub |

# %% [markdown]
# # Music prediction (JSB)
#
# This tutorial shows how to:
#
# 1. Interpret musical notes as a temporal stream of spikes, where the location in time matters (temporal coding).
# 1. Build a spiking network that predicts the next musical chord based on the chords seen so far.
# 1. Generate and play an entire song based on one-step predictions.

# %% [markdown]
# ## Musical notes as spikes
#
# A presence of a particular sound at a particular time location is what creates music, and what the common musical notation conveys through musical sheets. Alternatively, we can view the information on the presence of sounds or musical notes as spikes.
#
# A typical piano keyboard has 88 keys [(Wikimedia - Creative Commons)](https://commons.wikimedia.org/wiki/File:Piano_Keyboard_Diagram.svg):
#
# ![image.png](attachment:image.png)
#
# Therefore, for each timestep of a muscial piece, we can encode the information of the musical notes played by sending spikes to an input layer with 88 inputs. For illustration, let's use the JSB dataset from N. Boulanger-Lewandowski, et al., ICML 2012, that comprises chorales written by Johann Sebastian Bach. It can be loaded as follows:

# %%
import neuroaikit.dataset.datasets as aid
x = aid.JSB()

# %% [markdown]
# The tuple `x` contains the data split into: 229 train, 76 validation, and 77 test chorales: 

# %%
len(x), len(x[0]), len(x[1]), len(x[2])

# %% [markdown]
# Each music piece is a sequence of 88-dimensional vectors. The first training chorales is 129 time steps long:

# %%
x[0][0].shape

# %% [markdown]
# We can visualize the first 20 time steps of this chorales as spikes. Note that the most commonly used notes are in the center of the keyboard and the y axis is automatically cropped:

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(*np.where(x[0][0][0:20,:]), marker='|')
plt.title('Spike trains')
plt.xlabel('Timestep')
plt.xlim([-0.5,20.5])
#plt.ylim([0,88])
plt.ylabel('Input index')

# %% [markdown]
# ## Train a network

# %% [markdown]
# Based on the input spikes observed at 88 inputs, we can build a network that would output notes' predictions. In such case, the last layer should also have 88 outputs, as illustrated below:

# %% [markdown]
# ![image-3.png](attachment:image-3.png)

# %% [markdown]
# Let's import the required modules and build the network:

# %%
import neuroaikit as ai
import neuroaikit.tf as aitf
import tensorflow as tf

# %% [markdown]
# The network includes one hidden layer of spiking neurons and a dense output layer without activation that outputs the raw logits:

# %%
config = {'decay': 0.8, 'g': aitf.activations.leaky_rel}

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[None, 88]))
model.add(aitf.layers.SNU(150, **config, return_sequences=True))
model.add(tf.keras.layers.Dense(88)) #output logits

# %% [markdown]
# Then, we use the binary cross-entropy loss, that is appropriate for training our model that outputs a series of separate per-note probabilities. The loss is configured to operate on raw logits:

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[])

# %% [markdown]
# For training, we need to have pairs of input and output vectors. We create a dataset generator, whose final step is to copy the input vectors twice with a shift of one timestep:

# %%
ds = tf.data.Dataset.from_generator(lambda: x[0], tf.int32, output_shapes=[None,None])
ds = ds.map(lambda x: (tf.expand_dims(x[0:-1,:],0), tf.expand_dims(x[1:,:],0)))

# %% [markdown]
# Thus, the dataset generator returns pairs of sequences with 88 features:

# %%
example = ds.as_numpy_iterator()
example_x, example_y = example.next()
example_x.shape, example_y.shape

# %% [markdown]
# Let's train the model for 40 epochs:

# %%
import time
time_start = time.time()
model.fit(ds, epochs=40)
print('Finished. Total time: {0:.1f} [s]'.format(time.time() - time_start))

# %% [markdown]
# We see that the loss keeps on decreasing, which means that the model is improving. 

# %% [markdown]
# The accuracy of notes' predictions for neural networks is commonly assesed using the loss value of the avergage negative log-likelihood (the lower the better). SNU-based networks predict the notes quite well, as illustrated in the figure below taken from [S.Woźniak, et al., 2020](https://www.nature.com/articles/s42256-020-0187-0), where more detailed explanations are provided. Please note that the loss reported in the figure involves an averaging approach that is different from the one used in the example above, and thus the loss values are not directly comparable.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ## Play one-step predictions

# %% [markdown]
# It's interesting to hear what are the predictions of the network. We can take the output logits and threshold them to obtain the predicted notes. Then, we can play these notes.

# %% [markdown]
# To play the notes, we will make use of the fact that a note is charaterized by frequency:
#
# $f(n) = 2^{\frac{n-49}{12}} \times 440$ [Hz] - see https://en.wikipedia.org/wiki/Piano_key_frequencies
#
# so that we can write a simple sine-wave-based synthesizer to play the music:

# %%
from IPython.display import Audio

def play(song, note=0.25, rate=44100):
    t = np.linspace(0, note, int(rate * note))
    data = []
    for s in range(song.shape[0]):
        val = np.zeros_like(t)
        for n in np.where(song[s,:] == 1)[0]:
            f = 2**((n-49)/12) * 440
            #val += np.sin(2 * np.pi * f * t) # "sine" sound
            val += np.clip(2*np.sin(2 * np.pi * f * t),0,1) #"organ" sound
        val *= np.clip(30.0*np.sin(np.pi * t/note), 0.0, 1.0) #smoothen to avoid 'cracks'
        data.append(val)
    return Audio(np.hstack(data), rate=rate, autoplay=True)


# %% [markdown]
# Let's calculate the model next-step prediction's for the 10th song and play them:

# %%
logits = model.predict(np.expand_dims(x[0][10],0))
notes = (logits > 0)*1
play(notes[0,0:30,:])

# %% [markdown]
# Here is the original:

# %%
play(x[0][10][0:30,:])

# %% [markdown]
# For comparison, predictions from an untrained model:

# %%
config = {'decay': 0.8, 'g': aitf.activations.leaky_rel}
untrained_model = tf.keras.Sequential()
untrained_model.add(tf.keras.layers.InputLayer(input_shape=[None, 88]))
untrained_model.add(aitf.layers.SNU(150, **config, return_sequences=True))
untrained_model.add(tf.keras.layers.Dense(88)) #output logits

logits = untrained_model.predict(np.expand_dims(x[0][10],0))
notes = (logits > 0)*1
play(notes[0,0:30,:])

# %% [markdown]
# The original sounds the best, which shows that music prediction is challenging in general. In comparison to an untrained model, the trained one captures some musical concepts.
