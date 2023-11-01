# Neuro-inspired AI Toolkit

**Navigate to the Neuro-inspired AI Toolkit website https://ibm.github.io/neuroaikit/ for comprehensive documentation.**

This toolkit facilitates use of biologically-inspired AI in TensorFlow 2.x.
It enables to build SNU-based networks. For more deatail, 
see the toolkit website or the following paper:
* _S. Woźniak, A. Pantazi, T. Bohnstingl, and E. Eleftheriou_, ["Deep learning incorporating biologically inspired neural dynamics and in-memory computing"](http://www.nature.com/articles/s42256-020-0187-0), Nat Mach Intell, vol. 2, no. 6, pp. 325–336, Jun. 2020, doi: 10.1038/s42256-020-0187-0.

*Note*: Code for TensorFlow version 1.x is available as the Supplementary Material of [paper](http://www.nature.com/articles/s42256-020-0187-0).

## Installation
```
git clone https://github.com/IBM/neuroaikit.git
pip install --editable neuroaikit
```
*Note*: Editable install enables you to modify the toolkit code and thus to contribute your ideas. 

*Note*: Currently the installer does not install the requirements yet. Please see `requirements.txt` for some of the requirements.

To verify the installation:
```
import neuroaikit.tf as aitf
neuron = aitf.layers.SNU(1)
```
Please visit the toolkit website for more detail.

## Removal
```
pip uninstall neuroaikit
```

## Upgrade to TF 2.9 Changes and Diffs


### 1. Import Statements:

- In the original code, we maintain imports from the custom `neuroaikit.tf` module, which contains the specialized SNU (Spiking Neural Unit) layers and activations specific to our project.
- In TensorFlow 2.9, we now use standard TensorFlow modules, specifically `from tensorflow.keras.layers import RNN, Layer`, for SNU-related layers and models. This change is necessary to align with TensorFlow 2.9's conventions and ensure compatibility with SNU layers.

### 2. Class Inheritance:

- In the original code, we always inherit from `tf.keras.layers.Layer` for consistency with our custom SNU layers.
- In the modified TensorFlow 2.9 code, we use the `Layer` class from TensorFlow for class inheritance.

### 3. Instantiation of Layers:

- In the original code, we directly instantiate `tf.keras.layers.RNN` and use the `add_weight` method for defining layer variables.
- In the modified code for TensorFlow 2.9, we create instances of `RNN`, and the definition of layer variables is moved to the `build` method within the custom cell classes. This adjustment adheres to the TensorFlow 2.9 conventions.

### 4. `tf.reduce_max` Function:

- In the `SNULICell` class, the use of `tf.reduce_max` is introduced to replicate the lateral inhibition logic present in the original code. This adjustment is made to ensure that lateral inhibition functionality remains consistent in TensorFlow 2.9.

Overall, these modifications are made to accommodate TensorFlow 2.9's changes while retaining the integrity and compatibility of our custom `neuroaikit.tf` module, which contains SNU-related layers and activations.
