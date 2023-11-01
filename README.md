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

## Upgrade to TF 2.9

Some differences between our original TensorFlow 2.5 code and the modified code for TensorFlow 2.9:

1. **Import Statements:**
   - In the original code, we have imports from the `neuroaikit.tf` module, which is not standard TensorFlow; these remain by default.
   - Additionally, one could optionally also try the standard modules when/iff necessary:
   - Specifically, `import tensorflow as tf` and `from tensorflow.keras.layers import RNN, Layer` could be used for generic (non-SNU) layers and models.

2. **Class Inheritance:**
   - In the original code, we inherit always from `tf.keras.layers.Layer`. In the modified tf 2.9 code, the `Layer` class from TensorFlow could be also used for class inheritance.

3. **Instantiation of Layers:**
   - In the original code,  we directly instantiated `tf.keras.layers.RNN` and using `add_weight` for layer variables. In the modified code, `RNN` is instantiated and `add_weight` is used within the `build` method of the custom cell classes.

4. **Reduce Max Function:**
   - In the `SNULICell` class, the use of `tf.reduce_max` was added to mimic the lateral inhibition logic you had in the original code.

Overall, the modifications involve changing imports, class inheritance, and adjusting the code to conform to TensorFlow 2.9 conventions and API changes.



