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


