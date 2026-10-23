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

*Note*: Currently the `requirements.txt` file with for automatic installation of dependencies was removed due to frequent version changes. Depending on what you need to run, following packages may need to be installed to your environment via `pip install`:

* **TensorFlow** or **PyTorch** (depending on the desired backend)
* **Pandas**
* **Matplotlib**
* **tqdm**
* **scikit-learn**
* **Datasets** — *Required by the NER example.*

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


