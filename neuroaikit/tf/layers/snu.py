# """Contains generic SNU function for creating SNU layers.
# """

# from neuroaikit.tf.activations import *
# from .snubasiccell import SNUBasicCell
# from .snulicell import SNULICell

# def SNU(units, activation=step_function, decay=0.8, g=tf.identity, recurrent=False,
#         lateral_inhibition=False, #uses SNULICell
#         **args):
#     """This is a basic SNU layer.

#     :param units: Number of units to create in the layer
#     :param decay: Membrane potential decay multiplier, defaults to 0.8,
#         i.e. 0.8 of the previous membrane potential is retained
#     :param activation: Activation function, defaults to step_function. See TF_Misc.Activations.
#     :param g: Internal state activation function that optionally constraints the state,
#         defaults to tf.identity (no constraint)
#     :param recurrent: bool, defaults to False. If True, SNU includes recurrent connections inside entire layer.
#     :param lateral_inhibition: bool, defaults to False. If True, layer-wise lateral inhibition is used.
#     :param args: Additional arguments to the Keras layer constructor (e.g. name, trainable).
#     :return:
#     """
#     cell = SNUBasicCell
#     if lateral_inhibition:
#         cell = SNULICell
#     return tf.keras.layers.RNN(cell(units, activation=activation, decay=decay, g=g, recurrent=recurrent), **args)


#=================================================================
"""Contains generic SNU function for creating SNU layers in tf 2.9.
"""
import tensorflow as tf
from tensorflow.keras.layers import RNN
from neuroaikit.tf.activations import *
from .snubasiccell import SNUBasicCell
from .snulicell import SNULICell

def SNU(units, activation=step_function, decay=0.8, g=tf.identity, recurrent=False, lateral_inhibition=False, **kwargs):
    cell = SNUBasicCell
    if lateral_inhibition:
        cell = SNULICell
    return RNN(cell(units, activation=activation, decay=decay, g=g, recurrent=recurrent), **kwargs)
