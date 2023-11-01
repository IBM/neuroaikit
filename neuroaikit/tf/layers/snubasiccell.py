# """Contains basic SNU cell definition.
# """

# from neuroaikit.tf.activations import *


# class SNUBasicCell(tf.keras.layers.Layer):
#     """This is a basic SNU cell.

#     :param units: Number of units to create in the layer
#     :param decay: Membrane potential decay multiplier, defaults to 0.8,
#         i.e. 0.8 of the previous membrane potential is retained
#     :param activation: Activation function, defaults to step_function. See TF_Misc.Activations.
#     :param g: Internal state activation function that optionally constraints the state,
#         defaults to tf.identity (no constraint)
#     :param recurrent: bool, defaults to False. If True, SNU includes recurrent connections inside entire layer.
#     """

#     def __init__(self, units, decay=0.8, activation=step_function, g=tf.identity, recurrent=False, **kwargs):
#         """Constructor method"""
#         super(self.__class__, self).__init__(**kwargs)
#         self.units = units
#         self.state_size = (units, units)
#         self.decay = decay
#         self.activation = activation
#         self.g = g
#         self.recurrent = recurrent

#     def build(self, input_shape):
#         """Overriding build method that creates the variables

#         :param input_shape: Shape of the input
#         """
#         self.kernel = self.add_weight(shape=(input_shape[-1], self.units), name='kernel')
#         if self.recurrent:
#             self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='recurrent_kernel')
#         self.bias = self.add_weight(shape=(self.units,), initializer='ones', name='bias')
#         self.built = True

#     def call(self, inputs, states):
#         """Overriding call method that defines the cell dynamics' graph

#         :param inputs: Tensor representing the input in particular timestep
#         :param states: Tuple with previous state values
#         :return: Output values, State values.
#         """
#         (out_prev, Vm_prev) = states
#         Vm = Vm_prev * (1.0 - out_prev)
#         Vm = Vm * self.decay
#         Vm = Vm + tf.matmul(inputs, self.kernel)
#         if self.recurrent:
#             Vm = Vm + tf.matmul(out_prev, self.recurrent_kernel)
#         Vm = self.g(Vm)
#         overVth = Vm - self.bias
#         out = self.activation(overVth)
#         return out, (out, Vm)



#==============================================
"""Contains basic SNU cell definition in tf 2.9.
"""
from neuroaikit.tf.activations import *

class SNUBasicCell(Layer):
    def __init__(self, units, decay=0.8, activation=step_function, g=tf.identity, recurrent=False, **kwargs):
        super(SNUBasicCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = (units, units)
        self.decay = decay
        self.activation = activation
        self.g = g
        self.recurrent = recurrent

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), name='kernel')
        if self.recurrent:
            self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='ones', name='bias')
        self.built = True

    def call(self, inputs, states):
        (out_prev, Vm_prev) = states
        Vm = Vm_prev * (1.0 - out_prev)
        Vm = Vm * self.decay
        Vm = Vm + tf.matmul(inputs, self.kernel)
        if self.recurrent:
            Vm = Vm + tf.matmul(out_prev, self.recurrent_kernel)
        Vm = self.g(Vm)
        overVth = Vm - self.bias
        out = self.activation(overVth)
        return out, (out, Vm)