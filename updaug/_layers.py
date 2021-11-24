# -*- coding: utf-8 -*-
import tensorflow as tf


class InstanceNorm(tf.keras.layers.Layer):
    """
    Following equation 3 of "Instance Normalization: The Missing Ingredient for Fast Stylization" by
    Ulyanov et al. https://arxiv.org/pdf/1607.08022.pdf
    
    """
    def __init__(self):
        super(InstanceNorm, self).__init__()
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        assert len(inputs.shape) == 4, "instance normalization assumes input is a rank-4 tensor (N,H,W,C)"
        eps = tf.keras.backend.epsilon()
        
        mu = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        sigsq = tf.reduce_mean((inputs-mu)**2, axis=[1,2], keepdims=True)
        return (inputs-mu)/tf.math.sqrt(sigsq + eps)