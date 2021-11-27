# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class InstanceNorm(tf.keras.layers.Layer):
    """
    Following equation 3 of "Instance Normalization: The Missing Ingredient for Fast Stylization" by
    Ulyanov et al. https://arxiv.org/pdf/1607.08022.pdf
    
    """
    def __init__(self, scale=True, shift=True):
        super(InstanceNorm, self).__init__()

        self._scale = scale
        self._shift = shift
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        num_channels = input_shape[-1]
        
        if self._scale:
            self.gamma = self.add_weight(name="gamma",
                                         shape = (1, 1, 1, num_channels),
                                         initializer=tf.initializers.ones,
                                         trainable=True)
        else:
            self.gamma = 1
        if self._shift:
            self.beta = self.add_weight(name="beta",
                                         shape = (1, 1, 1, num_channels),
                                         initializer=tf.initializers.zeros,
                                         trainable=True)
        else:
            self.beta = 0
            

    def call(self, inputs):
        assert len(inputs.shape) == 4, "instance normalization assumes input is a rank-4 tensor (N,H,W,C)"
        eps = tf.keras.backend.epsilon()
        
        mu = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        sigsq = tf.reduce_mean((inputs-mu)**2, axis=[1,2], keepdims=True)
        x = (inputs-mu)/tf.math.sqrt(sigsq + eps)
        return x*self.gamma + self.beta
    
    
class ResidualBlock(tf.keras.layers.Layer):
    """

    
    """
    def __init__(self, k, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        
        self.sublayers = [
            tf.keras.layers.Conv2D(k, kernel_size, padding="same"),
            InstanceNorm(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(k, kernel_size, padding="same"),
            InstanceNorm(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Add()
        ]
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        net = inputs
        for s in self.sublayers[:-1]:
            net = s(net)
        
        return self.sublayers[-1]([net, inputs])
    
    
class AdaptiveInstanceNormalization(tf.keras.layers.Layer):
    """

    
    """
    def __init__(self, k, num_domains):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.k = k
        self.num_domains = num_domains

        self.gamma = tf.constant(np.random.uniform(0, 1, (num_domains,1,1,k)).astype(np.float32))
        self.beta = tf.constant(np.random.uniform(0, 1, (num_domains,1,1,k)).astype(np.float32))
        
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, domain):
        domain = tf.cast(domain, tf.float32)
        gamma = tf.einsum('ij,jklm->iklm', domain, self.gamma) # [batch_size, 1, 1, k]
        beta = tf.einsum('ij,jklm->iklm', domain, self.beta) # [batch_size, 1, 1, k]
        output = gamma*InstanceNorm()(inputs) + beta
        return output