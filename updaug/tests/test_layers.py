# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from updaug._layers import InstanceNorm, ResidualBlock, AdaptiveInstanceNormalization


def test_instancenorm():
    N = 13
    H = 5
    W = 7
    C = 3
    
    x = np.random.normal(0, 1, (N,H,W,C)).astype(np.float32)
    y = InstanceNorm()(x)
    means = tf.reduce_mean(y, axis=[1,2]).numpy()
    assert np.max(np.abs(means)) < 1e-5


def test_residualblock():
    N = 13
    H = 5
    W = 7
    C = 3
    
    x = np.random.normal(0, 1, (N,H,W,C)).astype(np.float32)
    block = ResidualBlock(C)
    y = block(x)
    
    assert x.shape == y.shape
    
def test_adaptiveinstancenorm():
    N = 13
    H = 5
    W = 7
    C = 3
    num_domains=4
    
    norm = AdaptiveInstanceNormalization(C, num_domains)
    assert norm.gamma.shape == (num_domains, 1, 1, C)
    
    
    x = np.random.normal(0, 1, (N,H,W,C)).astype(np.float32)
    domain = tf.one_hot(np.random.randint(0, num_domains, size=N), num_domains)
    y = norm(x, domain)
    assert x.shape == y.shape    
