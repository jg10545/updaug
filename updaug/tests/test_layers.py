# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from updaug._layers import InstanceNorm


def test_instancenorm():
    N = 13
    H = 5
    W = 7
    C = 3
    
    x = np.random.normal(0, 1, (N,H,W,C)).astype(np.float32)
    y = InstanceNorm()(x)
    means = tf.reduce_mean(y, axis=[1,2]).numpy()
    assert np.max(np.abs(means)) < 1e-5
