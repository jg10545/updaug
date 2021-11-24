# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from updaug._models import build_generator, build_discriminator


def test_build_generator():
    num_domains = 5
    generator = build_generator(num_domains)
    x = np.random.normal(0, 1, (1, 128, 128,3)).astype(np.float32)
    d = np.zeros((1,num_domains)).astype(np.int64)
    d[0,0] = 1
    y = generator([x, d])
    assert x.shape == y.shape
    
    
    
def test_build_discriminator():
    num_domains = 7
    disc = build_discriminator(num_domains)
    x = np.random.normal(0, 1, (1, 128, 128,3)).astype(np.float32)
    y = disc(x)
    assert y.shape == (1, num_domains)
