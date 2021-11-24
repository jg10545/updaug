# -*- coding: utf-8 -*-
import tensorflow as tf

def _l1_loss(a,b):
    return tf.reduce_mean(tf.abs(a-b))

def _edge_loss(a,b):
    a = tf.reduce_mean(a, axis=-1, keepdims=True)
    b = tf.reduce_mean(b, axis=-1, keepdims=True)
    sob_a = tf.image.sobel_edges(a)
    sob_b = tf.image.sobel_edges(b)
    return _l1_loss(sob_a, sob_b)
    


