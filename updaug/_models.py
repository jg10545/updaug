# -*- coding: utf-8 -*-
import tensorflow as tf

from updaug._layers import InstanceNorm, ResidualBlock, AdaptiveInstanceNormalization


def _build_encoder(num_channels=3):
    print("instance norm removed")
    inpt = tf.keras.layers.Input((None, None, num_channels))
    # 32 layer
    net = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same")(inpt)
    #net = InstanceNorm()(net)
    #net = tf.keras.layers.LayerNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)

    # 64 layer
    net = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(net)
    #net = InstanceNorm()(net)
    #net = tf.keras.layers.LayerNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)

    # 128 layers
    net = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(net)
    #net = InstanceNorm()(net)
    #net = tf.keras.layers.LayerNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)

    # residual blocks
    for _ in range(3):
        net = ResidualBlock(128)(net)
    return tf.keras.Model(inpt, net)

def _build_decoder(num_channels=3):
    inpt = tf.keras.layers.Input((None, None, 128))

    net = tf.keras.layers.Conv2D(64, 3, padding="same")(inpt)
    net = tf.keras.layers.LayerNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.UpSampling2D(size=2)(net)

    net = tf.keras.layers.Conv2DTranspose(32, 3,  padding="same")(net)
    net = tf.keras.layers.LayerNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.UpSampling2D(size=2)(net)

    net = tf.keras.layers.Conv2D(num_channels, 3, padding="same", activation="sigmoid")(net)
    return tf.keras.Model(inpt, net)


def build_generator(num_domains, num_channels=3):
    """
    
    """
    encoder = _build_encoder(num_channels)
    decoder = _build_decoder(num_channels)
    anorm = AdaptiveInstanceNormalization(128, num_domains)
    
    
    inpt = tf.keras.layers.Input((None, None, num_channels))
    domain_inpt = tf.keras.layers.Input((num_domains,), dtype=tf.int64)
    net = encoder(inpt)
    #net = anorm(net, domain_inpt)
    print("SHUNTING OUT ADAPTIVE INSTANCE NORM LAYER")
    net = tf.keras.layers.Add()([net, 0*anorm(net, domain_inpt)])
    net = decoder(net)
    return tf.keras.Model([inpt, domain_inpt], net)



def build_discriminator(num_domains, num_channels=3):
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = tf.keras.layers.Conv2D(32, 4, strides=2)(inpt)
    net = tf.keras.layers.Activation(tf.nn.leaky_relu)(net)
    for k in [64, 128, 256]:
        net = tf.keras.layers.Conv2D(k, 4, strides=2)(net)
        net = InstanceNorm()(net)
        net = tf.keras.layers.Activation(tf.nn.leaky_relu)(net)
    
    net = tf.keras.layers.Conv2D(num_domains, 3)(net)
    # the global pooling layer isn't mentioned in the paper- without it
    # you get a rank-4 output though
    #net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Activation("sigmoid")(net)
    return tf.keras.Model(inpt, net)




