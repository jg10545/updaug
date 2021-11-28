# -*- coding: utf-8 -*-
import tensorflow as tf

from updaug._loss import _l1_loss, _edge_loss


def _build_generator_training_step(gen, disc, opt, lam1=1, lam2=10, lam3=10, lam4=100):
    #@tf.function
    def trainstep(a, adom, b, bdom):
        # a, b: [batch_size, H, W, num_channels]
        # adom, bdom: [batch_size, num_domains]
        with tf.GradientTape() as tape:
            # map each training point back to its original domain 
            # e.g. run as an autoencoder
            a_prime = gen([a, adom], training=True)
            b_prime = gen([b, bdom], training=True)
            # map each training point to a different domain
            # e.g. run in domain-transfer mode
            a_fake = gen([a, bdom], training=True)
            b_fake = gen([b, adom], training=True)
            # map each fake image back to the original domain
            # e.g. run like CycleGAN
            a_prime_prime = gen([a_fake, adom], training=True)
            b_prime_prime = gen([b_fake, bdom], training=True)
            
            # run fake data through discriminator
            disc_a_fake = tf.reduce_sum(disc(a_fake)*bdom, -1)
            disc_b_fake = tf.reduce_sum(disc(b_fake)*adom, -1)
            
            # -------------------- LOSSES --------------------
            # adversarial loss on fake images
            L_G_adv = tf.reduce_mean(-1*tf.math.log(disc_a_fake)) + \
                        tf.reduce_mean(-1*tf.math.log(disc_b_fake))
            # reconstruction loss from fake image back to original domain
            L_cross = _l1_loss(a, a_prime_prime) + _l1_loss(b, b_prime_prime)
            # autoencoder reconstruction loss
            L_self = _l1_loss(a, a_prime) + _l1_loss(b, b_prime)
            # edge loss
            L_edge = _edge_loss(a, a_fake) + _edge_loss(b, b_fake)
            
            total_loss = lam1*L_G_adv + lam2*L_cross + lam3*L_self + lam4*L_edge
            
            
        grads = tape.gradient(total_loss, gen.trainable_variables)
        opt.apply_gradients(zip(grads, gen.trainable_variables))
        return {"loss":total_loss, "generator_adv_loss":L_G_adv, "cross_loss":L_cross,
                   "self_loss":L_self, "edge_loss":L_edge}
    return trainstep


def _build_discriminator_training_step(gen, disc, opt):
    def trainstep(a, adom, b, bdom):
        # a, b: [batch_size, H, W, num_channels]
        # adom, bdom: [batch_size, num_domains]
        # map each training point to a different domain
        # e.g. run in domain-transfer mode
        a_fake = gen([a, bdom])
        b_fake = gen([b, adom])
        with tf.GradientTape() as tape:
            # run real data through discriminator
            disc_a = tf.reduce_sum(disc(a, training=True)*adom, -1)
            disc_b = tf.reduce_sum(disc(b, training=True)*bdom, -1)
            # run fake data through discriminator
            disc_a_fake = tf.reduce_sum(disc(a_fake, training=True)*bdom, -1)
            disc_b_fake = tf.reduce_sum(disc(b_fake, training=True)*adom, -1)
            
            # -------------------- LOSSES --------------------
            loss = -1*(tf.reduce_mean(tf.math.log(1-disc_a_fake)) + \
                    tf.reduce_mean(tf.math.log(1-disc_b_fake)) + \
                    tf.reduce_mean(tf.math.log(disc_a)) + \
                    tf.reduce_mean(tf.math.log(disc_b)))
            
        grads = tape.gradient(loss, disc.trainable_variables)
        opt.apply_gradients(zip(grads, disc.trainable_variables))
        return {"disc_loss":loss}
    return trainstep