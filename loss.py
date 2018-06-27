# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def KL_loss(mu, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def NLLNormal(pred, target):
    c = -0.5 * tf.log(2 * np.pi)
    multiplier = 1.0 / (2.0 * 1)
    tmp = tf.square(pred - target)
    tmp *= -multiplier
    tmp += c
    return tmp

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, E, opt, training_set, minibatch_size, reals,
    cond_weight = 1.0): # Weight of the conditioning term.

    z_out, mu_out, sd_out, eps_out = fp32(E.get_output_for(reals, is_training=True))
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(z_out, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))

    with tf.name_scope('ReconPenalty'):
        # Recon loss
        img_loss = tf.reduce_mean(tf.reduce_sum(NLLNormal(fake_images_out, reals), [1,2,3]))
        # KL-divergence
        latent_loss = KL_loss(mu_out, sd_out)

    zp = tf.random_normal(tf.shape(z_out), 0.0, 1.0)
    fake_images_out_zp = G.get_output_for(zp, labels, is_training=True)
    fake_scores_out_zp, fake_labels_out_zp = fp32(D.get_output_for(fake_images_out_zp, is_training=True))

    img_size = D.input_shapes[0][1:]

    loss = - fake_scores_out - fake_scores_out_zp - img_loss/(img_size[0]*img_size[1]*img_size[2])

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight

    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, E, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    zp = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    z_out, mu_out, sd_out, eps_out = fp32(E.get_output_for(reals, is_training=True))

    fake_images_out = G.get_output_for(z_out, labels, is_training=True)
    fake_images_out_zp = G.get_output_for(zp, labels, is_training=True)

    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    fake_scores_out_zp, fake_labels_out_zp = fp32(D.get_output_for(fake_images_out_zp, is_training=True))

    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    fake_scores_out_zp = tfutil.autosummary('Loss/fake_prior_scores', fake_scores_out_zp)
    loss = (fake_scores_out + fake_scores_out_zp) - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).
for _ in range(D_repeats):
def E_recon(G, D, E, opt, training_set, minibatch_size, reals,
    cond_weight = 1.0): # Weight of the conditioning term.

    z_out, mu_out, sd_out, eps_out = fp32(E.get_output_for(reals, is_training=True))
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(z_out, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))

    with tf.name_scope('ReconPenalty'):
        # Recon loss
        img_loss = tf.nn.l2_loss(reals - recon_images_out)
        # KL-divergence
        latent_loss = KL_loss(mu_out, sd_out)

    img_size = D.input_shapes[0][1:]

    loss = latent_loss/(tf.shape(z_out)[0]*tf.shape(z_out)[1]) - img_loss/(img_size[0]*img_size[1]*img_size[2])
    #
    #
    # z_out, mu_out, sd_out, eps_out = fp32(E.get_output_for(reals, is_training=True))
    # labels = training_set.get_random_labels_tf(minibatch_size)
    # # For reconstruction use only the mean
    # recon_images_out = fp32(G.get_output_for(z_out, labels, is_training=True))
    #
    # with tf.name_scope('ReconPenalty'):
    #     img_loss = tf.nn.l2_loss(reals - recon_images_out)
    #     # img_loss = tf.reduce_sum(tf.squared_difference(recon_images_out, reals), 1)
    #     # latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd_out - tf.square(mu_out) - tf.exp(2.0 * sd_out), 1)
    #     latent_loss = latent_loss = KL_loss(mu_out, sd_out)
    #
    # loss = img_loss + latent_loss

    return loss
