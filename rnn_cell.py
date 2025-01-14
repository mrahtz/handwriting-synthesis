from collections import namedtuple
import typing

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

from tf_utils import dense_layer, shape


class LSTMAttentionCellState(typing.NamedTuple):
    h1: tf.Tensor
    c1: tf.Tensor
    h2: tf.Tensor
    c2: tf.Tensor
    h3: tf.Tensor
    c3: tf.Tensor
    alpha: tf.Tensor
    beta: tf.Tensor
    kappa: tf.Tensor
    w: tf.Tensor
    phi: tf.Tensor


class LSTMAttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        attention_values,
        attention_values_lengths,
        num_output_mixture_components,
        bias,
        reuse=None,
    ):
        self.reuse = reuse
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values
        self.attention_values_lengths = attention_values_lengths
        self.window_size = shape(self.attention_values, 2)
        self.char_len = tf.shape(attention_values)[1]
        self.batch_size = tf.shape(attention_values)[0]
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6 * self.num_output_mixture_components + 1
        self.bias = bias

    @property
    def state_size(self):
        return LSTMAttentionCellState(
            h1=self.lstm_size,
            c1=self.lstm_size,
            h2=self.lstm_size,
            c2=self.lstm_size,
            h3=self.lstm_size,
            c3=self.lstm_size,
            alpha=self.num_attn_mixture_components,
            beta=self.num_attn_mixture_components,
            kappa=self.num_attn_mixture_components,
            w=self.window_size,
            phi=self.char_len,
        )

    @property
    def output_size(self):
        return self.lstm_size

    def zero_state(self, batch_size, dtype):
        return LSTMAttentionCellState(
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.window_size]),
            tf.zeros([batch_size, self.char_len]),
        )

    def __call__(self, inputs, state, scope=None):
        with tf.compat.v1.variable_scope(scope or type(self).__name__, reuse=tf.compat.v1.AUTO_REUSE):
            # lstm 1
            layer_1_input = tf.concat([state.w, inputs], axis=1)
            cell1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
            layer_1_output, layer_1_state = cell1(layer_1_input, state=(state.c1, state.h1))

            # attention
            attention_inputs = tf.concat([state.w, inputs, layer_1_output], axis=1)
            attention_params = dense_layer(
                attention_inputs,
                3 * self.num_attn_mixture_components,
                scope="attention",
            )
            alpha, beta, kappa = tf.split(tf.nn.softplus(attention_params), 3, axis=1)
            kappa = state.kappa + kappa / 25.0
            beta = tf.clip_by_value(beta, 0.01, np.inf)

            kappa_flat, alpha_flat, beta_flat = kappa, alpha, beta
            kappa, alpha, beta = (
                tf.expand_dims(kappa, 2),
                tf.expand_dims(alpha, 2),
                tf.expand_dims(beta, 2),
            )

            enum = tf.reshape(tf.range(self.char_len), (1, 1, self.char_len))
            u = tf.cast(
                tf.tile(enum, (self.batch_size, self.num_attn_mixture_components, 1)),
                tf.float32,
            )
            phi_flat = tf.reduce_sum(
                alpha * tf.exp(-tf.square(kappa - u) / beta), axis=1
            )

            phi = tf.expand_dims(phi_flat, 2)
            sequence_mask = tf.cast(
                tf.sequence_mask(self.attention_values_lengths, maxlen=self.char_len),
                tf.float32,
            )
            sequence_mask = tf.expand_dims(sequence_mask, 2)
            w = tf.reduce_sum(phi * self.attention_values * sequence_mask, axis=1)

            # lstm 2
            layer_2_input = tf.concat([inputs, layer_1_output, w], axis=1)
            cell2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
            layer_2_output, layer_2_state = cell2(layer_2_input, state=(state.c2, state.h2))

            # lstm 3
            layer_3_input = tf.concat([inputs, layer_2_output, w], axis=1)
            cell3 = tf.contrib.rnn.LSTMCell(self.lstm_size)
            layer_3_out, layer_3_state = cell3(layer_3_input, state=(state.c3, state.h3))

            new_state = LSTMAttentionCellState(
                layer_1_state.h,
                layer_1_state.c,
                layer_2_state.h,
                layer_2_state.c,
                layer_3_state.h,
                layer_3_state.c,
                alpha_flat,
                beta_flat,
                kappa_flat,
                w,
                phi_flat,
            )

            return layer_3_out, new_state

    def output_function(self, state):
        params = dense_layer(
            state.h3, self.output_units, scope="gmm", reuse=tf.compat.v1.AUTO_REUSE
        )
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)
        mu1, mu2 = tf.split(mus, 2, axis=1)
        mus = tf.stack([mu1, mu2], axis=2)
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1)

        covar_matrix = [
            tf.square(sigma1),
            rhos * sigma1 * sigma2,
            rhos * sigma1 * sigma2,
            tf.square(sigma2),
        ]
        covar_matrix = tf.stack(covar_matrix, axis=2)
        covar_matrix = tf.reshape(
            covar_matrix, (self.batch_size, self.num_output_mixture_components, 2, 2)
        )

        mvn = tfd.MultivariateNormalFullCovariance(
            loc=mus, covariance_matrix=covar_matrix
        )
        b = tfd.Bernoulli(probs=es)
        c = tfd.Categorical(probs=pis)

        sampled_e = b.sample()
        sampled_coords = mvn.sample()
        sampled_idx = c.sample()

        idx = tf.stack([tf.range(self.batch_size), sampled_idx], axis=1)
        coords = tf.gather_nd(sampled_coords, idx)
        return tf.concat([coords, tf.cast(sampled_e, tf.float32)], axis=1)

    def termination_condition(self, state: LSTMAttentionCellState):
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), tf.int32)
        final_char = char_idx >= self.attention_values_lengths - 1
        past_final_char = char_idx >= self.attention_values_lengths
        output = self.output_function(state)
        es = tf.cast(output[:, 2], tf.int32)
        is_eos = tf.equal(es, tf.ones_like(es))
        return tf.math.logical_or(tf.math.logical_and(final_char, is_eos), past_final_char)

    def _parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            gmm_params,
            [
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1,
            ],
            axis=-1,
        )
        pis = pis * (1 + tf.expand_dims(self.bias, 1))
        sigmas = sigmas - tf.expand_dims(self.bias, 1)

        pis = tf.nn.softmax(pis, axis=-1)
        pis = tf.where(pis < 0.01, tf.zeros_like(pis), pis)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        es = tf.where(es < 0.01, tf.zeros_like(es), es)

        return pis, mus, sigmas, rhos, es
