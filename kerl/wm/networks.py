import functools
import operator
from typing import Tuple, Optional, NamedTuple

import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras import losses
from keras import optimizers
from keras.models import Model
from keras.layers import (
    Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, ZeroPadding2D,
    Cropping2D, Concatenate, LSTM, TimeDistributed, Activation, Input)

from kerl.a2c.agent import a2c_advantage_estimator
from kerl.common.agent import EpisodeSample
from kerl.common.models import StatefulNet
from kerl.common.utils import (
    calc_optimal_convnet_padding, ConvOp, ImagePadding,
    PopArtLayer, OneHotConverter)


class WorldModelVAE:
    """
    Variational autoencoder. Its purpose here is to take a bunch of screenshots
    from the environment and encode them into a compact
    low-dimension representation. The reverse transformation is also possible.
    It's a canonical implementation of paper
    [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    loosely following architecture described by the [World Models]
    (https://arxiv.org/abs/1803.10122) paper.
    """
    # The encoder's parameters. Decoder uses the same, just in reverse order
    # (applying transposed convolutions). To make this possible, the input
    # will also be padded.
    encoder_filters = [32, 64, 128, 128, 128]
    encoder_kernel_size = 4
    encoder_strides = 2

    def __init__(self, name: str, observation_shape: Tuple[int, int, int],
                 optimizer: Optional[optimizers.Optimizer],
                 latent_dim: int, vae_beta=1.0,
                 default_activation: str='selu'):
        self.latent_dim = latent_dim
        width, height, num_channels = observation_shape
        # To make convolutions exactly reversible, the original observation
        # gets padded the right number of zero-filled rows and columns
        ob_pad_width, adjusted_ob_size = (
            calc_optimal_convnet_padding(
                (width, height),
                ([ConvOp(kernel_size=self.encoder_kernel_size,
                         strides=self.encoder_strides)]
                 * len(self.encoder_filters))))
        adjusted_ob_shape = adjusted_ob_size + (num_channels,)
        self._encoder_model, body_encoded_shape = self._build_encoder(
            name, default_activation, adjusted_ob_shape)
        self._decoder_model = self._build_decoder(
            name, adjusted_ob_shape, body_encoded_shape, default_activation)
        self.trainable_model = self._build_trainable_model(
            name, observation_shape, ob_pad_width,
            self._encoder_model, self._decoder_model, vae_beta,
            optimizer)
        self._compression_model = self._build_compression_model(
            name, observation_shape, self.trainable_model, self._encoder_model)
        self._decompression_model = self._build_decompression_model(
            name, latent_dim, self.trainable_model, self._decoder_model)

    def _build_encoder(self, name: str, default_activation: str,
                       observation_shape: Tuple[int, int, int]):
        observation = Input(shape=observation_shape,
                            name=name + 'encoder_observation')

        encoded_pic = observation
        for layer_id, num_filters in enumerate(self.encoder_filters):
            encoded_pic = (
                Conv2D(
                    filters=num_filters, kernel_size=4, strides=2,
                    activation=default_activation,
                    name=name + 'enc' + str(layer_id),
                    **({'input_shape': observation_shape}
                       if layer_id == 0 else {}))
                (encoded_pic))

        body_encoded_shape = K.int_shape(encoded_pic)[1:]

        flat_encoded_pic = Flatten()(encoded_pic)
        z_mu = Dense(self.latent_dim, name=name + 'z_mu')(flat_encoded_pic)
        z_log_var = (
            Dense(self.latent_dim, name=name + 'z_log_var')
            (flat_encoded_pic))

        def sampling(args):
            z_mean, _z_log_var = args
            epsilon = K.random_normal(
                shape=K.shape(z_mean),
                mean=0., stddev=1.0)
            return z_mean + K.exp(_z_log_var / 2) * epsilon

        z = Lambda(sampling,
                   output_shape=(self.latent_dim,))([z_mu, z_log_var])
        return (
            Model(inputs=observation, outputs=[z_mu, z_log_var, z],
                  name=name + 'encoder_model'),
            body_encoded_shape)

    def _build_decoder(self, name: str,
                       observation_shape: Tuple[int, int, int],
                       encoded_shape: Tuple[int, int, int],
                       default_activation: str):
        decoder_filters = list(reversed(self.encoder_filters))[1:]
        # decoder_filters[-1] = observation_shape[-1]  # copy num_channels
        z = Input(shape=(self.latent_dim,), name=name + 'latent_z')
        flat_encoded_pic_size = functools.reduce(operator.mul, encoded_shape)
        pre_decoder = Dense(
            flat_encoded_pic_size,
            activation=default_activation, name=name + 'latent_fc')(z)
        pre_decoder = Reshape(encoded_shape, name='fc2decoder')(pre_decoder)

        decoded_image = pre_decoder
        for layer_id, num_filters in enumerate(decoder_filters):
            decoded_image = (
                Conv2DTranspose(
                    filters=num_filters, kernel_size=4, strides=2,
                    activation=default_activation,
                    name=name + 'dec' + str(layer_id))
                (decoded_image))
        decoded_image = (
            Conv2DTranspose(
                filters=3, kernel_size=4, strides=2,
                activation='sigmoid',
                name=name + 'color_dim_reducer')
            (decoded_image))

        # In line below is a workaround for bug presented in Keras 2.1.8
        # leading to undefined shape of tensors returned by Conv2DTranspose
        # https://github.com/keras-team/keras/issues/6777
        # noinspection PyTypeChecker
        decoded_image.set_shape((None,) + observation_shape)
        model = Model(inputs=z, outputs=decoded_image,
                      name=name + 'decoder_model')
        return model

    def _build_trainable_model(self, name: str,
                               observation_shape: Tuple[int, int, int],
                               ob_pad_width: ImagePadding,
                               encoder: Model, decoder: Model,
                               vae_beta: float,
                               optimizer: optimizers.Optimizer):

        observation = Input(shape=observation_shape,
                            name=name + 'trainable_observation')

        adjusted_observation = ZeroPadding2D(
            ob_pad_width, name=name + 'pre_conv_pad')(observation)
        z_mu, z_log_var, z = encoder(adjusted_observation)
        adjusted_decoded_image = decoder(z)
        decoded_image = Cropping2D(
            ob_pad_width, name=name + 'post_conv_crop')(adjusted_decoded_image)

        model = Model(inputs=observation, outputs=decoded_image,
                      name=name + 'trainable_model')
        if optimizer is not None:
            vae_loss = self.form_vae_loss(
                adjusted_decoded_image, adjusted_observation,
                vae_beta, z_log_var, z_mu)
            model.add_loss(vae_loss)

            if K.backend() == 'tensorflow':
                # noinspection PyPackageRequirements
                import tensorflow as tf
                kwargs = {
                    'options': tf.RunOptions(
                        report_tensor_allocations_upon_oom=True)
                }
            else:
                kwargs = {}
            model.compile(optimizer, loss=[None], **kwargs)
        return model

    @staticmethod
    def _build_compression_model(name: str,
                                 observation_shape: Tuple[int, int, int],
                                 trainable_model: Model,
                                 encoder: Model):
        observation = Input(shape=observation_shape,
                            name=name + 'observation_to_encode')
        padding_layer = trainable_model.get_layer(name + 'pre_conv_pad')
        adjusted_observation = padding_layer(observation)
        z_mu, z_log_var, z = encoder(adjusted_observation)
        variance = Lambda(K.exp, output_shape=K.int_shape(z_log_var),
                          name=name + 'variance')(z_log_var)
        return Model(inputs=observation, outputs=[z_mu, variance, z],
                     name=name + 'compress_model')

    @staticmethod
    def _build_decompression_model(name: str,
                                   latent_dim_size: int,
                                   trainable_model: Model,
                                   decoder: Model):
        z = Input(shape=(latent_dim_size,), name=name + 'z_to_decode')
        adjusted_decoded_image = decoder(z)
        cropping_layer = trainable_model.get_layer(name + 'post_conv_crop')
        decoded_image = cropping_layer(adjusted_decoded_image)
        return Model(inputs=z, outputs=decoded_image, name=name + 'decompress')

    @staticmethod
    def form_vae_loss(predicted_image, true_image,
                      vae_beta, z_log_var, z_mu):
        kl_loss = (
            -0.5 * K.sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var),
                         axis=-1))
        batch_size = K.shape(true_image)[0]
        # The original "World Models" paper states the authors used
        # L2 distance as auto-encoder`s reconstruction loss.
        # However it seems that binary_crossentropy trains faster and fits
        # better here due to the colors of the frames already being normalized
        # to 0..1 range.
        reconstruction_loss = (
            K.sum(
                K.reshape(
                    K.binary_crossentropy(true_image, predicted_image),
                    (batch_size, -1)),
                axis=-1))
        vae_loss = K.mean(reconstruction_loss + vae_beta * kl_loss)
        return vae_loss

    def compress(self, observations):
        return self._compression_model.predict_on_batch(observations)

    def decompress(self, z_vectors):
        return self._decompression_model.predict_on_batch(z_vectors)


class EnvModelInput(NamedTuple):
    encoded_obs: np.ndarray
    actions: np.ndarray
    expected_z: np.ndarray


class EnvModelOutput(NamedTuple):
    mu: np.ndarray
    variances: np.ndarray
    mixture_weights: np.ndarray
    rnn_outputs: np.ndarray
    predicted_reward: np.ndarray
    is_done: np.ndarray


class EnvModelPrediction(NamedTuple):
    observations: np.ndarray  # (batch_size, time_steps, latent_dim_size)
    is_done: np.ndarray  # (batch_size, time_steps), type: bool


class WorldEnvModel(StatefulNet):
    """
    MDN-RNN model (a combination of a Mixture Density Network and an RNN)
    used to predict future observations in the environment.

    The implementation is based on papers
    [Generating Sequences With Recurrent Neural Networks]
    (https://arxiv.org/pdf/1308.0850/)
    [A Neural Representation of Sketch Drawings]
    (https://arxiv.org/pdf/1704.03477/)
    except it uses just simple diagonal covariance matrix.
    """
    def __init__(self, name: str, batch_size: int, time_steps: Optional[int],
                 num_actions: int, latent_dim_size: int, num_rnn_units: int,
                 mixture_size: int, temperature: float,
                 optimizer: Optional[optimizers.Optimizer]=None):
        encoded_obs = Input(
            batch_shape=(batch_size, time_steps, latent_dim_size),
            name=name + 'z_input')
        actions = Input(
            batch_shape=(batch_size, time_steps,),
            dtype='int32', name=name + 'a_input')
        actions_categorical = Lambda(
            lambda x: K.one_hot(x, num_actions),
            output_shape=(time_steps, num_actions))(actions)
        rnn_input = Concatenate(
            name=name + 'merged_input')([encoded_obs, actions_categorical])
        self._rnn_layer = LSTM(num_rnn_units, return_sequences=True,
                               stateful=True, name=name + 'rnn')
        rnn_outputs = self._rnn_layer(rnn_input)
        temperature = K.constant(temperature)
        param_group_size = latent_dim_size * mixture_size
        expected_z = Input(
            shape=(time_steps, latent_dim_size,),
            name=name + 'z_input_expected')
        mu = TimeDistributed(
            Dense(param_group_size), name=name + 'mu')(rnn_outputs)
        mu = TimeDistributed(Reshape((mixture_size, latent_dim_size)))(mu)
        log_variances = TimeDistributed(
            Dense(param_group_size), name=name + 'log_var')(rnn_outputs)
        log_variances = TimeDistributed(
            Reshape((mixture_size, latent_dim_size)))(log_variances)
        variances = TimeDistributed(
            Lambda(K.exp, output_shape=(mixture_size, latent_dim_size)),
            name=name + 'variances')(log_variances)
        raw_mixture_weights = TimeDistributed(
            Dense(mixture_size),
            name=name + 'raw_mix_weights')(rnn_outputs)
        raw_mixture_weights = Lambda(
            lambda x: x / temperature, output_shape=(mixture_size,),
            name=name + 'apply_temp')(raw_mixture_weights)
        mixture_weights = Activation('softmax')(raw_mixture_weights)
        predicted_reward = TimeDistributed(
            Dense(1,),
            name=name + 'predicted_reward')(rnn_outputs)
        is_done = TimeDistributed(
            Dense(1, activation='sigmoid'),
            name=name + 'predicted_done')(rnn_outputs)

        model = Model(
            name=name + 'model',
            inputs=EnvModelInput(
                encoded_obs=encoded_obs,
                actions=actions,
                expected_z=expected_z),
            outputs=EnvModelOutput(
                mu=mu,
                variances=variances,
                mixture_weights=mixture_weights,
                rnn_outputs=rnn_outputs,
                predicted_reward=predicted_reward,
                is_done=is_done))
        if optimizer is not None:
            expected_z_expanded = K.expand_dims(expected_z, 2)

            # The loss as it would look like formally, as shown in the papers:
            # pdf = (
            #         K.exp(-0.5 * K.sum(K.square(expected_z_expanded - mu)
            #                            / variances, axis=-1))
            #         / K.sqrt(np.power(2 * np.pi, latent_dim_size)
            #                  * K.prod(variances, axis=-1)))
            # mixture_pdf = K.sum(mixture_weights * pdf, axis=-1)
            # mixture_loss = K.mean(K.sum(K.log(mixture_pdf), axis=-1))

            # The same loss re-written to be more numerically stable
            mixture_loss = K.mean(
                -K.sum(
                    K.logsumexp(
                        K.log(mixture_weights)
                        - 0.5 * K.sum(K.square(expected_z_expanded - mu)
                                      / variances,
                                      axis=-1)
                        - 0.5 * latent_dim_size * K.log(2 * np.pi)
                        - 0.5 * K.sum(log_variances, axis=-1),
                        axis=-1),
                    axis=-1))
            model.add_loss(mixture_loss)
            model.compile(optimizer, loss=[
                None, None, None, None,
                losses.mean_squared_error,   # loss for reward predictions
                losses.binary_crossentropy,  # loss for the "done" signal
            ])
        self.model = model

    @staticmethod
    def draw_samples(env_output: EnvModelOutput)->EnvModelPrediction:
        batch_size, time_steps, mixtures = env_output.mixture_weights.shape
        latent_dim_size = env_output.mu.shape[-1]
        observations = np.zeros((batch_size, time_steps, latent_dim_size))
        for b in range(batch_size):
            for t in range(time_steps):
                mixture = np.random.choice(
                    mixtures, p=env_output.mixture_weights[b, t])
                means = env_output.mu[b, t, mixture]
                variances = env_output.variances[b, t, mixture]
                samples = (
                    means +
                    np.random.standard_normal(means.shape) * np.sqrt(variances)
                )
                observations[b, t] = samples
        return EnvModelPrediction(
            observations=observations,
            is_done=(env_output.is_done > 0.5).reshape(batch_size, time_steps))

    def current_states(self):
        return K.batch_get_value(self._rnn_layer.states)

    def reset_states(self, states=None):
        self._rnn_layer.reset_states(states)

    @staticmethod
    def state_for_controller(states):
        return states[0]


class I2AController:
    """
    Controller is responsible for determining the course of actions,
    based on the information provided by the VAE and MDN-RNN.

    Unlike the original paper, where the controller was a simple linear unit
    optimized using CMA-ES algorithm, this code constructs a bit more
    sophisticated controller with two hidden layers and two outputs
    representing policy and value functions. It is trained using classical
    Advantage-Actor-Critic algorithm.
    """
    def __init__(self, name: str, observation_size: int, num_actions: int,
                 optimizer: optimizers.Optimizer, normalize_returns=True,
                 reward_scale=1.0):
        self._controller_model = self.make_controller_model(
            name, observation_size, num_actions)
        self.pop_art_layer = (
            PopArtLayer(name=name + 'norm_V') if normalize_returns
            else lambda x: x)
        self.trainable_model = self.make_trainable_controller(
            name, observation_size, self._controller_model, optimizer,
            self.pop_art_layer)
        self._to_one_hot_actions = OneHotConverter(num_actions)
        self.reward_scale = reward_scale
        self.normalize_returns = normalize_returns

    @staticmethod
    def make_controller_model(name: str, observation_size: int,
                              num_actions: int):
        obs = Input(shape=(observation_size,), name=name + 'obs_input')
        fc = (Dense(64, activation='selu', name=name + 'dense1')
              (obs))
        fc = (Dense(64, activation='selu', name=name + 'dense2')
              (fc))
        policy_output = (
            Dense(num_actions, activation='softmax', name=name + 'pi')
            (fc))
        value_output = (
            Dense(1, activation=None, name=name + 'V')
            (fc))
        return Model(inputs=obs,
                     outputs=[policy_output, value_output],
                     name=name + 'base_controller')

    @staticmethod
    def make_trainable_controller(name: str, observation_size: int,
                                  controller: Model,
                                  optimizer: optimizers.Optimizer,
                                  pop_art_layer: PopArtLayer):
        obs = Input(shape=(observation_size,), name=name + 'obs_input')
        adv_input = Input(
            shape=(1,), dtype='float32', name=name + 'advantages')
        policy_output, value_output = controller(obs)
        value_output = pop_art_layer(value_output)

        def vf_loss(y_true, y_pred):
            loss = 0.5 * K.mean(K.sum(K.square(y_pred - y_true), axis=-1))
            return loss

        def policy_loss(y_true, y_pred):
            pg_loss = (
                K.reshape(adv_input, (-1,)) *
                K.categorical_crossentropy(target=y_true, output=y_pred))
            entropy_loss = -K.sum(y_pred * K.log(y_pred), axis=-1)
            return K.mean(pg_loss - 0.01 * entropy_loss)

        model = Model(inputs=[obs, adv_input],
                      outputs=[policy_output, value_output],
                      name=name + 'trainable')
        model.compile(optimizer, loss=[policy_loss, vf_loss])
        return model

    def train_on_sample(self, sample: EpisodeSample,
                        encoded_obs_and_world_state: np.ndarray,
                        reward_discount: float):
        batch_advantages, expected_returns = a2c_advantage_estimator(
            sample, reward_discount, self.reward_scale,
            self.pop_art_layer if self.normalize_returns else None)
        loss_value = self.trainable_model.train_on_batch(
            x=[encoded_obs_and_world_state, batch_advantages.reshape((-1, 1))],
            y=[self._to_one_hot_actions(sample.batch_actions.reshape(-1)),
               np.expand_dims(expected_returns.reshape(-1), -1)])
        if not np.all(np.isfinite(loss_value)):
            raise RuntimeError("Non-finite loss detected")
        loss_dict = {n: v for n, v in
                     zip(self.trainable_model.metrics_names, loss_value)}
        return loss_dict

    def predict_on_actions(self, observations):
        return self._controller_model.predict_on_batch(observations)
