from abc import ABCMeta
from typing import Any

import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.models import Model
from keras.layers import TimeDistributed, LSTM

from kerl.common.models import PolicyGradientNet, baseline_cnn_vision_model


class A2CNet(PolicyGradientNet, metaclass=ABCMeta):
    entropy_coeff = 0.01
    vf_coeff = 0.5

    def vf_loss(self, y_true, y_pred):
        loss = (self.vf_coeff *
                K.mean(K.sum(K.square(y_pred - y_true), axis=-1)))
        return loss

    def policy_loss(self, y_true, y_pred):
        pg_loss = (self.adv_input *
                   K.categorical_crossentropy(target=y_true, output=y_pred))
        entropy_loss = -K.sum(y_pred * K.log(y_pred), axis=-1)
        return K.mean(pg_loss - self.entropy_coeff * entropy_loss)

    def make_trainable_model(self, main_model) -> Model:
        policy_output, value_output = main_model([self.obs_input])
        model = Model(
            name=main_model.name + '_trainable',
            inputs=[self.obs_input, self.adv_input],
            outputs=[policy_output, value_output])
        model.compile(self.optimizer, loss=[self.policy_loss, self.vf_loss])
        return model


class ConvolutionalA2CNet(A2CNet):

    # noinspection PyAttributeOutsideInit
    def init_extra_layers(self):
        vision_net = baseline_cnn_vision_model(
            self.observation_input_shape, self.name)
        self.vision_net_layer = TimeDistributed(
            vision_net, name=self.name + 'vision')

    def make_main_model(self):
        processed_observation = self.vision_net_layer(self.obs_input)
        policy_output = self.policy_output_layer(processed_observation)
        value_pre_output = self.value_output_layer(processed_observation)
        value_output = self.norm_value_output_layer(value_pre_output)
        model = Model(
            name=self.name + 'cnn_model',
            inputs=[self.obs_input],
            outputs=[policy_output, value_output])
        return model

    def current_states(self) -> Any:
        return None

    def reset_states(self, states=None):
        pass


class RecurrentA2CNet(ConvolutionalA2CNet):
    num_rnn_unis = 256

    # noinspection PyAttributeOutsideInit
    def init_extra_layers(self):
        super().init_extra_layers()
        self._rnn_layer = LSTM(
            self.num_rnn_unis, return_sequences=True,
            recurrent_activation='sigmoid',
            stateful=True, implementation=2, name=self.name + 'rnn')

    def make_main_model(self):
        processed_observation = self.vision_net_layer(self.obs_input)
        rnn_output = self._rnn_layer(processed_observation)
        policy_output = self.policy_output_layer(rnn_output)
        value_pre_output = self.value_output_layer(rnn_output)
        value_output = self.norm_value_output_layer(value_pre_output)
        model = Model(
            name=self.name + 'rnn_model',
            inputs=[self.obs_input],
            outputs=[policy_output, value_output])
        return model

    def reset_states(self, states=None):
        self._rnn_layer.reset_states(states)

    def current_states(self) -> Any:
        return K.batch_get_value(self._rnn_layer.states)

    def reset_particular_states(self, mask):
        if not np.any(mask):
            return
        states = self.current_states()
        if states is not None:
            for state in states:
                state[mask] = 0
            self.reset_states(states)

