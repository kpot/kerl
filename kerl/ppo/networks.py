from abc import ABCMeta
from typing import Any

from keras.models import Model
from keras.layers import TimeDistributed, LSTM, Input
# noinspection PyPep8Naming
from keras import backend as K

from kerl.common.models import PolicyGradientNet, baseline_cnn_vision_model


class PPONet(PolicyGradientNet, metaclass=ABCMeta):
    # can be overridden via constructor kwargs
    clip_range = 0.2
    entropy_coeff = 0.01
    vf_coeff = 0.5

    # noinspection PyAttributeOutsideInit
    def init_extra_layers(self):
        self.old_probs_input = Input(
            shape=(None, self.num_actions), dtype='float32',
            name=self.name + 'old_probs')
        self.old_values_input = Input(
            shape=(None, 1), dtype='float32',
            name=self.name + 'old_values')

    def vf_loss(self, y_true, y_pred):
        y_pred_clipped = (
            self.old_values_input + K.clip(y_pred - self.old_values_input,
                                           -self.clip_range, self.clip_range))
        vf_losses1 = K.square(y_pred - y_true)
        vf_losses2 = K.square(y_pred_clipped - y_true)
        return self.vf_coeff * K.mean(K.maximum(vf_losses1, vf_losses2))

    def policy_loss(self, y_true, y_pred):
        neg_log_prob_diff = (
            K.categorical_crossentropy(
                target=y_true, output=self.old_probs_input)
            - K.categorical_crossentropy(target=y_true, output=y_pred))
        ratio = K.exp(neg_log_prob_diff)
        pg_loss1 = -self.adv_input * ratio
        pg_loss2 = -self.adv_input * K.clip(ratio,
                                            1.0 - self.clip_range,
                                            1.0 + self.clip_range)
        pg_loss = K.mean(K.maximum(pg_loss1, pg_loss2))
        entropy_loss = K.mean(-K.sum(y_pred * K.log(y_pred), axis=-1))
        return pg_loss - self.entropy_coeff * entropy_loss

    def make_trainable_model(self, main_model)->Model:
        policy_output, value_output = main_model([self.obs_input])
        model = Model(
            name=main_model.name + '_trainable',
            inputs=[self.obs_input, self.adv_input,
                    self.old_probs_input, self.old_values_input],
            outputs=[policy_output, value_output])
        model.compile(self.optimizer, loss=[self.policy_loss, self.vf_loss])
        return model


class ConvolutionalPPONet(PPONet):

    # noinspection PyAttributeOutsideInit
    def init_extra_layers(self):
        super().init_extra_layers()
        vision_net = baseline_cnn_vision_model(
            self.observation_input_shape, self.name, activation='relu')
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
        pass

    def reset_states(self, states=None):
        pass


class RecurrentPPONet(ConvolutionalPPONet):
    num_rnn_units = 256

    # noinspection PyAttributeOutsideInit
    def init_extra_layers(self):
        super().init_extra_layers()
        self.rnn_layer = LSTM(
            self.num_rnn_units, return_sequences=True,
            implementation=2, stateful=True,
            recurrent_activation='sigmoid',
            name=self.name + 'rnn')

    def make_main_model(self):
        processed_observation = self.vision_net_layer(self.obs_input)
        rnn_output = self.rnn_layer(processed_observation)
        policy_output = self.policy_output_layer(rnn_output)
        value_pre_output = self.value_output_layer(rnn_output)
        value_output = self.norm_value_output_layer(value_pre_output)
        model = Model(
            name=self.name + 'rnn_model',
            inputs=[self.obs_input],
            outputs=[policy_output, value_output])
        return model

    def current_states(self) -> Any:
        return K.batch_get_value(self.rnn_layer.states)

    def reset_states(self, states=None):
        self.rnn_layer.reset_states(states)

