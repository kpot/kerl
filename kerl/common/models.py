from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Any

import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import (
    Input, Reshape, ZeroPadding2D, Conv2D, Flatten, Dense, TimeDistributed)

from kerl.common.utils import calc_optimal_convnet_padding, ConvOp, PopArtLayer


def baseline_cnn_vision_model(input_shape: Tuple[int, int, int, int],
                              name: str,
                              activation='relu') -> Model:
    """
    A somewhat classical CNN network used in many papers, like
    "Playing Atari with Deep Reinforcement Learning" or
    "Imagination-Augmented Agents for Deep Reinforcement Learning" by DeepMind,
    as well as by OpenAI and many other researchers.

    :param input_shape: (width, height, channels, num_time_frames)
    :param name: any name
    :param activation: activations that is going to be used
        for all non-linear units
    """
    assert len(input_shape) == 4
    width_height, num_channels, time_steps = (
        input_shape[0:2], input_shape[2], input_shape[3])
    # Adjusting the input to be easier processed by convolutions
    input_4d = Input(shape=input_shape, name=name + 'single_observation')
    input_3d = Reshape(width_height + (num_channels * time_steps,))(input_4d)
    pad_width, adjusted_width_height = (
        calc_optimal_convnet_padding(
            width_height,
            [ConvOp(kernel_size=8, strides=4),
             ConvOp(kernel_size=4, strides=2),
             ConvOp(kernel_size=3, strides=1)]))
    adjusted_input = (
        ZeroPadding2D(pad_width, name=name + 'input_pad')
        (input_3d))

    # Convolutions
    x = Conv2D(32, kernel_size=8, strides=4, padding='valid',
               name=name + 'conv1', activation=activation)(adjusted_input)
    x = Conv2D(64, kernel_size=4, strides=2, padding='valid',
               name=name + 'conv2', activation=activation)(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='valid',
               name=name + 'conv3', activation=activation)(x)

    # FC and output layers
    x = Flatten()(x)
    x = Dense(512, activation=activation, name=name + 'fc')(x)
    model = Model(inputs=input_4d, outputs=x)
    return model


class StatefulNet(metaclass=ABCMeta):
    @abstractmethod
    def current_states(self) -> Any:
        pass

    @abstractmethod
    def reset_states(self, states=None):
        pass

    def reset_particular_states(self, mask):
        """
        Sets particular states in the batch to zeros (according to the mask)
        This default code will work only if `current_states` returns a list
        of tensors with the first dimension being batch_size, and if "reset"
        of that state means zeroing it.
        For more complicated architectures an overridden version
        might be necessary.
        :param mask: a boolean 1-D tensor the size of (batch_size,)
        """
        if not np.any(mask):
            return
        states = self.current_states()
        if states is not None:
            for state in states:
                state[mask] = 0
            self.reset_states(states)


class PolicyGradientNet(StatefulNet, metaclass=ABCMeta):
    """
    Provides building blocks for various networks expected to be trained
    with policy gradient algorithm and having discreet
    action space (like CNN-based and RNN-based variants of agents).

    Keras doesn't allow simple creation of loss functions which rely on some
    extra inputs, which is the case for many bit-more-complex networks, like
    variational autoencoders or RL networks.

    This class is built to work around such limitation by creating two models
    with shared weights, one of which is used only for predictions, and
    another one - for training (the last one also contains extra inputs
    to calculate the losses properly).
    """

    def __init__(self,
                 name: str,
                 observation_input_shape: Tuple[Union[int, None], ...],
                 batch_size: int,
                 num_actions: int,
                 optimizer: optimizers.Optimizer,
                 normalize_returns: bool,
                 **kwargs):
        self.name = name
        self.observation_input_shape = observation_input_shape
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.normalize_returns = normalize_returns
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.obs_input = Input(
            batch_shape=(batch_size, None) + observation_input_shape,
            name=name + 'obs_input')
        self.adv_input = Input(
            batch_shape=(batch_size, None), dtype='float32',
            name=name + 'advantages')
        self.policy_output_layer = TimeDistributed(
            Dense(num_actions, activation='softmax'),
            name=name + 'pi')
        self.value_output_layer = TimeDistributed(
            Dense(1, activation=None),
            name=name + 'V')
        self.value_pop_art_layer = PopArtLayer(name=name + 'pop_art')
        if self.normalize_returns:
            self.norm_value_output_layer = TimeDistributed(
                self.value_pop_art_layer,
                name=name + 'norm_V')
        else:
            self.norm_value_output_layer = lambda x: x
        self.init_extra_layers()
        self.main_model = self.make_main_model()
        self.trainable_model = self.make_trainable_model(self.main_model)

    @abstractmethod
    def init_extra_layers(self):
        pass

    @abstractmethod
    def make_main_model(self):
        pass

    @abstractmethod
    def make_trainable_model(self, main_model) -> Model:
        pass

