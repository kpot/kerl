from typing import Callable, ClassVar, Optional, Tuple, Union, List

import numpy as np
from keras import optimizers

from kerl.a2c.networks import A2CNet
from kerl.common.agent import MultiAgent, EpisodeSample
from kerl.common.multi_gym_env import MultiGymEnv
from kerl.common.utils import OneHotConverter, PopArtLayer


def a2c_advantage_estimator(
        sample: EpisodeSample, reward_discount: float,
        reward_scale: float,
        pop_art_layer: Optional[PopArtLayer]=None) -> Tuple[
            np.ndarray, np.ndarray]:
    """
    Based on the data sampled from the environment,
    calculates expected return and advantages for one step of A2C update,
    normalizing them if necessary.
    :returns: a tuple of advantages and expected returns
    """
    batch_size, num_steps = sample.batch_actions.shape[:2]
    expected_returns = np.zeros((batch_size, num_steps + 1))
    de_normalized_bootstrap = (
        pop_art_layer.de_normalize(sample.bootstrap_values)
        if pop_art_layer is not None
        else sample.bootstrap_values)
    not_done = np.float32(~sample.batch_dones)
    expected_returns[:, num_steps] = de_normalized_bootstrap * not_done[:, -1]
    for t in range(num_steps - 1, -1, -1):
        expected_returns[:, t] = (
                reward_scale * sample.batch_rewards[:, t] +
                not_done[:, t] * reward_discount * expected_returns[:, t + 1])
    expected_returns = expected_returns[:, :num_steps]
    expected_returns = (
        pop_art_layer.update_and_normalize(expected_returns)[0]
        if pop_art_layer is not None
        else expected_returns)
    batch_advantages = expected_returns - sample.batch_values
    return batch_advantages, expected_returns


class A2CMultiAgent(MultiAgent):

    def __init__(self, env: MultiGymEnv,
                 optimizer_maker: Callable[[], optimizers.Optimizer],
                 net_class: ClassVar[A2CNet],
                 **kwargs):
        super().__init__(env, **kwargs)
        optimizer = optimizer_maker()
        self.to_one_hot_actions = OneHotConverter(env.action_space.n)
        width, height, rgb_channels = env.observation_space.shape
        self.net = net_class(
            name='agent_',
            observation_input_shape=(
                width, height, rgb_channels, env.num_observations_stacked),
            batch_size=self._num_envs,
            num_actions=self._num_actions,
            optimizer=optimizer,
            normalize_returns=self.normalize_returns)  # type: A2CNet

    @property
    def models(self):
        return [self.net.trainable_model]

    def reset_states(self, states=None):
        self.net.reset_states(states)

    def reset_particular_agents(self, mask: Union[List[bool], np.ndarray]):
        self.net.reset_particular_states(mask)

    def current_states(self):
        return self.net.current_states()

    def multi_env_act(self, one_step_observations):
        states_before = self.current_states()
        obs_in_time = np.expand_dims(one_step_observations, 1)
        policy_output, value_output = self.net.main_model.predict_on_batch(
            [obs_in_time])
        sampled_actions = [np.random.choice(self._num_actions, p=po)
                           for po in policy_output[:, 0]]
        return (policy_output[:, 0], sampled_actions, value_output[:, 0, 0],
                states_before)

    def train_on_sample(self, sample: EpisodeSample):
        if sample.last_internal_states is not None:
            if np.allclose(sample.last_internal_states[0],
                           sample.agent_internal_states[0][0]):
                print('Warning! The state in the beginning of the sample'
                      'is the same as the state at the end of the sample.'
                      'The RNN does not do anything!')
        states_before_training = self.current_states()
        self.reset_states(sample.agent_internal_states[0])
        batch_advantages, expected_returns = a2c_advantage_estimator(
            sample, self.reward_discount, self.reward_scale,
            self.net.value_pop_art_layer if self.normalize_returns else None)
        loss_value = self.net.trainable_model.train_on_batch(
            x=[sample.batch_observations, batch_advantages],
            y=[self.to_one_hot_actions(sample.batch_actions),
               np.expand_dims(expected_returns, -1)])
        if not np.all(np.isfinite(loss_value)):
            raise RuntimeError("Non-finite loss detected")
        self.net.reset_states(states_before_training)
        return {n: v for n, v in zip(self.net.trainable_model.metrics_names,
                                     loss_value)}

