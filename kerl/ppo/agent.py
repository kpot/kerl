"""
[Proximal Policy Optimization Algorithm (PPO)]
(https://arxiv.org/abs/1707.06347)

This implementation mostly follows the official baseline from OpenAI
https://github.com/openai/baselines/tree/master/baselines/ppo2
although it doesn't use generalized advantage estimator proposed by the paper,
sticking instead with the older estimator described by Mnih et al. in
[Asynchronous Methods for Deep Reinforcement Learning]
(https://arxiv.org/abs/1602.01783)
"""

from typing import Callable, ClassVar, Optional, Tuple, Union, List

import numpy as np
from keras import optimizers

from kerl.a2c.agent import a2c_advantage_estimator
from kerl.common.agent import MultiAgent, EpisodeSample
from kerl.common.multi_gym_env import MultiGymEnv
from kerl.common.utils import OneHotConverter, PopArtLayer
from kerl.ppo.networks import PPONet


def generalized_advantage_estimations(
        sample: EpisodeSample,
        reward_discount: float,
        reward_scale: float,
        gae_lambda: float,
        pop_art_layer: Optional[PopArtLayer]=None) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """
    Implementation of Generalized Advantage Estimator described in
    https://arxiv.org/abs/1506.02438
    """
    if pop_art_layer is not None:
        next_values = pop_art_layer.de_normalize(sample.bootstrap_values)
        values = pop_art_layer.de_normalize(sample.batch_values)
    else:
        next_values = sample.bootstrap_values
        values = sample.batch_values

    batch_size = sample.batch_observations.shape[0]
    num_steps = sample.batch_observations.shape[1]
    not_done = np.float32(~sample.batch_dones)
    batch_advantages = np.zeros((batch_size, num_steps))
    next_advantages = 0
    for t in range(num_steps - 1, -1, -1):
        next_values = (
            next_values if t == num_steps - 1
            else values[:, t + 1])
        delta = (reward_scale * sample.batch_rewards[:, t]
                 + reward_discount * not_done[:, t] * next_values
                 - values[:, t])
        batch_advantages[:, t] = next_advantages = (
            delta +
            reward_discount * gae_lambda * not_done[:, t] * next_advantages)
    expected_returns = batch_advantages + values

    if pop_art_layer is not None:
        expected_returns, mean, std_dev = (
            pop_art_layer.update_and_normalize(expected_returns))
        batch_advantages = batch_advantages / (std_dev + pop_art_layer.epsilon)
    return batch_advantages, expected_returns


class PPOMultiAgent(MultiAgent):
    n_epochs = 3
    adv_estimator_lambda = 0.95
    use_gae = True
    clip_range = 0.2

    def __init__(self, env: MultiGymEnv,
                 optimizer_maker: Callable[[], optimizers.Optimizer],
                 net_class: ClassVar[PPONet],
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
            clip_range=self.clip_range,
            normalize_returns=self.normalize_returns)  # type: PPONet

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
        policy_output, value_output = (
            self.net.main_model.predict_on_batch([obs_in_time]))
        sampled_actions = [np.random.choice(self._num_actions, p=po)
                           for po in policy_output[:, 0]]
        return (policy_output[:, 0], sampled_actions, value_output[:, 0, 0],
                states_before)

    def train_on_sample(self, sample: EpisodeSample):
        if sample.last_internal_states is not None:
            if np.allclose(sample.last_internal_states[0],
                           sample.agent_internal_states[0][0]):
                print('Alarm! The state in the beginning of the sample is the '
                      'same as the state at the end of the sample.'
                      ' The RNN does not do anything!')
        states_before_training = self.current_states()
        if self.use_gae:
            batch_advantages, expected_returns = (
                generalized_advantage_estimations(
                    sample, self.reward_discount, self.reward_scale,
                    self.adv_estimator_lambda,
                    (self.net.value_pop_art_layer if self.normalize_returns
                     else None)))
        else:
            batch_advantages, expected_returns = a2c_advantage_estimator(
                sample, self.reward_discount, self.reward_scale,
                (self.net.value_pop_art_layer if self.normalize_returns
                 else None))

        loss_values = np.array([0, 0, 0], dtype='float32')
        for epoch in range(self.n_epochs):
            loss_values += self.net.trainable_model.train_on_batch(
                x=[sample.batch_observations, batch_advantages,
                   sample.batch_probs,
                   np.expand_dims(sample.batch_values, -1)],
                y=[self.to_one_hot_actions(sample.batch_actions),
                   np.expand_dims(
                       expected_returns, -1)])
            if not np.all(np.isfinite(loss_values)):
                raise RuntimeError("Non-finite loss detected")
        self.net.reset_states(states_before_training)
        return {n: v for n, v
                in zip(self.net.trainable_model.metrics_names,
                       loss_values / self.n_epochs)}

