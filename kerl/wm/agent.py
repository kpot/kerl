from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Callable, Union, List, Dict

import numpy as np
from keras import optimizers

from kerl.common.agent import MultiAgent, EpisodeSample
from kerl.common.multi_gym_env import MultiGymEnv
from kerl.wm.networks import WorldModelVAE, WorldEnvModel, I2AController


class WMInternalStates(NamedTuple):
    env_model_states: list


def rename_total_loss(loss_dict: Dict[str, float],
                      model_name: str) -> Dict[str, float]:
    full_loss = loss_dict['loss']
    result = loss_dict.copy()
    del result['loss']
    result[model_name + '_loss'] = full_loss
    return result


class WMMultiAgent(MultiAgent):
    latent_dim_size = 128
    world_rnn_units = 512

    def __init__(self, env: MultiGymEnv,
                 optimizer_maker: Callable[[], optimizers.Optimizer],
                 **kwargs):
        vae_optimizer = optimizer_maker()
        env_model_optimizer = optimizer_maker()
        controller_optimizer = optimizer_maker()
        super().__init__(env, **kwargs)
        width, height, rgb_channels = env.observation_space.shape
        vae = WorldModelVAE(
            'agent_vae_',
            (width, height, rgb_channels),
            vae_optimizer, latent_dim=self.latent_dim_size)
        self.vae = vae
        self.world = WorldEnvModel(
            'agent_world_', batch_size=self._num_envs, time_steps=None,
            num_actions=self._num_actions,
            latent_dim_size=self.latent_dim_size,
            num_rnn_units=self.world_rnn_units,
            mixture_size=6, temperature=1.0, optimizer=env_model_optimizer)
        self.controller = I2AController(
            'a2c_agent_',
            observation_size=self.latent_dim_size + self.world_rnn_units,
            num_actions=self._num_actions, optimizer=controller_optimizer,
            normalize_returns=self.normalize_returns,
            reward_scale=self.reward_scale)
        self.executor = ThreadPoolExecutor(max_workers=3)

    @property
    def models(self):
        return [self.vae.trainable_model, self.world.model,
                self.controller.trainable_model]

    def reset_states(self, states=None):
        self.world.reset_states(
            states.env_model_states if states is not None else None)

    def reset_particular_agents(self, mask: Union[List[bool], np.ndarray]):
        self.world.reset_particular_states(mask)

    def current_states(self):
        return WMInternalStates(
            env_model_states=self.world.current_states())

    def multi_env_act(self, one_step_observations):
        batch_size = one_step_observations.shape[0]
        internal_states = self.current_states()
        current_obs = one_step_observations[:, :, :, :, -1]
        _, _, encoded_current_obs = self.vae.compress(current_obs)

        encoded_obs_and_world_state = (
            np.concatenate(
                [self.world.state_for_controller(
                    internal_states.env_model_states),
                 encoded_current_obs],
                axis=-1)
            .reshape((batch_size, -1)))

        policy_output, value_output = self.controller.predict_on_actions(
            encoded_obs_and_world_state)
        sampled_actions = np.array(
            [np.random.choice(self._num_actions, p=po)
             for po in policy_output])

        encoded_current_obs_with_time = encoded_current_obs.reshape(
            batch_size, 1, self.latent_dim_size)
        self.world.model.predict_on_batch(
            x=[encoded_current_obs_with_time,
               sampled_actions,
               np.zeros_like(encoded_current_obs_with_time)])

        return (policy_output, sampled_actions, value_output[:, 0],
                internal_states)

    def train_on_sample(self, sample: EpisodeSample):
        # Encoding observations using VAE so we could use them to train both
        # the environment model and the controller.
        current_obs = sample.batch_observations[:, :, :, :, :, -1]
        last_obs = sample.last_observations[:, :, :, :, -1]
        batch_size, time_steps, width, height, channels = current_obs.shape
        _, _, encoded_current_obs = self.vae.compress(
            np.reshape(current_obs,
                       (batch_size * time_steps, width, height, channels)))
        encoded_current_obs_with_time = encoded_current_obs.reshape(
            batch_size, time_steps, self.latent_dim_size)
        _, _, encoded_last_obs = self.vae.compress(last_obs)
        encoded_future_obs_with_time = np.concatenate(
            [encoded_current_obs_with_time[:, 1:],
             np.expand_dims(encoded_last_obs, 1)],
            axis=1)
        # Training all networks
        all_losses = {}
        all_losses.update(
            self.train_vae(sample))
        all_losses.update(
            self.train_env_model(
                encoded_current_obs_with_time,
                encoded_future_obs_with_time,
                sample))
        all_losses.update(
            self.train_controller(
                encoded_current_obs_with_time, self.reward_discount, sample))
        return all_losses

    def train_vae(self, sample: EpisodeSample):
        batch_size, time_steps, width, height, channels, frames = (
            sample.batch_observations.shape)
        env_observations = (
            sample.batch_observations[:, :, :, :, :, -1]
            .reshape((batch_size * time_steps, width, height, channels)))
        vae_loss = self.vae.trainable_model.train_on_batch(
            x=env_observations, y=None)
        return {self.vae.trainable_model.name + '_loss': vae_loss}

    def train_controller(self, encoded_current_obs_with_time, reward_discount,
                         sample: EpisodeSample):
        batch_size, time_steps, width, height, channels, frames = (
            sample.batch_observations.shape)
        rnn_states_before_actions = np.transpose(
            [self.world.state_for_controller(item.env_model_states)
             for item in sample.agent_internal_states],
            axes=(1, 0, 2))
        encoded_obs_and_world_state = (
            np.concatenate(
                [rnn_states_before_actions, encoded_current_obs_with_time],
                axis=-1)
            .reshape((batch_size * time_steps, -1)))
        controller_loss = self.controller.train_on_sample(
            sample, encoded_obs_and_world_state, reward_discount)
        return rename_total_loss(
            controller_loss, self.controller.trainable_model.name)

    def train_env_model(self, encoded_current_obs_with_time,
                        encoded_future_obs_with_time, sample: EpisodeSample):
        state_before_training = self.world.current_states()
        self.world.reset_states(
            sample.agent_internal_states[0].env_model_states)
        env_model_loss = self.world.model.train_on_batch(
            x=[encoded_current_obs_with_time,
               sample.batch_actions,
               encoded_future_obs_with_time],
            y=[np.expand_dims(sample.batch_rewards, axis=-1),
               np.expand_dims(np.float32(sample.batch_dones), axis=-1)])
        env_model_named_losses = {
            n: v for n, v in zip(self.world.model.metrics_names,
                                 env_model_loss)}
        self.world.reset_states(state_before_training)
        return rename_total_loss(env_model_named_losses,
                                 self.world.model.name)
