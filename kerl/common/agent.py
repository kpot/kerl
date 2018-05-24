from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import NamedTuple, Optional, Tuple, List, Any, Dict, Union

import h5py
import numpy as np
from keras.models import Model
from keras.engine.topology import (
    save_weights_to_hdf5_group, load_weights_from_hdf5_group_by_name)
from keras import backend

from kerl.common.multi_gym_env import MultiGymEnv
from kerl.common.utils import save_optimizer_weights, load_optimizer_weights


class EpisodeSample(NamedTuple):
    """
    Stores data collected while running a multi-episode during several steps.
    """
    # boolean flags indicating the end of the episode (game over)
    # *after* the action was taken
    batch_dones: np.ndarray
    # discreet actions being taken during each step
    batch_actions: np.ndarray
    # distribution of probabilities of different discrete actions,
    # outputted by the agent
    batch_probs: np.ndarray
    # observation being fed in the agent to decide upon the batch_actions
    batch_observations: np.ndarray
    # expected future returns
    batch_values: np.ndarray
    # reward received after the action has been taken
    batch_rewards: np.ndarray
    # expected future return from the action past the last executed action
    # in the current sample
    bootstrap_values: np.ndarray
    # observation after the last action in the series was made,
    # i.e. next after the last in batch_observations
    # (not the last action in the episode!)
    last_observations: np.ndarray
    # agent's internal (RNN) states (before it chooses an action)
    agent_internal_states: Optional[Any]
    # agent's internal state after the sampling was done (useful for debugging)
    last_internal_states: Optional[Any]


class MultiAgent(metaclass=ABCMeta):
    """
    Represents an abstract deep RL agent, which can act and learn
    in multiple environments.
    It can employ multiple neural for the task, providing unified interface
    for saving their parameters during the training, accessing current states,
    and so on.
    """

    reward_discount = 0.99
    reward_scale = 1.0
    normalize_returns = True

    def __init__(self, env: MultiGymEnv, **kwargs):
        self._num_envs = env.num_envs
        self._num_actions = env.action_space.n
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(
                    'Cannot reassign property called {}'.format(key))
            setattr(self, key, value)

    @property
    @abstractmethod
    def models(self)->List[Model]:
        pass

    @abstractmethod
    def multi_env_act(self, one_step_observations)->Tuple[
            np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        pass

    @abstractmethod
    def train_on_sample(self, sample: EpisodeSample):
        pass

    @abstractmethod
    def reset_particular_agents(self, mask: Union[List[bool], np.ndarray]):
        pass

    @abstractmethod
    def reset_states(self, states=None):
        pass

    @abstractmethod
    def current_states(self)->Any:
        pass

    def save_model(self, model_path):
        with h5py.File(model_path, 'w') as f:
            for model in self.models:
                model_group = f.create_group(model.name)
                save_weights_to_hdf5_group(model_group, model.layers)
                save_optimizer_weights(model, model_group)
                f.flush()

    def load_model(self, model_path):
        with h5py.File(model_path, 'r') as f:
            for model in self.models:
                try:
                    hdf5_group = f[model.name]
                except KeyError:
                    print('Given HDF5 file ({hdf}) does not '
                          'contain model {model}. Skipping.'
                          .format(model=model.name, hdf=model_path))
                else:
                    kwargs = {}
                    if 'skip_mismatch' in (load_weights_from_hdf5_group_by_name
                                           .__code__.co_varnames):
                        kwargs['skip_mismatch'] = True
                    load_weights_from_hdf5_group_by_name(
                        hdf5_group, model.layers, **kwargs)
                    load_optimizer_weights(model, hdf5_group)

    def learning_rates(self)->Dict[str, float]:
        models = self.models
        learning_rates = backend.batch_get_value(
            [model.optimizer.lr for model in models])
        return {model.name: rate
                for model, rate in zip(models, learning_rates)}


def episode_sampler(num_steps: int,
                    multi_env: MultiGymEnv,
                    agent: MultiAgent,
                    observations: np.ndarray)->EpisodeSample:
    """
    Runs an agent through a multi-episode during num_steps, recording
    all observations, rewards, actions, etc.
    It assumes the episode has already been started earlier, and
    the agent is fully initialized if necessary.
    :param num_steps: during how many steps run the multi-episode
    :param multi_env: a multi-environment generating the episode
    :param agent: a multi-agent, making decisions
    :param observations: an initial multi-observation
        (like the last_observation field) from the previous run
        of this function.
    :returns: all recordings packed in a NamedTuple
    """
    num_envs = multi_env.num_envs
    batch_dones = np.zeros((num_envs, num_steps), dtype='bool')
    batch_actions = np.zeros((num_envs, num_steps), dtype='int32')
    batch_probs = np.zeros((num_envs, num_steps, multi_env.action_space.n))
    batch_observations = np.zeros(
        (num_envs, num_steps,)
        + multi_env.observation_space.shape
        + (multi_env.num_observations_stacked,))
    batch_values = np.zeros((num_envs, num_steps))
    batch_rewards = np.zeros((num_envs, num_steps))
    internal_states = []

    for step in range(num_steps):
        policy_output, sampled_actions, value_output, internal_state = (
            agent.multi_env_act(observations))
        batch_observations[:, step] = observations
        batch_actions[:, step] = sampled_actions
        batch_values[:, step] = value_output
        internal_states.append(deepcopy(internal_state))
        observations, rewards, dones, infos = multi_env.step(sampled_actions)
        batch_rewards[:, step] = rewards
        batch_dones[:, step] = dones
        batch_probs[:, step] = policy_output
    # We need a glimpse into the future value for bootstrapping, but
    # we don't want to damage the state the agent is currently in
    end_sample_states = deepcopy(agent.current_states())
    _, _, bootstrapped_values, _ = agent.multi_env_act(observations)
    agent.reset_states(end_sample_states)

    samples = EpisodeSample(
        batch_dones=batch_dones,
        batch_actions=batch_actions,
        batch_probs=batch_probs,
        batch_observations=batch_observations,
        batch_values=batch_values,
        batch_rewards=batch_rewards,
        bootstrap_values=bootstrapped_values,
        last_observations=observations,
        agent_internal_states=internal_states,
        last_internal_states=end_sample_states)
    return samples
