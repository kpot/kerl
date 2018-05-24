import random
import sys
import multiprocessing
from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple, Callable, List, Union

import numpy as np
from gym import spaces
from gym.core import ObservationWrapper
from PIL import Image


def gym_env_worker(worker_id, env_factory, master_channel, worker_channel):
    """
    Will be executed by a process spawned by ParallelGymEnv to run one
    of its multiple OpenAI Gym environments.
    """
    gym_env = env_factory(worker_id)
    master_channel.close()
    try:
        while True:
            request = worker_channel.recv()
            command, params = (
                request if isinstance(request, tuple)
                else (request, tuple()))
            if command == 'stop':
                break
            elif hasattr(gym_env, command):
                env_attr = getattr(gym_env, command)
                if callable(env_attr):
                    worker_channel.send(env_attr(*params))
                else:
                    worker_channel.send(env_attr)
            else:
                print('Unknown command', command,
                      file=sys.stderr, flush=True)
    finally:
        worker_channel.close()
        gym_env.close()


def query_worker(channel,
                 command: str,
                 args: Optional[Tuple]=None):
    channel.send((command, args))
    return channel.recv()


class ColorNormalizer:
    """
    Normalizes colors from whatever range the observation space uses
    to [0..1] range.
    """

    def __init__(self, observation_space):
        self._low_color = np.mean(observation_space.low)
        self._color_range = np.mean(
            observation_space.high - observation_space.low)
        self.blank_observation = np.zeros(
            observation_space.shape, dtype='float32')

    def normalize_colors(self, observations):
        return (observations - self._low_color) / self._color_range


class DummyColorNormalizer:
    def __init__(self, observation_space):
        self.blank_observation = observation_space.low

    @staticmethod
    def normalize_colors(observations):
        return observations


class MultiGymEnv(metaclass=ABCMeta):
    """
    Base for all wrappers around multiple simultaneously working
    OpenAI Gym environments.

    The main difference here is that such wrapper operates with multiple
    actions at once, and returns multiple observations for each
    of the environments it runs.

    The observations also become different. Each now combines multiple
    most recent "frames" stacked in extra "time" dimension (the last one).
    """
    def __init__(self, num_envs: int, observation_space, action_space,
                 stack_observations: int, normalize_colors: bool=True):
        assert num_envs > 0
        assert stack_observations > 0
        self.num_envs = num_envs
        self.num_observations_stacked = stack_observations
        self.observation_space = observation_space
        self.action_space = action_space
        self.recent_observations = np.zeros(
            (stack_observations, num_envs,) + self.observation_space.shape,
            dtype=np.float32)
        self.recent_dones = np.zeros((num_envs,), dtype=np.bool)
        self.color_normalizer = (
            ColorNormalizer(self.observation_space)
            if normalize_colors
            else DummyColorNormalizer(self.observation_space))

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode: str):
        pass

    @abstractmethod
    def step(self, actions: Union[List[int], np.ndarray]):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def close(self):
        pass

    def export_recent_observations(self):
        return np.transpose(self.recent_observations, (1, 2, 3, 4, 0))

    def stack_observations(self, new_observations: list)->np.ndarray:
        """
        Accepts a list of observations (frames) returned by each environment
        (each having shape env.observation_space.shape)
        and adds these frames to previously known frames.
        :return: numpy.ndarray the shape of
          (num_envs, <the shape of a single observation>, num_stacked_frames).
        """
        self.recent_observations[0:-1, :, :, :, :] = (
            self.recent_observations[1:, :, :, :, :])
        self.recent_observations[-1, :, :, :, :] = (
            self.color_normalizer.normalize_colors(new_observations))
        return self.export_recent_observations()


class ParallelGymEnv(MultiGymEnv):
    """
    Runs multiple simulations inside individual processes. See MultiGymEnv
    for more info.
    """
    def __init__(self, env_factory: Callable, num_envs: int,
                 stack_observations: int, normalize_colors: bool=True):
        self.observations_to_stack = stack_observations
        comm_channels = [multiprocessing.Pipe() for _ in range(num_envs)]
        self.all_workers = [
            multiprocessing.Process(
                target=gym_env_worker,
                kwargs={'worker_id': i + random.randint(0, 1000),
                        'env_factory': env_factory,
                        'master_channel': master_channel,
                        'worker_channel': worker_channel})
            for i, (master_channel, worker_channel)
            in enumerate(comm_channels)]
        for p in self.all_workers:
            p.daemon = True
            p.start()
        self.comm_channels = []  # type: List[multiprocessing.Connection]
        for master_channel, worker_channel in comm_channels:
            worker_channel.close()
            self.comm_channels.append(master_channel)
        super().__init__(
            num_envs=num_envs,
            observation_space=query_worker(
                self.comm_channels[0], 'observation_space'),
            action_space=query_worker(
                self.comm_channels[0], 'action_space'),
            stack_observations=stack_observations,
            normalize_colors=normalize_colors)

    def query_all_workers(self, command: str, args: Optional[Tuple]=None):
        if args is None:
            args = ()
        for c in self.comm_channels:
            c.send((command, args))
        returns = [c.recv() for c in self.comm_channels]
        if isinstance(returns[0], tuple):
            return zip(*returns)
        return returns

    def reset(self):
        observations = self.query_all_workers('reset')
        self.recent_dones[...] = False
        self.recent_observations[:] = self.color_normalizer.blank_observation
        return self.stack_observations(observations)

    def reset_recently_done(self):
        awaiting = []
        for i, (c, done) in enumerate(zip(self.comm_channels,
                                          self.recent_dones)):
            if done:
                c.send(('reset', ()))
                awaiting.append((i, c))
        for i, c in awaiting:
            new_observation = c.recv()
            self.recent_observations[0:-1, i, :, :, :] = (
                self.color_normalizer.blank_observation)
            self.recent_observations[-1, i, :, :, :] = (
                self.color_normalizer.normalize_colors(new_observation))
            self.recent_dones[i] = False
        return self.export_recent_observations()

    def render(self, mode='human'):
        """
        Renders only a single frame for each environment,
        and only in rgb_array mode.
        """
        assert mode == 'rgb_array'
        return self.query_all_workers('render', (mode,))

    def step(self, actions):
        for action, channel in zip(actions, self.comm_channels):
            channel.send(('step', (action,)))
        returns = (c.recv() for c in self.comm_channels)
        observations, rewards, dones, infos = zip(*returns)
        # Blanking observations for simulations that have been completed before
        for ob, done in zip(observations, self.recent_dones):
            if done:
                ob[...] = self.observation_space.low
        self.recent_dones = np.array(dones)
        return (self.stack_observations(observations),
                np.array(rewards), self.recent_dones, infos)

    def seed(self, seed=None):
        self.query_all_workers('seed', (seed,))

    def close(self):
        for w in self.comm_channels:
            w.send('stop')


class MultiGymImitation(MultiGymEnv):
    """
    Imitates behaviour multi-gym environment (like ParallelGymEnv) but using
    only one pre-defined gym environment.
    Unlike ParallelGymEnv, doesn't use any threads or processes
    and allows normal rendering.
    See MultiGymEnv for more info.
    """
    def __init__(self, env, stack_observations: int,
                 normalize_colors: bool=True):
        self.env = env
        super().__init__(
            num_envs=1,
            observation_space=env.observation_space,
            action_space=env.action_space,
            stack_observations=stack_observations,
            normalize_colors=normalize_colors)

    def reset(self):
        observation = self.env.reset()
        return self.stack_observations([observation])

    def render(self, *args, **kwargs):
        observation = self.env.render(*args, **kwargs)
        return [observation]

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions[0])
        return (self.stack_observations([observation]),
                np.array([reward]),
                np.array([done]),
                [info])

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()


class ObservationResizer(ObservationWrapper):
    def __init__(self, env, new_size):
        super().__init__(env)
        orig_obs_space = env.observation_space
        self.new_size = new_size
        self.observation_space = spaces.Box(
            low=orig_obs_space.low.reshape(-1)[0],
            high=orig_obs_space.high.reshape(-1)[0],
            shape=new_size + (3,),
            dtype=orig_obs_space.dtype)

    def observation(self, observation):
        image = Image.fromarray(observation, 'RGB')
        resized_image = image.resize(self.new_size, Image.NEAREST)
        resized_array = np.array(resized_image)
        return resized_array
