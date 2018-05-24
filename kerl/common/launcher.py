import argparse
import json
import multiprocessing
import time
from abc import ABCMeta, abstractmethod
from pprint import pprint
from typing import Tuple, Optional, Dict, Any
import os.path
import sys

import h5py
from keras import optimizers
import keras.backend
import numpy as np
import gym

from kerl.common.agent import episode_sampler, MultiAgent
from kerl.common.history import TrainHistoryRecorder
from kerl.common.multi_gym_env import (
    ParallelGymEnv, MultiGymImitation, ObservationResizer, MultiGymEnv)
from kerl.common.utils import MovieWriter

TrainingMetadata = Dict[str, Any]
METADATA_KEY = 'kerl_metadata'


def save_training_metadata(model_path: str, metadata: TrainingMetadata):
    with h5py.File(model_path, 'r+') as hdf:
        hdf.attrs[METADATA_KEY] = (
            json.dumps(metadata).encode('utf-8'))


def load_training_metadata(model_path: str) -> Optional[TrainingMetadata]:
    with h5py.File(model_path, 'r') as hdf:
        if METADATA_KEY not in hdf.attrs:
            return None
        data = json.loads(hdf.attrs[METADATA_KEY].decode('utf-8'))
        return data


def contain_tf_gpu_mem_usage():
    """
    By default TensorFlow may try to reserve all available GPU memory
    making it impossible to train multiple agents at once.
    This function will disable such behaviour in TensorFlow.
    """
    try:
        # noinspection PyPackageRequirements
        import tensorflow as tf
    except ImportError:
        pass
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory
        sess = tf.Session(config=config)
        set_session(sess)


def backend_specific_tweaks():
    if keras.backend.backend() == 'tensorflow':
        contain_tf_gpu_mem_usage()


class CmdLineParsers:
    def __init__(self, description):
        self.main_parser = argparse.ArgumentParser(description=description)
        self.main_parser.add_argument(
            '--model-path', type=str, metavar='PATH',
            help='Where to store the model', required=True)
        self.sub_parsers = self.main_parser.add_subparsers(
            help='sub-command help', dest='command')
        self.train_parser = self.sub_parsers.add_parser(
            'train', help='Launches training')
        self.train_parser.add_argument(
            '--gym-env', type=str, metavar='ENV_NAME',
            help='Name of the OpenAI Gym environment, '
                 'like "Pong-v0" or "MsPacman-v0"',
            required=True)
        self.train_parser.add_argument(
            '--num-envs', type=int, default=multiprocessing.cpu_count(),
            help='How many simulation environments to use')
        self.train_parser.add_argument(
            '--history-path', type=str, default='', metavar='PATH',
            help='File for recording average score after each episode')
        self.train_parser.add_argument(
            '--lr', type=float, default=2e-4, help='Learning rate',
            metavar='FLOAT_NUM')
        self.train_parser.add_argument(
            '--time-horizon', type=int, default=5,
            help='Time horizon for game samples')
        self.train_parser.add_argument(
            '--norm-returns', action='store_true',
            help='Use Pop-Art return normalization')
        self.train_parser.add_argument(
            '--reward-scale', type=float, default=1.0, metavar='FLOAT_NUM',
            help='Scale rewards the agent sees with the given coefficient')
        self.train_parser.add_argument(
            '--resize-frames', type=str,
            metavar='WIDTHxHEIGHT', required=False,
            help='Resize observation frames to size WxH before stacking')

        self.play_parser = self.sub_parsers.add_parser(
            'play', help='Watch how the bot plays')
        self.play_parser.add_argument(
            '--gym-env', type=str,
            help='Name of the OpenAI Gym environment, '
                 'like "Pong-v0" or "MsPacman-v0"',
            required=False)
        self.play_parser.add_argument(
            '--algo', type=str,
            help='Which algorithm/network use for playing '
                 '(overrides metadata)',
            required=False)
        self.play_parser.add_argument(
            '--record', type=str, help='Record the play as a movie',
            metavar='PATH.(mp4|gif)', required=False)
        self.play_parser.add_argument(
            '--record-agents-view', type=str, metavar='PATH.(mp4|gif)',
            help='Record a video of what the agent actually sees')
        self.play_parser.add_argument(
            '--no-window', action='store_true',
            help='Do not show the gameplay window')

    def parse_args(self, *args, **kwargs):
        return self.main_parser.parse_args(*args, **kwargs)

    def print_usage(self, *args, **kwargs):
        self.main_parser.print_usage(*args, **kwargs)


class Launcher(metaclass=ABCMeta):
    """
    Abstracts most of the code necessary for training ang running agents
    as well as adjusting typical hyper-parameters via command line.
    Necessary customizations can be made by overriding key methods.
    """

    def __init__(self, parsers: CmdLineParsers):
        self.cmd_line_parsers = parsers
        self.options = parsers.parse_args()

    @abstractmethod
    def construct_agent(self, env: MultiGymEnv,
                        common_agent_kwargs: dict,
                        for_training: bool,
                        metadata: Optional[TrainingMetadata]) -> MultiAgent:
        pass

    @property
    @abstractmethod
    def frames_stacked_per_observation(self) -> int:
        pass

    @abstractmethod
    def extra_metadata(self) -> TrainingMetadata:
        pass

    def run_custom_command(self):
        print('Unsupported command {}\n'.format(self.options.command))
        self.cmd_line_parsers.print_usage()

    def run(self):
        backend_specific_tweaks()

        if self.options.command == 'train':
            history_path = (
                self.options.history_path
                if self.options.history_path
                else (self.options.model_path.rsplit('.', 1)[0]
                      + '_history.txt'))
            resize_frames = (
                tuple(int(i) for i in self.options.resize_frames.split('x'))
                if self.options.resize_frames
                else None)
            self.train(self.options.gym_env, self.options.model_path,
                       history_path, num_envs=self.options.num_envs,
                       num_steps=self.options.time_horizon,
                       normalize_returns=self.options.norm_returns,
                       reward_scale=self.options.reward_scale,
                       resize_frames=resize_frames)
        elif self.options.command == 'play':
            self.play(self.options.gym_env, self.options.model_path,
                      self.options.record, self.options.record_agents_view,
                      not self.options.no_window)
        else:
            self.run_custom_command()

    def create_optimizer(self):
        learning_rate = getattr(self.options, 'lr', 2e-4)
        optimizer = optimizers.RMSprop(
            lr=learning_rate, epsilon=1e-5, rho=0.99, clipnorm=0.5,
            decay=1e-4)
        return optimizer

    @staticmethod
    def exit(status: int, message: str):
        if message:
            print(message, file=sys.stderr if status != 0 else sys.stdout)
        sys.exit(status)

    def play(self, env_name: str, model_path: str, movie_path: Optional[str],
             agent_view_movie_path: str, show_gameplay_window: bool):
        # Loading and initializing the environment
        agent, env = self.prepare_agent_for_playing(env_name, model_path)
        # Preparing to record a video if necessary
        human_view_recorder = (
            None if movie_path is None
            else MovieWriter(movie_path))
        agent_view_recorder = (
            None if agent_view_movie_path is None
            else MovieWriter(agent_view_movie_path))
        # The simulation itself
        observation = env.reset()
        agent.reset_states()
        done = False
        total_score = 0
        frame_num = 0
        while not done:
            frame_num += 1
            if show_gameplay_window:
                env.render()
            if human_view_recorder is not None:
                frame_picture = env.render(mode='rgb_array')[0]
                human_view_recorder.save_frame(frame_picture)
            if agent_view_recorder is not None:
                frame_picture = np.uint8(255 * observation[0, :, :, :, -1])
                agent_view_recorder.save_frame(frame_picture)
            if (human_view_recorder is not None
                    or agent_view_recorder is not None):
                if frame_num > 1:
                    print("\b" * MovieWriter.frame_max_digits, end='')
                else:
                    print('Recorded frames: ', end='')
                print('{:#6}'.format(frame_num), end='', flush=True)

            policy_output, sampled_action, value_output, _ = (
                agent.multi_env_act(observation))
            observation, rewards, done, _ = env.step(sampled_action)
            total_score += rewards.sum()

        print('\nFull episode reward:', total_score)
        if human_view_recorder is not None:
            human_view_recorder.finish()
        if agent_view_recorder is not None:
            agent_view_recorder.finish()

    def prepare_agent_for_playing(self, env_name, model_path):
        metadata = load_training_metadata(model_path)
        if metadata:
            print('Network`s metadata')
            pprint(metadata)
        else:
            print('Network has no metadata')
        if env_name is None:
            if not metadata:
                self.exit(
                    1, 'Cannot read Gym environment name from the saved '
                       'model, please provide it explicitly via '
                       '--gym-env option')
            env_name = metadata['env_name']
        single_env = gym.make(env_name)
        if metadata and metadata.get('resize_frames') is not None:
            single_env = ObservationResizer(
                single_env, tuple(metadata['resize_frames']))
        env = MultiGymImitation(
            single_env,
            stack_observations=self.frames_stacked_per_observation,
            normalize_colors=True)
        env.seed(int(time.time()))
        print('observation space', env.observation_space)
        print('action space', env.action_space)
        # Initialing the agent
        agent = self.construct_agent(
            env,
            {'normalize_returns': metadata.get('normalize_returns', False)},
            for_training=False,
            metadata=metadata)
        if not os.path.exists(model_path):
            raise RuntimeError(f'Model file {model_path} doesn\'t exist')
        agent.load_model(model_path)
        return agent, env

    def train(self, gym_env_name: str, model_path: str, history_path: str,
              num_envs: int = 2, num_steps: int = 5,
              max_samples: int = int(1e9),
              normalize_returns: bool = True, reward_scale: float = 1.0,
              resize_frames: Optional[Tuple[int, int]] = None):
        # Initializing the environment
        def make_worker_env(worker_id):
            worker_env = gym.make(gym_env_name)
            worker_env.seed(worker_id)
            if resize_frames is not None:
                worker_env = ObservationResizer(worker_env, resize_frames)
            return worker_env

        env = ParallelGymEnv(
            env_factory=make_worker_env, num_envs=num_envs,
            stack_observations=self.frames_stacked_per_observation,
            normalize_colors=True)
        print('observation space', env.observation_space)
        print('action space', env.action_space)
        # Initializing the agent
        agent_kwargs = {
            'normalize_returns': normalize_returns,
            'reward_scale': reward_scale}
        agent = self.construct_agent(
            env, agent_kwargs, for_training=True, metadata=None)
        if os.path.exists(model_path):
            agent.load_model(model_path)
        # Prepare metadata that we later save with the model.
        # Some of them (like normalize_returns or resize_frames)
        # are important for re-constructing the model exactly as it was during
        # the training, which allows Keras to load it properly.
        # Some others may play a role of notes, preserving hyper-parameters
        # used for training (like reward_scale).
        metadata = self.extra_metadata()
        metadata.update({
            'env_name': gym_env_name,
            'resize_frames': resize_frames,
            'normalize_returns': normalize_returns,
            'reward_scale': reward_scale})

        # The main loop, sampling from multiple running simulations,
        # and automatically resetting the ones where the episode is over
        episode_rewards = np.zeros((num_envs,))
        observations = env.reset()
        agent.reset_states()
        history_recorder = TrainHistoryRecorder(history_path, num_steps)
        for sample_idx in range(max_samples):
            samples = episode_sampler(num_steps, env, agent, observations)
            loss_info = agent.train_on_sample(samples)
            episode_rewards += np.sum(samples.batch_rewards, axis=-1)
            completed_episodes = samples.batch_dones[:, -1]
            history_recorder.record(
                sample_idx, completed_episodes, episode_rewards)
            observations = env.reset_recently_done()
            agent.reset_particular_agents(completed_episodes)
            episode_rewards[completed_episodes] = 0
            # auto-saving
            print('Time block:', sample_idx, 'Loss value:', loss_info)
            if sample_idx > 0 and sample_idx % int(500 / num_steps) == 0:
                agent.save_model(model_path)
                save_training_metadata(model_path, metadata)
                history_recorder.flush_records()
                print('The model has been saved. Metadata:', end='')
                pprint(metadata)
                print('Hint: training was launched with\n', ' '.join(sys.argv))
