from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kerl.common.agent import MultiAgent
from kerl.common.launcher import CmdLineParsers, Launcher, TrainingMetadata
from kerl.common.multi_gym_env import MultiGymEnv
from kerl.common.utils import MovieWriter
from kerl.wm.agent import WMMultiAgent
from kerl.wm.networks import EnvModelOutput


class WMLauncher(Launcher):
    frames_stacked_per_observation = 1

    def construct_agent(self, env: MultiGymEnv, common_agent_kwargs: dict,
                        for_training: bool,
                        metadata: Optional[TrainingMetadata]) -> MultiAgent:
        latent_dim = (self.options.latent_dim if for_training
                      else metadata['latent_dim_size'])
        world_rnn_units = (self.options.world_rnn_units
                           if for_training else metadata['world_rnn_units'])
        common_agent_kwargs = dict(
            common_agent_kwargs,
            latent_dim_size=latent_dim,
            world_rnn_units=world_rnn_units)
        return WMMultiAgent(env, self.create_optimizer, **common_agent_kwargs)

    def extra_metadata(self) -> TrainingMetadata:
        return {
            'latent_dim_size': self.options.latent_dim,
            'world_rnn_units': self.options.world_rnn_units,
        }

    def run_custom_command(self):
        if self.options.command == 'check':
            self.check(self.options.gym_env, self.options.model_path)

    def check(self, gym_env_name, model_path):
        agent, env = self.prepare_agent_for_playing(gym_env_name, model_path)
        observation = env.reset()
        agent.reset_states()
        done = False
        total_score = 0
        video_recorder = MovieWriter(self.options.record, movie_width=1024)
        obs_shape = env.observation_space.shape
        # Creating small images for spacing frames and labeling them
        frame_labels, frame_spacer = self.video_spacer_and_labels(
            obs_shape,
            ['Past observation', 'After VAE', 'Predicted future', 'Future'])
        while not done:
            prev_observation = observation[0, :, :, :, -1].copy()
            _, _, encoded_prev_observation = agent.vae.compress(
                np.array([prev_observation]))
            decoded_prev_observation = agent.vae.decompress(
                encoded_prev_observation)

            policy_output, sampled_action, value_output, _ = (
                agent.multi_env_act(observation))
            world_predictions = agent.world.model.predict_on_batch(
                x=[np.expand_dims(encoded_prev_observation, axis=1),
                   sampled_action,
                   np.expand_dims(np.zeros_like(encoded_prev_observation),
                                  axis=1)])

            env_predictions = EnvModelOutput(*world_predictions)
            sampled_predictions = agent.world.draw_samples(env_predictions)
            decoded_env_predictions = agent.vae.decompress(
                sampled_predictions.observations[:, 0])

            _, _, compressed_observation = agent.vae.compress(
                np.array([prev_observation]))

            observation, rewards, done, _ = env.step(sampled_action)
            new_observation = observation[0, :, :, :, -1].copy()
            # _, _, encoded_new_observation = agent._vae.compress(
            #     np.array([new_observation]))
            # decoded_new_observation = agent._vae.decompress(
            #     encoded_new_observation)

            base_frame = np.uint8(
                255 *
                np.concatenate(
                    [prev_observation,
                     frame_spacer,
                     decoded_prev_observation[0],
                     frame_spacer,
                     decoded_env_predictions[0],
                     frame_spacer,
                     new_observation],
                    axis=1))
            frame = np.concatenate(
                [frame_labels, base_frame],
                axis=0)
            video_recorder.save_frame(frame)
            print('Recorded frames:', video_recorder.frame_counter)

            total_score += rewards.sum()
        print('Full episode reward:', total_score)
        video_recorder.finish()

    @staticmethod
    def video_spacer_and_labels(obs_shape, labels):
        frame_spacer = np.zeros((obs_shape[0], 11, obs_shape[2]))
        font = ImageFont.load_default()
        sample_text_size = font.getsize('SAMPLE')
        frame_labels = Image.new(
            'RGB',
            (3 * frame_spacer.shape[1] + 4 * obs_shape[1],
             sample_text_size[1] + 4),
            (0, 0, 0))
        draw = ImageDraw.Draw(frame_labels, 'RGB')
        for i, label in enumerate(labels):
            draw.text(
                (i * (obs_shape[1] + frame_spacer.shape[1]) + 10, 2), label,
                font=font, fill=(255, 255, 255))
        frame_labels = np.array(frame_labels)
        return frame_labels, frame_spacer


if __name__ == '__main__':
    parsers = CmdLineParsers('WorldModel agent launcher')
    check_parser = parsers.sub_parsers.add_parser(
        'check', help='Check how correctly WM sees the world')
    check_parser.add_argument(
        '--gym-env', type=str,
        help='Name of the OpenAI Gym environment, '
             'like "Pong-v0" or "MsPacman-v0"',
        required=False)
    check_parser.add_argument(
        '--record', type=str, help='Record the play as a movie',
        metavar='PATH.(mp4|gif)', required=True)
    # Extra arguments for training
    parsers.train_parser.add_argument(
        '--latent-dim', type=int, default=128,
        help='Number of latent space dimensions for VAE part of the model')
    parsers.train_parser.add_argument(
        '--world-rnn-units', type=int, default=512,
        help='Number of hidden units inside MDN-RNN model')
    launcher = WMLauncher(parsers)
    launcher.run()
