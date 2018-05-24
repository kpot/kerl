from typing import Any, Dict, Optional

from kerl.common.agent import MultiAgent
from kerl.common.launcher import Launcher, CmdLineParsers
from kerl.common.multi_gym_env import MultiGymEnv
from kerl.ppo.agent import PPOMultiAgent
from kerl.ppo.networks import ConvolutionalPPONet, RecurrentPPONet


class PPOLauncher(Launcher):

    frames_stacked_per_observation = 4

    def extra_metadata(self) -> Dict[str, Any]:
        return {'network': self.options.network}

    def construct_agent(self, env: MultiGymEnv, common_agent_kwargs: dict,
                        for_training: bool,
                        metadata: Optional[Dict[str, Any]]) -> MultiAgent:
        network = self.options.network or (metadata or {}).get('network')
        common_agent_kwargs = dict(
            common_agent_kwargs,
            use_gae=(not self.options.no_gae) if for_training else True)
        if for_training:
            common_agent_kwargs['clip_range'] = self.options.clip_range
        if network == 'CNN':
            return PPOMultiAgent(
                env, self.create_optimizer, ConvolutionalPPONet,
                **common_agent_kwargs)
        elif network == 'RNN':
            return PPOMultiAgent(
                env, self.create_optimizer, RecurrentPPONet,
                **common_agent_kwargs)
        else:
            self.exit(1, 'You need to specify a correct name of the network '
                         'architecture via --network option')


if __name__ == '__main__':
    parsers = CmdLineParsers('PPO agent launcher')
    for parser, required in ((parsers.train_parser, True),
                             (parsers.play_parser, False)):
        parser.add_argument(
            '--network', type=str, choices=('CNN', 'RNN'),
            help='Which of the networks we train', required=required)
    parsers.train_parser.add_argument(
        '--no-gae', action='store_true',
        help='Do not use Generalized Advantage Estimator')
    parsers.train_parser.add_argument(
        '--clip-range', type=float, default=0.2,
        help='Clipping parameter (epsilon) for PPO')
    launcher = PPOLauncher(parsers)
    launcher.run()
