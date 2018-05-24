from typing import Any, Dict, Optional

from kerl.a2c.agent import A2CMultiAgent
from kerl.a2c.networks import ConvolutionalA2CNet, RecurrentA2CNet
from kerl.common.agent import MultiAgent
from kerl.common.launcher import Launcher, CmdLineParsers
from kerl.common.multi_gym_env import MultiGymEnv


class A2CLauncher(Launcher):

    frames_stacked_per_observation = 4

    def extra_metadata(self) -> Dict[str, Any]:
        return {'network': self.options.network}

    def construct_agent(self, env: MultiGymEnv, common_agent_kwargs: dict,
                        for_training: bool,
                        metadata: Optional[Dict[str, Any]]) -> MultiAgent:
        network = self.options.network or (metadata or {}).get('network')
        if network == 'CNN':
            return A2CMultiAgent(
                env, self.create_optimizer, ConvolutionalA2CNet,
                **common_agent_kwargs)
        elif network == 'RNN':
            return A2CMultiAgent(
                env, self.create_optimizer,
                RecurrentA2CNet, **common_agent_kwargs)
        else:
            self.exit(1, 'You need to specify a correct name of the network '
                         'architecture via --network option')


if __name__ == '__main__':
    parsers = CmdLineParsers('A2C agent launcher')
    for parser, required in ((parsers.train_parser, True),
                             (parsers.play_parser, False)):
        parser.add_argument(
            '--network', type=str, choices=('CNN', 'RNN'),
            help='Which of the networks we train', required=required)
    launcher = A2CLauncher(parsers)
    launcher.run()
