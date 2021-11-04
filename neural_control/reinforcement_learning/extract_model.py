from stable_baselines3.sac import SAC
import torch

class SACPolicyModule(torch.nn.Module):
    '''
    :param agent_path: the path from where to load the reinforcement learning agent
    '''
    def __init__(self, agent_path):
        super().__init__()
        agent = SAC.load(agent_path)

        self.actor = agent.policy.actor
        self.obs_shape = agent.observation_space.shape

    '''
    :param obs: pytorch tensor of shape (N, 164)
    '''
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, deterministic=True)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    policy = SACPolicyModule(args.path).to(args.device)
    n = 1 # Batch dimension
    obs = torch.zeros(n, *policy.obs_shape).to(args.device)
    print(policy(obs))
    print("Model has %i parameters" % sum(p.numel() for p in policy.parameters()))
