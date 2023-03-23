"""Agent class which wraps gymnasium outputs into pytorch tensors."""

from typing import Union

import torch
import torch.cuda as cuda

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


# cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class Agent():
    """
    The most basic class which all the other agents inherit.

    Attributes
    ----------
    self.name : str
        Name of the agent
    self.ob_space : gym.spaces.Box
        Observation space of the environment
    self.ac_space : gym.spaces.Box | gym.spaces.Discrete
        Action space of the environment
    self.is_discrete : bool
        True if action space is discrete, else False
    self.dim_ob : int
        Dimension of the observation space
    self.dim_ac : int
        Dimension of the action space
    self.n_ac : int | None
        The number of actions the agent can choose - only when is_discrete True

    Methods
    -------
    __repr__()
        Return summary of the agent information
    get_ac()
        Sample and return random action
    """
    def __init__(self,
                 ob_space: Box,
                 ac_space: Union[Box, Discrete]
                 ) -> None:        
        """
        Parameters
        ----------
        ob_space : gym.spaces.Box
            Observation space of the environment
        ac_space : gym.spaces.Box | gym.spaces.Discrete
            Action space of the environment
        """

        self.name = "Agent"
        
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.is_discrete = (type(self.ac_space) == Discrete)

        self.dim_ob = self.ob_space.shape[0]
        
        if self.is_discrete:
            self.dim_ac = 1
            self.n_ac = self.ac_space.n
        else:
            self.dim_ac = self.ac_space.shape[0]
            self.n_ac = None

    def __repr__(self) -> str:
        """Return summary of the agent information."""
        ret_str = f"Agent Name: {self.name}, Discrete: {self.is_discrete}"

        if self.is_discrete:
            ret_str += f", No. of Actions: {self.n_ac}"
        else:
            ret_str += f", Action Space Dimension: {self.dim_ac}"

        return ret_str

    def get_ac(self) -> torch.Tensor:
        """
        Sample and return random action.
        
        Returns
        -------
        torch.Tensor
            Sampled random action of shape `[self.dim_ac]`
        """
        ac = self.ac_space.sample()
        
        if self.is_discrete:
            ac_tensor = torch.tensor([ac], dtype=torch.int32, device=device)
        else:
            ac_tensor = torch.tensor(ac, dtype=torch.float32, device=device)

        return ac_tensor


if __name__ == "__main__":
    
    env_discrete = gym.make("CartPole-v1")
    agent_discrete = Agent(env_discrete.observation_space, env_discrete.action_space)

    print(" Discrete Case ".center(60, '-'))
    print(agent_discrete)

    [print(agent_discrete.get_ac()) for i in range(3)]

    print()

    env_continuous = gym.make("BipedalWalker-v3")
    agent_continuous = Agent(env_continuous.observation_space, env_continuous.action_space)

    print(" Continuous Case ".center(60, '-'))
    print(agent_continuous)

    [print(agent_continuous.get_ac()) for i in range(3)]
