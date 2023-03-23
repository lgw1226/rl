"""Environment class which wraps gymnasium outputs into pytorch tensors."""

from itertools import count
from typing import Union, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import torch
import torch.cuda as cuda

device = 'cuda' if cuda.is_available else 'cpu'


class Environment():
    """
    Wrap gym inputs & outputs into pytorch tensors.
    
    Attributes
    ----------
    self.name : str
        Name of the environment
    self.render_mode : str | None
        Render mode of the gym environment
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
        Return summary of the agent information.
    step()
        Take an action in the environment, return observation, reward, done signal.
    reset()
        Reset environment, return initial observation.
    get_spaces()
        Return observation and action spaces.
    """
    def __init__(self,
                 env_name: str,
                 render_mode: Union[str, None] = None
                 ) -> None:
        """
        Parameters
        ----------
        env_name : str
            Name of the environment passed to gym.make
        render_mode : str | None
            Render mode of the gym environment
        """
        
        self.name = env_name
        self.render_mode = render_mode

        self._env = gym.make(self.name, render_mode=self.render_mode)  # hidden

        self.ob_space = self._env.observation_space
        self.ac_space = self._env.action_space

        self.is_discrete = (type(self.ac_space) == Discrete)  # if False -> action is continuous

        self.dim_ob = self.ob_space.shape[0]
        
        if self.is_discrete:
            self.dim_ac = 1
            self.n_ac = self.ac_space.n
        else:
            self.dim_ac = self.ac_space.shape[0]
            self.n_ac = None

    def __repr__(self) -> str:
        """Return summary of the agent information."""

        ret_str = f"Environment Name: {self.name}\nDiscrete: {self.is_discrete} "

        if self.is_discrete:
            ret_str += f"\nNo. of Actions: {self.n_ac}"
        else:
            ret_str += f"\nAction Space Dimension: {self.dim_ac}"

        ret_str += f"\nObservation Space Dimension: {self.dim_ob}"

        return ret_str

    def step(self,
             ac: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take an action in the environment, return observation, reward, done signal.

        Parameters
        ----------
        ac : torch.Tensor
            Action chosen from the agent policy

        Returns
        -------
        tuple
            tensor(ob) -> shape `[self.dim_ob]`
            tensor([rwd]) -> shape `[1]`
            tensor([done]) -> shape `[1]` (0 or 1)
        """
        if self.is_discrete: ac = ac.item()

        ob, rwd, terminated, truncated, _ = self._env.step(ac)

        ob_t = torch.tensor(ob, dtype=torch.float32, device=device)
        rwd_t = torch.tensor([rwd], dtype=torch.float32, device=device)
        done_t = torch.tensor([terminated or truncated], dtype=torch.float32, device=device)

        return ob_t, rwd_t, done_t  # tensor(ob), tensor([rew]), tensor([done (as 0 or 1)])
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment, return initial observation.

        Returns
        -------
        torch.Tensor
            Initial observation of shape `[self.dim_ob]`
        """
        ob, _ = self._env.reset()

        return torch.tensor(ob, dtype=torch.float32, device=device)
    
    def get_spaces(self) -> Tuple[Box, Union[Box, Discrete]]:
        """
        Return observation and action spaces.

        Return
        ------
        gym.spaces.Box
            Observation space of the environment
        gym.spaces.Box | gym.spaces.Discrete
            Action space of the environments
        """
        return self.ob_space, self.ac_space

if __name__ == "__main__":

    env = Environment("BipedalWalker-v3", render_mode="human")
    print(env)

    n_episode = 5
    print(f"Playing {n_episode} episodes with random actions...")

    ob = env.reset()

    for episode in range(n_episode):

        for t in count():

            ac = env.ac_space.sample()
            ob, _, done = env.step(ac)

            if done:
                ob = env.reset()
                break
