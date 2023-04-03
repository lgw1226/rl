"""Include utility functions for RL algorithms."""

import sys; sys.path.append('./')

from itertools import count
from typing import Tuple

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from blocks import Agent, Environment


def test(agent: Agent,
         environment: Environment,
         n_episode: int = 1
         ) -> torch.Tensor:
    """Test agent, return average return per episode."""
    
    ob = environment.reset()
    total_rwd = 0
    
    for _ in range(n_episode):  # for each episode

        for _ in count():  # for each timestep

            ac = agent.get_ac(ob)
            ob, rwd, done = environment.step(ac)

            total_rwd += rwd

            if done:
                ob = environment.reset()
                break

    return total_rwd / n_episode

def plot_returns(epoch_returns: torch.Tensor,
                 show_result: bool = False
                 ) -> None:
    """
    Plot return per each epoch.

    Parameters
    ----------
    epoch_returns : torch.Tensor
        Tensor containing return from each epoch
    show_result : bool = False
        If True, plot with title "Result"
    """

    epoch_returns_np = epoch_returns.to('cpu').numpy()

    plt.figure(1)

    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Epoch")
    plt.ylabel("Return")
    
    plt.plot(epoch_returns_np)

    if len(epoch_returns) >= 100:
        means = epoch_returns.unfold(0, 100, 1).mean(1).view(-1).to('cpu')
        means = torch.cat((torch.ones(99) * means[0], means))

        plt.plot(means.numpy())
    
    plt.pause(0.01)

def tuple_of_tensors_to_tensor(tuple_of_tensors: Tuple[torch.Tensor, ...]
                               ) -> torch.Tensor:
    """
    Convert tuple of torch.Tensors to torch.Tensor.

    Parameters
    ----------
    tuple_of_tensors : Tuple[torch.Tensor, ...]
        Tuple consisting of tensors
    
    Returns
    -------
    torch.Tensor
    """

    return torch.stack(list(tuple_of_tensors), dim=0)
