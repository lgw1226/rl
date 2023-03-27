"""Include utility functions for RL algorithms."""

from typing import Tuple

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

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

def save_nn(neural_net: nn.Module, save_path: str, filename : str):
    """
    Save neural network parameters to the given path.

    Parameters
    ----------
    neural_net : nn.Module
        Neural network whose parameters (state_dict) are saved
    save_path : str
        Path to save parameters to
    filename : str
        Name of the file to save
    """

    torch.save(neural_net.state_dict(), save_path + filename + '.pt')

def load_nn(neural_net: nn.Module, load_path : str, filename : str):
    """
    Load neural network parameters from the given path.

    Parameters
    ----------
    neural_net : nn.Module
        Neural network whose parameters (state_dict) are loaded
    save_path : str
        Path to load parameters from
    filename : str
        Name of the file to load
    """

    neural_net.load_state_dict(torch.load(load_path + filename + '.pt'))

def mp4_to_gif(mp4_path: str, gif_path: str, filename: str):
    """
    Convert .mp4 to .gif.

    Parameters
    ----------
    mp4_path : str
        Path to .mp4 file which is converted
    gif_path : str
        Path to save result .gif file
    filename : str
        Same for both .mp4 and .gif files
    """

    video_clip = VideoFileClip(mp4_path + filename + '.mp4')
    video_clip.write_gif(gif_path + filename + '.gif')

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
