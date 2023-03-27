import torch
import torch.nn as nn

from moviepy.editor import VideoFileClip


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