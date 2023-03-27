"""Contain utility classes."""

import os
from datetime import datetime
from typing import Tuple


class LogDirs():
    """
    Contain directory paths required to logging.

    Attributes
    ----------
    self.env_name : str
    self.repr_name : str
        Representitive name for each algorithm (ex: DDPG, SAC, VPG, ...)
    self.timestamp : str
        Datetime information of format (YMD-HMS)
    self.base_dir : str
    self.tensorboard_dir : str
    self.modelstate_dir : str
    self.plot_dir : str

    Methods
    -------
    get_path()
        Return tensorboard, modelstate, plot paths
    """
    def __init__(self, env_name: str, dir_name: str) -> None:
        """Create directories to save tensorboard runs, model states, and plots."""

        self.env_name = env_name
        self.dir_name = dir_name
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # "./log/DDPG/CartPole-v1/20230324-135550/"
        self.base_dir = f"./log/{self.dir_name}/{self.env_name}/{self.timestamp}/"
        
        # "./log/DDPG/CartPole-v1/20230324-135550/tb/"
        self.tensorboard_dir = self.base_dir + "tb/"
        if os.path.exists(self.tensorboard_dir): raise FileExistsError
        else: os.makedirs(self.tensorboard_dir)

        # "./log/DDPG/CartPole-v1/20230324-135550/ms/"
        self.modelstate_dir = self.base_dir + "ms/"
        if os.path.exists(self.modelstate_dir): raise FileExistsError
        else: os.makedirs(self.modelstate_dir)

        # "./log/DDPG/CartPole-v1/20230324-135550/plt/"
        self.plot_dir = self.base_dir + "plt/"
        if os.path.exists(self.plot_dir): raise FileExistsError
        else: os.makedirs(self.plot_dir)

    def get_paths(self) -> Tuple[str, str, str]:
        """Return base, tensorboard, modelstate, plot paths."""

        return self.base_dir, self.tensorboard_dir, self.modelstate_dir, self.plot_dir


if __name__ == "__main__":

    logdirs = LogDirs("CartPole-v1", "DDPG")
    base_dir, tb_dir, ms_dir, plt_dir = logdirs.get_paths()

    print(base_dir)
    print(tb_dir)
    print(ms_dir)
    print(plt_dir)
