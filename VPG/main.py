import sys; sys.path.append('./')

from blocks import Agent, Environment
from utils import test


if __name__ == "__main__":

    env_name = "CartPole-v1"

    env = Environment(env_name)
    agent = Agent(env.ob_space, env.ac_space)

    print(test(agent, env, 3))