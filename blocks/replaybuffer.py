import random
from collections import namedtuple, deque

import torch.cuda as cuda

from utils import tuple_of_tensors_to_tensor


device = 'cuda' if cuda.is_available() else 'cpu'


# transition
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer(object):
    """Replay buffer which saves transition: (s, a, r, s', d)."""
    def __init__(self, maxlen: int) -> None:
        """Create a replay buffer with given max length."""
        self.memory = deque([], maxlen=maxlen)

    def push(self, *args) -> None:
        """Save a transition.
        
        Parameters
        ----------
        state : torch.Tensor
        action : torch.Tensor
        reward : torch.Tensor
        next_state : torch.Tensor
        done : torch.Tensor
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        """Sample transitions, return sampled transitions.
        
        Returns
        -------
        list
            `batch_size` state, action, reward, next_state, done
        """

        batch = [*zip(*random.sample(self.memory, batch_size))]
        batch_t = [tuple_of_tensors_to_tensor(e).to(device) for e in batch]

        return batch_t
    
    def __len__(self) -> int:
        """Return the number of transitions in the replay buffer."""
        return len(self.memory)