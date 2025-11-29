import torch
from typing import Generator, Dict, List
from rsl_rl.storage.storage import Dataset, Transition, Storage

class TrajectorySamplingStorage(Storage):
    """
    Storage that maintains a buffer of transitions and allows sampling full trajectories
    of a fixed length (episode_length).
    
    This mimics the behavior of TrajectoryUniformSamplingQueue in the JAX implementation,
    where we sample (num_envs, episode_length, ...) batches.
    """

    def __init__(self, num_envs: int, max_replay_size: int, episode_length: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.max_replay_size = max_replay_size  # Total number of time steps per env
        self.episode_length = episode_length
        self.device = device

        self._data = {}
        self._insert_idx = 0  # Current insertion index (per env)
        self._size = 0        # Current size (per env)
        self._full = False

    def append(self, dataset: Dataset) -> None:
        """
        Appends a dataset of transitions to the storage.
        dataset is a list of transitions (dicts).
        We assume the dataset comes from `num_envs` environments.
        """
        # Convert list of dicts to dict of tensors/lists
        # Assuming dataset is a list of transitions, where each transition is a dict
        # and values are tensors of shape (num_envs, ...)
        
        if not dataset:
            return

        # Process first item to initialize buffers if needed
        first_item = dataset[0]
        for key, value in first_item.items():
            if key not in self._data:
                # Initialize buffer: (max_replay_size, num_envs, *feature_dim)
                # value shape is (num_envs, *feature_dim)
                self._data[key] = torch.zeros(
                    (self.max_replay_size, self.num_envs, *value.shape[1:]),
                    dtype=value.dtype,
                    device=self.device
                )

        # Append data
        for transition in dataset:
            for key, value in transition.items():
                self._data[key][self._insert_idx] = value.to(self.device)
            
            self._insert_idx = (self._insert_idx + 1) % self.max_replay_size
            self._size = min(self._size + 1, self.max_replay_size)
            if self._size == self.max_replay_size:
                self._full = True

    def sample_trajectories(self) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of trajectories.
        Returns a dictionary where each value has shape (num_envs, episode_length, *feature_dim).
        
        The sampling logic:
        For each environment, we sample a random start index such that we can retrieve
        a contiguous sequence of length `episode_length`.
        """
        if self._size < self.episode_length:
            raise ValueError(f"Not enough data to sample trajectories. Size: {self._size}, Required: {self.episode_length}")

        # Determine valid start indices
        # We treat the buffer as a circular buffer.
        # However, to simplify "contiguous" sampling without wrapping logic in the return tensor,
        # we can just sample start indices carefully or implement wrapping.
        # The JAX implementation seems to use `mode='wrap'` which implies circular buffer logic.
        
        # Let's implement circular buffer retrieval.
        
        # We want to sample `num_envs` trajectories (one per env) or just a batch of trajectories?
        # The JAX code samples `num_envs` indices from `num_envs` available (without replacement),
        # effectively shuffling environments, and then for each env samples a time sequence.
        # Here we can just sample one trajectory per environment for simplicity, or shuffle envs.
        
        # Random start index for each environment
        # Valid start range: [0, _size - 1]
        # But we need `episode_length` steps.
        # If buffer is full (circular), any start index is valid if we handle wrapping.
        # If not full, valid start indices are [0, _size - episode_length].
        
        if self._full:
            # Buffer is full, circular.
            # We can start anywhere.
            start_indices = torch.randint(0, self.max_replay_size, (self.num_envs,), device=self.device)
        else:
            # Buffer not full, linear from 0 to _size.
            # Valid start indices: [0, _size - episode_length]
            start_indices = torch.randint(0, self._size - self.episode_length + 1, (self.num_envs,), device=self.device)

        # Construct indices: (episode_length, num_envs)
        # indices[t, env] = (start_indices[env] + t) % max_size
        
        time_offsets = torch.arange(self.episode_length, device=self.device).unsqueeze(1) # (L, 1)
        indices = (start_indices.unsqueeze(0) + time_offsets) % self.max_replay_size # (L, N)
        
        batch = {}
        for key, buffer in self._data.items():
            # buffer: (T, N, ...)
            # We want to gather according to indices.
            # indices: (L, N)
            # We need to expand indices to match feature dims
            
            # Advanced indexing:
            # buffer[indices, torch.arange(num_envs)]
            
            # indices is (L, N). We want result (L, N, ...)
            # buffer is (T, N, ...)
            
            # Let's use gather or simple indexing.
            # buffer[indices] would treat indices as indexing the first dim.
            # But we also need to match the second dim (env index).
            
            env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(0).expand(self.episode_length, -1)
            
            # buffer[indices, env_indices] -> (L, N, ...)
            sampled = buffer[indices, env_indices]
            
            # Transpose to (N, L, ...) to match expected batch format (Batch, Time, ...)
            batch[key] = sampled.transpose(0, 1)
            
        return batch

    @property
    def initialized(self) -> bool:
        return self._size >= self.episode_length

    @property
    def sample_count(self) -> int:
        return self._size * self.num_envs
