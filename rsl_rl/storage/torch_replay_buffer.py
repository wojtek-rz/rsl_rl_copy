import torch
from typing import Dict, Generator, List

from rsl_rl.storage import Storage

# prev_obs, prev_obs_info, actions, rewards, next_obs, next_obs_info, dones, data
Transition = Dict[str, torch.Tensor]
Dataset = List[Transition]


class ReplayBuffer(Storage):
    def __init__(self, num_envs: int, max_replay_size: int, episode_length: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.max_replay_size = max_replay_size
        self.episode_length = episode_length
        self.device = device

        self.buffers = {}
        self.insert_position = 0

    def append(self, dataset: Dataset) -> None:
        """Adds transitions to the storage.

        Args:
            dataset (Dataset): The transitions to add to the storage.
                               Expected to be a list of transitions (dicts) representing a chunk of time (unroll_len).
        """
        if not dataset:
            return

        # Determine unroll_len from the input dataset size
        unroll_len = len(dataset)
        first_item = dataset[0]

        # Initialize buffers if this is the first insertion
        if not self.buffers:
            for key, value in first_item.items():
                # value shape is expected to be (num_envs, *data_dim)
                # buffer shape: (max_replay_size, num_envs, *data_dim)
                self.buffers[key] = torch.zeros(
                    (self.max_replay_size, *value.shape),
                    dtype=value.dtype,
                    device=self.device
                )

        # Stack the new data into tensors: (unroll_len, num_envs, *data_dim)
        new_data = {}
        for key in self.buffers.keys():
            # We assume all items in dataset have the same keys and shapes
            new_data[key] = torch.stack([item[key] for item in dataset], dim=0).to(self.device)
            

        # Check if we need to shift data to make room
        if self.insert_position + unroll_len > self.max_replay_size:
            # The buffer is full or will overflow.
            # We implement the shifting logic: shift data to the left, append new data at the end.
            # Note: User specified `_data[:unroll_len] = _data[unroll_len:]` which implies a shift,
            # but dimensionally `_data[:-unroll_len] = _data[unroll_len:]` is the correct operation
            # to shift the tail to the head.
            
            shift = unroll_len
            
            for key in self.buffers:
                # Shift existing data to the left (backwards)
                # This overwrites the oldest data at the beginning
                self.buffers[key][:-shift] = self.buffers[key][shift:].clone()
                
                # Append the new data at the end
                self.buffers[key][-shift:] = new_data[key]
            
            # The buffer is now full, so insert_position stays at the end
            self.insert_position = self.max_replay_size
        else:
            # There is enough space, just append at the current position
            for key in self.buffers:
                self.buffers[key][self.insert_position : self.insert_position + unroll_len] = new_data[key]
            
            self.insert_position += unroll_len

    def batch_generator(self, batch_size: int, batch_count: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generates a batch of transitions.

        Args:
            batch_size (int): The number of trajectories (environments) to sample per batch.
            batch_count (int): The number of batches to generate.

            The batch_size should be from the original implementation equal to num_envs
            The batch_count should be 1
        Returns:
            A generator that yields a dictionary of tensors of shape (episode_length, batch_size, *data_dim).
        """
        valid_end = self.insert_position
        
        # We need at least episode_length data to sample a full sequence
        if valid_end < self.episode_length:
            return

        for _ in range(batch_count):
            # 1. Select Environments
            # Sample `batch_size` environment indices randomly
            env_idxs = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

            # 2. Select Start Times
            # The episode start is sampled from range [0 ... insert_position - episode_length)
            max_start = valid_end - self.episode_length
            # If max_start is 0 (buffer has exactly episode_length items), randint(0, 0) fails, so we handle it.
            if max_start > 0:
                start_idxs = torch.randint(0, max_start, (batch_size,), device=self.device)
            else:
                start_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

            # 3. Construct Advanced Indexing Grids
            # time_idxs: (episode_length, batch_size)
            # We create a column of [0, 1, ..., episode_length-1] and add it to the row of start_idxs
            time_range = torch.arange(self.episode_length, device=self.device).unsqueeze(1)
            time_idxs = start_idxs.unsqueeze(0) + time_range

            # env_idxs_expanded: (episode_length, batch_size)
            # We expand the chosen env indices to match the time dimension
            env_idxs_expanded = env_idxs.unsqueeze(0).expand(self.episode_length, -1)

            # 4. Gather Data
            batch = {}
            for key, buf in self.buffers.items():
                # buf shape: (max_replay_size, num_envs, *data_dim)
                # We use the indices to extract the batch
                # Result shape: (episode_length, batch_size, *data_dim)
                batch[key] = buf[time_idxs, env_idxs_expanded]

            yield batch

    def sample_count(self) -> int:
        """Returns how many individual time-steps are stored in the storage."""
        return self.insert_position * self.num_envs
