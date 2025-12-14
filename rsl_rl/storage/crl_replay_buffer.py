import torch
from typing import Dict, Generator, List

# dict_keys(['actions', 'actor_observations', 'critic_observations', 'dones', 'next_actor_observations', 'next_critic_observations', 'rewards', 'timeouts'])

Transition = Dict[str, torch.Tensor]
Dataset = List[Transition]


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        max_replay_size: int,
        episode_length: int,
        min_replay_size: int = 0,
        skip_size: int = 0,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.max_replay_size = max_replay_size
        self.episode_length = episode_length
        self.min_replay_size = min_replay_size
        self.skip_size = skip_size
        self.device = device

        self.buffers = {}
        self.insert_position = 0
        self.skipped_counter = 0

    @property
    def initialized(self) -> bool:
        return self.insert_position >= self.min_replay_size

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

        # Ignore initial items if needed
        if self.skipped_counter < self.skip_size:
            if unroll_len <= self.skip_size - self.skipped_counter:
                self.skipped_counter += unroll_len
                return

            start_idx = self.skip_size - self.skipped_counter
            dataset = dataset[start_idx:]
            unroll_len = len(dataset)
            self.skipped_counter = self.skip_size

        # Initialize buffers if this is the first insertion
        if not self.buffers:
            for key, value in first_item.items():
                self.buffers[key] = torch.zeros(
                    (self.max_replay_size, *value.shape),
                    dtype=value.dtype,
                    device=self.device,
                )
            # Add traj_id buffer if dones is present
            if "dones" in first_item:
                self.buffers["traj_id"] = torch.zeros(
                    (self.max_replay_size, self.num_envs),
                    dtype=torch.long,
                    device=self.device,
                )
            self.next_traj_ids = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )

        # Stack the new data into tensors: (unroll_len, num_envs, *data_dim)
        new_data = {}
        for key in self.buffers.keys():
            if key == "traj_id":
                continue
            # We assume all items in dataset have the same keys and shapes
            new_data[key] = torch.stack([item[key] for item in dataset], dim=0).to(
                self.device
            )

        # Calculate trajectory IDs
        if "dones" in new_data:
            dones = new_data["dones"].long()
            cumsum_dones = torch.cumsum(dones, dim=0)
            shifted_cumsum = torch.cat(
                [
                    torch.zeros(
                        (1, self.num_envs), device=self.device, dtype=torch.long
                    ),
                    cumsum_dones[:-1],
                ],
                dim=0,
            )
            new_data["traj_id"] = self.next_traj_ids + shifted_cumsum
            self.next_traj_ids += cumsum_dones[-1]

        # Check if we need to shift data to make room
        if self.insert_position + unroll_len > self.max_replay_size:
            shift = unroll_len
            for key in self.buffers:
                self.buffers[key][:-shift] = self.buffers[key][shift:].clone()
                self.buffers[key][-shift:] = new_data[key]
            self.insert_position = self.max_replay_size
        else:
            for key in self.buffers:
                self.buffers[key][
                    self.insert_position : self.insert_position + unroll_len
                ] = new_data[key]
            self.insert_position += unroll_len

    def batch_generator(
        self, batch_size: int, batch_count: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
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
            if batch_size < self.num_envs:
                env_idxs = torch.randint(
                    0, self.num_envs, (batch_size,), device=self.device
                )
            elif batch_size == self.num_envs:
                env_idxs = torch.arange(0, self.num_envs, device=self.device)
            else:
                raise ValueError("batch_size cannot be larger than num_envs")

            # 2. Select Start Times
            # The episode start is sampled from range [0 ... insert_position - episode_length)
            max_start = valid_end - self.episode_length
            # If max_start is 0 (buffer has exactly episode_length items), randint(0, 0) fails, so we handle it.
            if max_start > 0:
                start_idxs = torch.randint(
                    0, max_start, (batch_size,), device=self.device
                )
            else:
                start_idxs = torch.zeros(
                    (batch_size,), dtype=torch.long, device=self.device
                )

            # 3. Construct Advanced Indexing Grids
            # time_idxs: (episode_length, batch_size)
            # We create a column of [0, 1, ..., episode_length-1] and add it to the row of start_idxs
            time_range = torch.arange(
                self.episode_length, device=self.device
            ).unsqueeze(1)
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

    @property
    def sample_count(self) -> int:
        """Returns how many individual time-steps are stored in the storage."""
        return self.insert_position * self.num_envs
