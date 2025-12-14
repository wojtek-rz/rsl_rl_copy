import torch
import unittest
from rsl_rl_link.rsl_rl.storage.crl_replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.num_envs = 3
        self.unroll_len = 2
        self.max_replay_size = 20
        self.episode_length = 5
        self.device = "cpu"
        self.buffer = ReplayBuffer(self.num_envs, self.max_replay_size, self.episode_length, self.device)

    def test_append_and_sample(self):
        # Create dummy data
        # We will insert 10 batches of data (total 20 steps), filling the buffer
        
        for i in range(10):
            dataset = []
            for j in range(self.unroll_len):
                step_val = i * self.unroll_len + j
                transition = {
                    "obs": torch.full((self.num_envs, 1), step_val, dtype=torch.float32),
                    "dones": torch.zeros((self.num_envs,), dtype=torch.float32)
                }
                # Make environment 0 finish an episode at step 8
                if step_val == 8:
                    transition["dones"][0] = 1.0
                
                dataset.append(transition)
            
            self.buffer.append(dataset)

        # Check buffer size
        self.assertEqual(self.buffer.insert_position, 20)
        
        # Check traj_ids
        # Env 0: Steps 0-8 should have ID 0. Step 9 should have ID 1.
        traj_ids = self.buffer.buffers['traj_id']
        print("\nTrajectory IDs (Env 0):", traj_ids[:, 0])
        
        self.assertTrue(torch.all(traj_ids[0:9, 0] == 0)) # Steps 0-8 are ID 0
        self.assertTrue(torch.all(traj_ids[9:, 0] == 1))  # Steps 9+ are ID 1
        
        # Env 1: Should all be ID 0 (no dones)
        self.assertTrue(torch.all(traj_ids[:, 1] == 0))

        # Test Sampling
        batch_gen = self.buffer.batch_generator(batch_size=self.num_envs, batch_count=1)
        batch = next(batch_gen)
        
        # Check shapes
        self.assertEqual(batch['obs'].shape, (self.episode_length, self.num_envs, 1))
        self.assertEqual(batch['traj_id'].shape, (self.episode_length, self.num_envs))
        
        print("\nSampled Batch Obs (Env 0):")
        print(batch['obs'][:, 0].flatten())
        print("Sampled Batch Traj ID (Env 0):")
        print(batch['traj_id'][:, 0].flatten())

        # Verify continuity
        # The sampled observations should be consecutive integers (e.g., 5, 6, 7, 8, 9)
        obs_env0 = batch['obs'][:, 0].flatten()
        diffs = obs_env0[1:] - obs_env0[:-1]
        self.assertTrue(torch.all(diffs == 1.0))

    def test_circular_buffer(self):
        # Fill buffer completely
        for i in range(10):
            dataset = [{"obs": torch.zeros((self.num_envs, 1)), "dones": torch.zeros((self.num_envs,))}] * self.unroll_len
            self.buffer.append(dataset)
            
        # Insert one more batch to trigger shift
        dataset = [{"obs": torch.ones((self.num_envs, 1)), "dones": torch.zeros((self.num_envs,))}] * self.unroll_len
        self.buffer.append(dataset)
        
        # Check that the last elements are ones
        self.assertTrue(torch.all(self.buffer.buffers['obs'][-self.unroll_len:] == 1.0))
        # Check that the first elements are zeros (shifted)
        self.assertTrue(torch.all(self.buffer.buffers['obs'][0] == 0.0))
        
        self.assertEqual(self.buffer.insert_position, 20)

if __name__ == '__main__':
    unittest.main()
