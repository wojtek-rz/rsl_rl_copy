import numpy as np
import torch
from torch import nn, optim
from typing import Any, Callable, Dict, Tuple, Union

from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.modules import Network, GaussianChimeraNetwork, GaussianNetwork
from rsl_rl.storage.torch_replay_buffer import ReplayBuffer
from rsl_rl.storage.storage import Dataset


def similarity_matrix(sa_repr: torch.Tensor, g_repr: torch.Tensor):
    """
    Using L2 distance as similarity metric as per https://arxiv.org/pdf/2408.11052

    Should output a (B, B) matrix where entry (i, j) is the similarity between
    sa_repr[i] and g_repr[j].
    """
    batch_size = sa_repr.size(0)

    diff = (sa_repr.view(batch_size, 1, -1) - g_repr.view(1, batch_size, -1)) ** 2

    return -torch.sum(diff, dim=-1)  # (B, B)


def info_nce_loss(
    sa_repr: torch.Tensor, g_repr: torch.Tensor, temp=1, version="symmetric"
):
    """
    InfoNCE loss (symmetric version)

    Input
    sa_repr: (B, D) - sa_encoder_output (questions)
    g_repr: (B, D) - g_encoder_output (correct answers for the paired question)

    sa_repr[i] is a question for which g_repr[i] is the correct answer.
    As negative answers we use g_repr[j] for j != i .
    """

    logits = similarity_matrix(sa_repr, g_repr) / temp  # (B, B)

    match version:
        case "symmetric":
            return torch.mean(
                2 * torch.diagonal(logits)
                - torch.logsumexp(logits, dim=1)
                - torch.logsumexp(logits, dim=0),
            )
        case "forward":
            return torch.mean(torch.diagonal(logits) - torch.logsumexp(logits, dim=1))
        case "forward_alt":
            labels = torch.arange(logits.size(0)).to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
            return -loss
        case _:
            raise ValueError(f"Unknown version: {version}")


class CRL(AbstractActorCritic):
    """Contrastive Reinforcement Learning algorithm."""

    def __init__(
        self,
        env,
        repr_dim: int = 64,
        goal_dim: int = 2,
        episode_length: int = 1000,
        min_replay_size: int = 1000,
        max_replay_size: int = 10_000,
        crl_loss_version: str = "symmetric",
        sa_hidden_dims: list = [256, 256],
        g_hidden_dims: list = [256, 256],
        temp: float = 1.0,
        action_max: float = 1.0,
        action_min: float = -1.0,
        actor_lr: float = 1e-4,
        actor_noise_std: float = 1.0,
        alpha: float = 0.2,
        alpha_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gradient_clip: float = 1.0,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        target_entropy: float = None,
        logsumexp_penalty: float = 0.1,
        chimera: bool = True,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(
            env,
            action_max=action_max,
            action_min=action_min,
            gamma=gamma,
            **kwargs,
        )

        self.episode_length = episode_length
        self.repr_dim = repr_dim
        self.temp = temp
        self.crl_loss_version = crl_loss_version
        self._gradient_clip = gradient_clip
        self._logsumexp_penalty = logsumexp_penalty

        self.storage = ReplayBuffer(
            num_envs=self.env.num_envs,
            max_replay_size=max_replay_size,  # Storage size is total transitions, we need per env
            episode_length=self.episode_length,
            min_replay_size=min_replay_size,
            device=self.device,
        )
        self._register_serializable("storage")

        assert (
            self._action_max < np.inf
        ), 'Parameter "action_max" needs to be set for CRL.'
        assert (
            self._action_min > -np.inf
        ), 'Parameter "action_min" needs to be set for CRL.'

        self._action_delta = 0.5 * (self._action_max - self._action_min)
        self._action_offset = 0.5 * (self._action_max + self._action_min)

        self.log_alpha = torch.tensor(
            np.log(alpha), dtype=torch.float32
        ).requires_grad_()
        self._target_entropy = target_entropy if target_entropy else -self._action_size
        self._register_serializable("log_alpha", "_gradient_clip")

        # Actor Network
        network_class = GaussianChimeraNetwork if chimera else GaussianNetwork
        self.actor = network_class(
            self._actor_input_size,
            self._action_size,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            std_init=actor_noise_std,
            **self._actor_network_kwargs,
        )

        # Encoders (Critic)
        print(f"{self._critic_input_size=}, {self._actor_input_size=}")

        self.sa_encoder = Network(
            self._critic_input_size,
            self.repr_dim,
            hidden_dims=sa_hidden_dims,
            activations=["elu"] * len(sa_hidden_dims) + ["linear"],
        )

        self.g_encoder = Network(
            goal_dim,
            self.repr_dim,
            hidden_dims=g_hidden_dims,
            activations=["elu"] * len(g_hidden_dims) + ["linear"],
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.critic_optimizer = optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()),
            lr=critic_lr,
        )

        self._register_serializable("actor", "sa_encoder", "g_encoder")
        self._register_serializable(
            "actor_optimizer", "log_alpha_optimizer", "critic_optimizer"
        )

        # Dummy critic for AbstractActorCritic compatibility if needed
        self.critic = self.sa_encoder

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def draw_actions(
        self, obs: torch.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        # breakpoint()
        actor_obs, critic_obs = self._process_observations(obs, env_info)
        action = self._sample_action(actor_obs, compute_logp=False)
        data = {
            "actor_observations": actor_obs.clone(),
            "critic_observations": critic_obs.clone(),
        }
        return action, data

    def get_inference_policy(self, device=None) -> Callable:
        self.to(device)
        self.eval_mode()

        def policy(obs, env_info=None):
            obs, _ = self._process_observations(obs, env_info)
            actions = self._scale_actions(self.actor.forward(obs))
            return actions

        return policy

    def eval_mode(self):
        super().eval_mode()
        self.actor.eval()
        self.sa_encoder.eval()
        self.g_encoder.eval()
        return self

    def train_mode(self):
        super().train_mode()
        self.actor.train()
        self.sa_encoder.train()
        self.g_encoder.train()
        return self

    def to(self, device: str):
        super().to(device)
        self.actor.to(device)
        self.sa_encoder.to(device)
        self.g_encoder.to(device)
        self.log_alpha.to(device)
        return self


    def hindsight_relabel(
        self, obs: torch.Tensor, actions: torch.Tensor, traj_ids: torch.Tensor
    ):
        T, B, _ = obs.shape
        arange = torch.arange(T, device=self.device)
        is_future_mask = (arange[:, None] < arange[None, :]).float()

        discount_mat = self.gamma ** (arange[None, :] - arange[:, None]).float()  # (T, T)
        base_probs = is_future_mask * discount_mat  # (T, T)
        
        goal_indices = []
        chunk_size = 100  # Process environments in chunks to save memory

        for i in range(0, B, chunk_size):
            # Slice traj_ids for the current chunk
            traj_ids_chunk = traj_ids[:, i : i + chunk_size]  # (T, chunk_B)
            chunk_B = traj_ids_chunk.shape[1]

            # Enforce same-trajectory constraint
            # (T, 1, chunk_B) == (1, T, chunk_B) -> (T, T, chunk_B)
            same_traj_chunk = (traj_ids_chunk[:, None, :] == traj_ids_chunk[None, :, :])
            same_traj_chunk = same_traj_chunk.permute(0, 2, 1).float()  # -> (T, chunk_B, T)

            # Broadcast base_probs to chunk shape and combine
            # base_probs is (T, T) -> (T, 1, T)
            probs_chunk = base_probs[:, None, :] * same_traj_chunk  # (T, chunk_B, T)
            
            # Add tiny diagonal probability
            probs_chunk = probs_chunk + torch.eye(T, device=self.device)[:, None, :] * 1e-5

            log_probs_chunk = torch.log(probs_chunk)
            # Flatten for Categorical: (T * chunk_B, T)
            log_probs_flat = log_probs_chunk.reshape(T * chunk_B, T)
            
            goal_index_flat = torch.distributions.Categorical(logits=log_probs_flat).sample()
            goal_index_chunk = goal_index_flat.reshape(T, chunk_B)
            goal_indices.append(goal_index_chunk)

        goal_index = torch.cat(goal_indices, dim=1)  # (T, B)

        idx = goal_index[:-1]  # (T-1, B)
        
        # Prepare a batch index for the environment dimension
        b_idx = torch.arange(B, device=self.device)           # (B,)
        b_idx_expanded = b_idx.unsqueeze(0).expand(T-1, B)  # (T-1, B) 

        # Then gather:
        future_state = obs[idx, b_idx_expanded]    # (T-1, B, ObsDim)

        states = obs[:-1, :]  # (T-1, B, ObsDim)
        actions = actions[:-1, :]  # (T-1, B, ActDim)
        goals = future_state[..., self.env.goal_indices]  # (T-1, B, Goaldim)

        return states, actions, goals
    
    def hindsight_relabel_memory_consuming(
        self, obs: torch.Tensor, actions: torch.Tensor, traj_ids: torch.Tensor
    ):
        T, B, _ = obs.shape
        arange = torch.arange(T, device=self.device)
        is_future_mask = (arange[:, None] < arange[None, :]).float()

        discount_mat = self.gamma ** (arange[None, :] - arange[:, None]).float()  # (T, T)
        probs = is_future_mask * discount_mat 

        # Enforce same-trajectory constraint
        same_traj = (traj_ids[:, None, :] == traj_ids[None, :, :])  # (T, T, B)
        same_traj = same_traj.permute(0, 2, 1).float() # -> (T, B, T)
        # Broadcast probs to per-env shape and combine with same-traj mask
        probs = probs[:, None, :] * same_traj # (T, B, T)
        # Add tiny diagonal probability for numerical safety
        probs = probs + torch.eye(T, device=self.device)[:, None, :] * 1e-5

        log_probs = torch.log(probs)
        log_probs_flat = log_probs.reshape(T * B, T)
        goal_index_flat = torch.distributions.Categorical(logits=log_probs_flat).sample()
        goal_index = goal_index_flat.reshape(T, B)

        idx = goal_index[:-1] # (T-1, B)

        # Prepare a batch index for the environment dimension
        b_idx = torch.arange(B, device=self.device)           # (B,)
        # To use advanced indexing we need to expand b_idx to shape (T-1, B)
        b_idx_expanded = b_idx.unsqueeze(0).expand(T-1, B)  # (T-1, B) 

        # Then gather:
        future_state = obs[idx, b_idx_expanded]    # (T-1, B, ObsDim)
        future_action = actions[idx, b_idx_expanded]  # (T-1, B, ActDim)

        states = obs[:-1, :] # (T-1, B, ObsDim)
        actions = actions[:-1, :] # (T-1, B, ActDim)
        goal = future_state[:, self.env.goal_indices] # (T-1, B, GoalDIm)


        return states, actions, future_state

    def make_batches(self, obs, actions, goals, batch_size):
        """
        obs:     (T-1, B, ObsDim)
        actions: (T-1, B, ActDim)
        goals:   (T-1, B, ObsDim)
        """
        Tm1, B = obs.shape[:2]
        N = Tm1 * B  # total transitions

        # Flatten time and env dims
        obs_flat = obs.reshape(N, -1)  # (N, ObsDim)
        actions_flat = actions.reshape(N, -1)  # (N, ActDim)
        goals_flat = goals.reshape(N, -1)  # (N, ObsDim)

        # Shuffle indices
        idx = torch.randperm(N, device=obs.device)

        obs_flat = obs_flat[idx]
        actions_flat = actions_flat[idx]
        goals_flat = goals_flat[idx]

        # Create batches
        batches = []
        for i in range(0, N, batch_size):
            batches.append(
                (
                    obs_flat[i : i + batch_size],
                    actions_flat[i : i + batch_size],
                    goals_flat[i : i + batch_size],
                )
            )

        return batches

    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        self.storage.append(dataset)

        if not self.initialized:
            return {}

        total_actor_loss = []
        total_alpha_loss = []
        total_critic_loss = []

        for trajectories in self.storage.batch_generator(
            batch_size=self.env.num_envs, batch_count=1
        ):
            # break
            obs = trajectories[
                "critic_observations"
            ]  # (episode_length, batch_size, ObsDim)
            actions = trajectories["actions"]  # (episode_length, batch_size, ActDim)
            traj_ids = trajectories["traj_id"]  # (episode_length, batch_size)

            states, actions, goals = self.hindsight_relabel(obs, actions, traj_ids)

            for b_state, b_actions, b_goals in self.make_batches(
                states, actions, goals, self._batch_size
            ):

                # --- Update Critic (Encoders) ---
                sa_repr = self.sa_encoder(torch.cat([b_state, b_actions], dim=-1))
                g_repr = self.g_encoder(b_goals)

                # InfoNCE Loss
                crl_loss = -info_nce_loss(
                    sa_repr, g_repr, temp=self.temp, version=self.crl_loss_version
                )

                # Logsumexp penalty
                if self._logsumexp_penalty > 0:
                    logits = similarity_matrix(sa_repr, g_repr) / self.temp
                    logsumexp = torch.logsumexp(logits + 1e-6, dim=1)
                    crl_loss = crl_loss + self._logsumexp_penalty * torch.mean(logsumexp ** 2)

                self.critic_optimizer.zero_grad()

                crl_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.sa_encoder.parameters(), self._gradient_clip
                )
                nn.utils.clip_grad_norm_(
                    self.g_encoder.parameters(), self._gradient_clip
                )
                self.critic_optimizer.step()

                # --- Update Actor ---
                # actor_obs = torch.cat([b_state, b_goals], dim=-1)
                actor_obs = torch.cat([b_state, b_goals], dim=-1)
                new_actions, log_prob = self._sample_action(
                    actor_obs, compute_logp=True
                )

                sa_repr_pi = self.sa_encoder(torch.cat([b_state, new_actions], dim=-1))
                g_repr_pi = self.g_encoder(b_goals)

                # Q-value (Energy)
                qf_pi = -torch.sum((sa_repr_pi - g_repr_pi) ** 2, dim=-1)

                actor_loss = (self.alpha.detach() * log_prob - qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- Update Alpha ---
                alpha_loss = (
                    self.alpha * (-log_prob - self._target_entropy).detach()
                ).mean()

                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                total_actor_loss.append(actor_loss.item())
                total_alpha_loss.append(alpha_loss.item())
                total_critic_loss.append(crl_loss.item())

        return {
            "actor": np.mean(total_actor_loss),
            "alpha": np.mean(total_alpha_loss),
            "critic": np.mean(total_critic_loss),
        }

    def _sample_action(self, observation, compute_logp=True):
        mean, std = self.actor.forward(observation, compute_std=True)
        dist = torch.distributions.Normal(mean, std)
        actions = dist.rsample()
        actions_normalized, actions_scaled = self._scale_actions(
            actions, intermediate=True
        )

        if not compute_logp:
            return actions_scaled

        action_logp = dist.log_prob(actions).sum(-1) - torch.log(
            1.0 - actions_normalized.pow(2) + 1e-6
        ).sum(-1)
        return actions_scaled, action_logp

    def _scale_actions(self, actions, intermediate=False):
        actions = actions.reshape(-1, self._action_size)
        action_normalized = torch.tanh(actions)
        action_scaled = action_normalized * self._action_delta + self._action_offset
        if intermediate:
            return action_normalized, action_scaled
        return action_scaled

    def register_terminations(self, terminations):
        pass
