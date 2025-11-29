import numpy as np
import torch
from torch import nn, optim
from typing import Any, Callable, Dict, Tuple, Union

from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.modules import Network, GaussianChimeraNetwork, GaussianNetwork
from rsl_rl.storage.trajectory_storage import TrajectorySamplingStorage
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


def info_nce_loss(sa_repr: torch.Tensor, g_repr: torch.Tensor, temp=1, version="symmetric"):
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
                2 * torch.diagonal(logits) - torch.logsumexp(logits, dim=1) - torch.logsumexp(logits, dim=0),
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
        goal_dim: int = 2,
        repr_dim: int = 64,
        sa_hidden_dims: list = [256, 256],
        g_hidden_dims: list = [256, 256],
        temp: float = 1.0,
        crl_loss_version: str = "symmetric",
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
        storage_initial_size: int = 0,
        storage_size: int = 100000,
        target_entropy: float = None,
        chimera: bool = True,
        gamma: float = 0.99,
        **kwargs
    ):
        self.episode_length = kwargs.pop("episode_length", 1000)
        super().__init__(env, action_max=action_max, action_min=action_min, gamma=gamma, **kwargs)

        self.goal_dim = goal_dim
        self.repr_dim = repr_dim
        self.temp = temp
        self.crl_loss_version = crl_loss_version
        self._gradient_clip = gradient_clip

        self.storage = TrajectorySamplingStorage(
            num_envs=self.env.num_envs,
            max_replay_size=storage_size // self.env.num_envs, # Storage size is total transitions, we need per env
            episode_length=self.episode_length,
            device=self.device
        )
        self._register_serializable("storage")

        assert self._action_max < np.inf, 'Parameter "action_max" needs to be set for CRL.'
        assert self._action_min > -np.inf, 'Parameter "action_min" needs to be set for CRL.'

        self._action_delta = 0.5 * (self._action_max - self._action_min)
        self._action_offset = 0.5 * (self._action_max + self._action_min)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32).requires_grad_()
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
            **self._actor_network_kwargs
        )

        # Encoders (Critic)
        # State-Action Encoder: (State + Action) -> Repr
        # State dim = Obs dim - Goal dim
        self.state_dim = self._actor_input_size - self.goal_dim
        
        self.sa_encoder = Network(
            self.state_dim + self._action_size,
            self.repr_dim,
            hidden_dims=sa_hidden_dims,
            activations=["elu"] * len(sa_hidden_dims) + ["linear"]
        )
        
        self.g_encoder = Network(
            self.goal_dim,
            self.repr_dim,
            hidden_dims=g_hidden_dims,
            activations=["elu"] * len(g_hidden_dims) + ["linear"]
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.critic_optimizer = optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()), 
            lr=critic_lr
        )

        self._register_serializable("actor", "sa_encoder", "g_encoder")
        self._register_serializable("actor_optimizer", "log_alpha_optimizer", "critic_optimizer")

        # Dummy critic for AbstractActorCritic compatibility if needed
        self.critic = self.sa_encoder 

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def draw_actions(self, obs: torch.Tensor, env_info: Dict[str, Any]) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        actor_obs, _ = self._process_observations(obs, env_info)
        action = self._sample_action(actor_obs, compute_logp=False)
        data = {"actor_observations": actor_obs.clone(), "critic_observations": actor_obs.clone()}
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

    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        breakpoint()

        self.storage.append(dataset)
        if not self.initialized:
            return {}

        total_actor_loss = []
        total_alpha_loss = []
        total_critic_loss = []

        # We perform updates based on batch_count, sampling trajectories each time
        for _ in range(self._batch_count):
            # Sample trajectories: (num_envs, episode_length, ...)
            trajectories = self.storage.sample_trajectories()
            
            # Flatten trajectories for processing if needed, or process as batch
            # JAX implementation flattens and then creates contrastive pairs.
            # Here we have (N, L, D).
            
            # We need to construct positive and negative pairs.
            # Positive pairs: (s_t, a_t) and g_t (where g_t is from the future of the same trajectory)
            # Negative pairs: (s_t, a_t) and g' (from other trajectories or other times)
            
            # Let's implement the logic from JAX `flatten_batch` roughly.
            # It samples a future goal for each timestep.
            
            
            obs = trajectories["actor_observations"] # (N, L, ObsDim)
            actions = trajectories["actions"] # (N, L, ActDim)
            dones = trajectories["dones"] # (N, L)
            
            state = obs[..., :self.state_dim] # (N, L, StateDim)
            
            # Sample future goals
            # For each t in [0, L-1], sample k > t.
            # We can vectorize this.
            
            N, L, _ = state.shape
            
            states_list = []
            actions_list = []
            goals_list = []

            # Precompute cumulative dones for fast range checking
            # cum_dones[n, t] = sum(dones[n, 0..t])
            cum_dones = torch.cumsum(dones.float(), dim=1)
            
            for t in range(L - 1):
                # Current state/action at time t across all N envs
                s_t = state[:, t]
                a_t = actions[:, t]
                
                range_size = L - 1 - t
                
                # Geometric distribution sampling
                # P(offset=k) \propto gamma^k for k in 1..range_size
                probs = self.gamma ** torch.arange(1, range_size + 1, device=self.device)
                probs = probs / probs.sum()
                
                # Sample offsets
                offsets = torch.multinomial(probs, num_samples=N, replacement=True)
                # offsets are 0-indexed, so add 1
                offsets = offsets + 1
                
                k = t + offsets
                
                g_t = state[torch.arange(N, device=self.device), k]
                
                g_t = g_t[:, :self.goal_dim]

                # Check for dones in range [t, k-1]
                # If done occurred at step i, then transition i -> i+1 is invalid.
                # So if any done in [t, k-1] is 1, then the path from t to k crosses a reset.
                
                end_idx = k - 1
                start_idx = t - 1
                
                val_end = cum_dones[torch.arange(N, device=self.device), end_idx]
                if start_idx >= 0:
                    val_start = cum_dones[:, start_idx]
                else:
                    val_start = torch.zeros(N, device=self.device)
                    
                has_done = (val_end - val_start) > 0.5 # Use 0.5 threshold for float comparison
                valid_mask = ~has_done
                
                if valid_mask.sum() > 0:
                    states_list.append(s_t[valid_mask])
                    actions_list.append(a_t[valid_mask])
                    goals_list.append(g_t[valid_mask])
                
            # Concatenate to create batch
            if len(states_list) > 0:
                batch_state = torch.cat(states_list, dim=0) # (TotalValid, StateDim)
                batch_actions = torch.cat(actions_list, dim=0) # (TotalValid, ActDim)
                batch_goals = torch.cat(goals_list, dim=0) # (TotalValid, GoalDim)
            else:
                return {}
            
            # Now we have a large batch. We can subsample it to `batch_size` if it's too big,
            # or just use it all if memory allows.
            # JAX code uses `batch_size`.
            
            total_samples = batch_state.shape[0]
            indices = torch.randperm(total_samples, device=self.device)[:self._batch_size]
            
            b_state = batch_state[indices]
            b_actions = batch_actions[indices]
            b_goals = batch_goals[indices]

            # --- Update Critic (Encoders) ---
            sa_repr = self.sa_encoder(torch.cat([b_state, b_actions], dim=-1))
            g_repr = self.g_encoder(b_goals)
            
            # InfoNCE Loss
            crl_loss = -info_nce_loss(sa_repr, g_repr, temp=self.temp, version=self.crl_loss_version)
            
            self.critic_optimizer.zero_grad()
            crl_loss.backward()
            nn.utils.clip_grad_norm_(self.sa_encoder.parameters(), self._gradient_clip)
            nn.utils.clip_grad_norm_(self.g_encoder.parameters(), self._gradient_clip)
            self.critic_optimizer.step()

            # --- Update Actor and Alpha ---
            # We need observations for the actor.
            # The actor takes `obs` which is usually [state, goal].
            # Here `b_state` is the state. `b_goals` is the goal we are trying to reach.
            # So we construct actor input as cat([b_state, b_goals]).
            
            # Note: In the original code, `obs` from env already has a goal.
            # But in GCRL, we are training the actor to reach *any* goal (the sampled future goal).
            # So we should replace the goal in `obs` with `b_goals`.
            
            # Construct actor input
            actor_input = torch.cat([b_state, b_goals], dim=-1)
            
            # Sample action
            pred_actions, pred_logp = self._sample_action(actor_input)
            
            # Compute representations for actor loss
            sa_repr_pi = self.sa_encoder(torch.cat([b_state, pred_actions], dim=-1))
            g_repr_pi = self.g_encoder(b_goals)
            
            # Similarity (Energy)
            sim_pi = -torch.sum((sa_repr_pi - g_repr_pi)**2, dim=-1)
            
            # Actor Loss
            actor_loss = (self.alpha.detach() * pred_logp - sim_pi).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self._gradient_clip)
            self.actor_optimizer.step()
            
            # Alpha update
            alpha_loss = -(self.log_alpha * (pred_logp + self._target_entropy).detach()).mean()
            
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            total_actor_loss.append(actor_loss.item())
            total_alpha_loss.append(alpha_loss.item())
            total_critic_loss.append(crl_loss.item())

        return {
            "actor": np.mean(total_actor_loss),
            "alpha": np.mean(total_alpha_loss),
            "critic": np.mean(total_critic_loss)
        }

    def _sample_action(self, observation, compute_logp=True):
        mean, std = self.actor.forward(observation, compute_std=True)
        dist = torch.distributions.Normal(mean, std)
        actions = dist.rsample()
        actions_normalized, actions_scaled = self._scale_actions(actions, intermediate=True)
        
        if not compute_logp:
            return actions_scaled
            
        action_logp = dist.log_prob(actions).sum(-1) - torch.log(1.0 - actions_normalized.pow(2) + 1e-6).sum(-1)
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
