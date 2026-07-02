import sys

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = self._resolve_device(getattr(cfg, "device", "auto"))
        self._replay_device = str(getattr(cfg, "replay_device", "auto")).lower()
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._seq_len = cfg.horizon + cfg.history_horizon + 1
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
        )
        self._batch_size = cfg.batch_size * self._seq_len
        self._num_eps = 0
        self._reward_priority_sampling = bool(
            getattr(cfg, "reward_priority_sampling", False)
        )
        self._priority_alpha = float(getattr(cfg, "reward_priority_alpha", 0.6))
        self._priority_beta = float(getattr(cfg, "reward_priority_beta", 0.4))
        self._priority_eps = float(getattr(cfg, "reward_priority_eps", 1e-3))
        self._priority_uniform_mix = float(
            getattr(cfg, "reward_priority_uniform_mix", 0.1)
        )
        self._priority_uniform_mix = min(max(self._priority_uniform_mix, 0.0), 1.0)
        self._priority_max_index_size = int(
            getattr(cfg, "reward_priority_max_index_size", 50_000_000)
        )
        self._episodes = {}
        self._storage_to_episode = None
        self._priority_index_dirty = True
        self._active_episode_ids = []
        self._episode_reward_weights = torch.empty(0, dtype=torch.float32)
        self._episode_window_counts = torch.empty(0, dtype=torch.float32)
        self._total_reward_weight = 0.0
        self._total_windows = 0
        self._last_sample_weights = torch.ones(cfg.batch_size, dtype=torch.float32)

    def _resolve_device(self, requested):
        requested = str(requested).lower()
        if requested in {"auto", "none", "???", ""}:
            requested = (
                "cuda"
                if sys.platform != "darwin" and torch.cuda.is_available()
                else "cpu"
            )
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        return torch.device(requested)

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        if (
            sys.platform == "darwin"
            or not torch.cuda.is_available()
            or self._device.type == "cpu"
        ):
            mem_free = 0
        else:
            mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes/1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory unless overridden.
        if self._replay_device in {"auto", "none", "???", ""}:
            storage_device = "cuda" if 2.5 * total_bytes < mem_free else "cpu"
        else:
            storage_device = str(self._resolve_device(self._replay_device))
        print(f"Using {storage_device.upper()} memory for storage.")
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td["obs"]
        action = td["action"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        task = td["task"][0] if "task" in td.keys() else None
        return self._to_device(obs, action, reward, task)

    def _init_priority_index(self):
        if not self._reward_priority_sampling or self._storage_to_episode is not None:
            return
        if self._capacity > self._priority_max_index_size:
            print(
                "Warning: disabling reward-prioritized sampling because buffer "
                f"capacity ({self._capacity:,}) exceeds reward_priority_max_index_size "
                f"({self._priority_max_index_size:,})."
            )
            self._reward_priority_sampling = False
            return
        try:
            self._storage_to_episode = torch.full(
                (self._capacity,), -1, dtype=torch.int32
            )
        except RuntimeError as err:
            print(
                "Warning: disabling reward-prioritized sampling because the "
                f"storage index map could not be allocated ({err})."
            )
            self._reward_priority_sampling = False

    def _to_storage_indices(self, indices):
        if isinstance(indices, slice):
            start = 0 if indices.start is None else indices.start
            stop = self._capacity if indices.stop is None else indices.stop
            step = 1 if indices.step is None else indices.step
            return torch.arange(start, stop, step, dtype=torch.long)
        if isinstance(indices, range):
            return torch.as_tensor(list(indices), dtype=torch.long)
        if isinstance(indices, torch.Tensor):
            return indices.detach().flatten().cpu().long()
        return torch.as_tensor(indices, dtype=torch.long).flatten().cpu()

    def _invalidate_overwritten_episodes(self, storage_indices):
        if not self._reward_priority_sampling or self._storage_to_episode is None:
            return
        old_eps = self._storage_to_episode[storage_indices].unique()
        for ep_id in old_eps.tolist():
            if ep_id >= 0 and ep_id in self._episodes:
                self._episodes[ep_id]["active"] = False
        self._storage_to_episode[storage_indices] = int(self._num_eps)

    def _episode_window_priorities(self, td):
        rewards = td["reward"].detach().flatten().float().cpu()
        num_windows = len(rewards) - self._seq_len + 1
        if num_windows <= 0:
            return torch.empty(0, dtype=torch.float32)

        # The training loss consumes rewards after the history context.
        start = 1 + self.cfg.history_horizon
        stop = start + num_windows + self.cfg.horizon - 1
        positive_rewards = rewards.clamp_min(0.0)
        windows = positive_rewards[start:stop].unfold(0, self.cfg.horizon, 1)
        return windows.sum(dim=-1).add(self._priority_eps).clamp_min(self._priority_eps)

    def _register_episode_priorities(self, storage_indices, td):
        if not self._reward_priority_sampling:
            return
        priorities = self._episode_window_priorities(td)
        if priorities.numel() == 0:
            return
        weights = priorities.pow(self._priority_alpha)
        self._episodes[int(self._num_eps)] = {
            "active": True,
            "indices": storage_indices,
            "priorities": priorities,
            "weights": weights,
            "weight_sum": float(weights.sum().item()),
            "num_windows": int(priorities.numel()),
        }
        self._priority_index_dirty = True

    def _rebuild_priority_index(self):
        active = [
            (ep_id, ep)
            for ep_id, ep in self._episodes.items()
            if ep["active"] and ep["num_windows"] > 0 and ep["weight_sum"] > 0
        ]
        self._active_episode_ids = [ep_id for ep_id, _ in active]
        if not active:
            self._episode_reward_weights = torch.empty(0, dtype=torch.float32)
            self._episode_window_counts = torch.empty(0, dtype=torch.float32)
            self._total_reward_weight = 0.0
            self._total_windows = 0
            self._priority_index_dirty = False
            return
        self._episode_reward_weights = torch.tensor(
            [ep["weight_sum"] for _, ep in active], dtype=torch.float32
        )
        self._episode_window_counts = torch.tensor(
            [ep["num_windows"] for _, ep in active], dtype=torch.float32
        )
        self._total_reward_weight = float(self._episode_reward_weights.sum().item())
        self._total_windows = int(self._episode_window_counts.sum().item())
        self._priority_index_dirty = False

    def _sample_episode(self, use_uniform):
        if use_uniform:
            probs = self._episode_window_counts / self._episode_window_counts.sum()
        else:
            probs = self._episode_reward_weights / self._episode_reward_weights.sum()
        episode_pos = int(torch.multinomial(probs, 1, replacement=True).item())
        ep_id = self._active_episode_ids[episode_pos]
        ep = self._episodes[ep_id]

        if use_uniform:
            start = int(torch.randint(ep["num_windows"], (1,)).item())
        else:
            local_probs = ep["weights"] / ep["weight_sum"]
            start = int(torch.multinomial(local_probs, 1, replacement=True).item())

        local_weight = float(ep["weights"][start].item())
        reward_prob = local_weight / max(self._total_reward_weight, self._priority_eps)
        uniform_prob = 1.0 / max(self._total_windows, 1)
        sample_prob = (
            (1.0 - self._priority_uniform_mix) * reward_prob
            + self._priority_uniform_mix * uniform_prob
        )
        return ep, start, sample_prob

    def _sample_reward_prioritized(self):
        if self._priority_index_dirty:
            self._rebuild_priority_index()
        if not self._active_episode_ids:
            td = self._buffer.sample().view(-1, self._seq_len).permute(1, 0)
            weights = torch.ones(self.cfg.batch_size, dtype=torch.float32)
            return td, weights

        window_indices, sample_probs = [], []
        for _ in range(self.cfg.batch_size):
            use_uniform = (
                torch.rand(()) < self._priority_uniform_mix
                or self._total_reward_weight <= 0
            )
            ep, start, sample_prob = self._sample_episode(bool(use_uniform))
            window_indices.append(ep["indices"][start : start + self._seq_len])
            sample_probs.append(sample_prob)

        storage_indices = torch.stack(window_indices, dim=0)
        td = self._buffer[storage_indices.reshape(-1)]
        td = td.view(self.cfg.batch_size, self._seq_len).permute(1, 0)

        probs = torch.tensor(sample_probs, dtype=torch.float32).clamp_min(1e-12)
        if self._priority_beta <= 0:
            weights = torch.ones_like(probs)
        else:
            weights = (1.0 / (max(self._total_windows, 1) * probs)).pow(
                self._priority_beta
            )
            weights = weights / weights.max().clamp_min(1e-12)
        return td, weights

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.ones_like(td["reward"], dtype=torch.int64) * self._num_eps

        # FIX for HumanoidBench #
        if len(td["episode"]) < self._seq_len:
            return self._num_eps
        ################################

        if self._num_eps == 0:
            self._buffer = self._init(td)
            self._init_priority_index()
        storage_indices = self._to_storage_indices(self._buffer.extend(td))
        self._invalidate_overwritten_episodes(storage_indices)
        self._register_episode_priorities(storage_indices, td)
        self._num_eps += 1
        return self._num_eps

    def sample(self, return_weights=False):
        """Sample a batch of subsequences from the buffer."""
        if self._reward_priority_sampling:
            td, weights = self._sample_reward_prioritized()
        else:
            td = self._buffer.sample().view(-1, self._seq_len).permute(1, 0)
            weights = torch.ones(self.cfg.batch_size, dtype=torch.float32)
        self._last_sample_weights = weights
        batch = tuple(self._prepare_batch(td))
        if return_weights:
            return (*batch, weights.to(self._device, non_blocking=True).unsqueeze(-1))
        return batch
