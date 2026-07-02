import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from ssmrl.common.buffer import Buffer
from tdmpc2.trainer.base import Trainer
from ssmrl.common.reward_visualization import log_reward_visualization_to_wandb


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx)[0], False, 0, 0
                while not done:
                    action = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, task=task_idx
                    )
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
            results.update(
                {
                    f"episode_reward+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_rewards
                    ),
                    f"episode_success+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_successes
                    ),
                }
            )
        return results

    def train(self):
        """Train a TD-MPC2 agent."""
        assert self.cfg.multitask and self.cfg.task in {
            "mt30",
            "mt80",
        }, "Offline training only supports multitask training with mt30 or mt80 task sets."

        # Load data
        assert self.cfg.task in self.cfg.data_dir, (
            f"Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, "
            f"please double-check your config."
        )
        fp = Path(os.path.join(self.cfg.data_dir, "*.pt"))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f"No data found at {fp}"
        print(f"Found {len(fps)} files in {fp}")

        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task == "mt80" else 501
        _cfg.buffer_size = 550_450_000 if self.cfg.task == "mt80" else 345_690_000
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc="Loading data"):
            td = torch.load(fp)
            assert td.shape[1] == _cfg.episode_length, (
                f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                f"please double-check your config."
            )
            for i in range(len(td)):
                self.buffer.add(td[i])
        assert (
            self.buffer.num_eps == self.buffer.capacity
        ), f"Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes."

        print(f"Training agent for {self.cfg.steps} iterations...")
        metrics = {}

        # Get reward visualization frequency from config
        reward_viz_freq = getattr(self.cfg, 'reward_viz_freq', self.cfg.eval_freq)

        for i in range(self.cfg.steps):
            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
                metrics = {
                    "iteration": i,
                    "total_time": time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f"{i}")
                self.logger.log(metrics, "pretrain")

            # Log reward prediction visualization
            if i % reward_viz_freq == 0 and i > 0:
                try:
                    reward_preds = self.agent.compute_reward_predictions(self.buffer, num_samples=1)
                    log_reward_visualization_to_wandb(
                        self.logger._wandb,
                        reward_preds['actual_rewards'],
                        reward_preds['predicted_rewards'],
                        step=i,
                        save_dir=self.logger._log_dir if hasattr(self.logger, '_log_dir') else None
                    )
                except Exception as e:
                    print(f"Warning: Failed to log reward visualization: {e}")

        self.logger.finish(self.agent)
