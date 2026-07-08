from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from ssmrl.trainer.base import Trainer
from ssmrl.common.reward_visualization import log_reward_visualization_to_wandb


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def control_metrics(self):
        if hasattr(self.agent, 'get_control_metrics'):
            return self.agent.get_control_metrics()
        return {'qpax_solver_failures': 0}

    @staticmethod
    def average_control_metrics(metrics_list):
        if not metrics_list:
            return {'qpax_solver_failures': 0}
        keys = set().union(*(metrics.keys() for metrics in metrics_list))
        averaged = {}
        for key in keys:
            values = [metrics.get(key, 0) for metrics in metrics_list]
            if all(isinstance(value, (int, float, np.number)) for value in values):
                averaged[key] = np.nanmean(values)
            else:
                string_values = []
                for value in values:
                    value = str(value)
                    if value and value not in {'0', 'none'} and value not in string_values:
                        string_values.append(value)
                averaged[key] = ' | '.join(string_values)
        return averaged

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes, ep_control_metrics = [], [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            self.agent.reset_for_control()
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            ep_control_metrics.append(self.control_metrics())
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        metrics = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )
        metrics.update(self.average_control_metrics(ep_control_metrics))
        return metrics

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True

        # Get reward visualization frequency from config
        reward_viz_freq = getattr(self.cfg, 'reward_viz_freq', self.cfg.eval_freq)

        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                control_metrics = self.control_metrics()
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                        **control_metrics,
                    )
                    train_metrics.update(self.common_metrics())

                    loss_keys = [k for k in train_metrics if 'loss' in k.lower()]
                    loss_metrics = {k: train_metrics[k] for k in loss_keys}
                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,
                                       **control_metrics,
                                       **loss_metrics}

                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()[0]
                self.agent.reset_for_control()
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            ## transform action to tensor first
            tensor_action = action if isinstance(action, torch.Tensor) else torch.from_numpy(action)
            self._tds.append(self.to_td(obs, tensor_action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = self.cfg.num_updates
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

                # Log reward prediction visualization
                if self._step % reward_viz_freq == 0 and self._step > self.cfg.seed_steps:
                    try:
                        reward_preds = self.agent.compute_reward_predictions(self.buffer, num_samples=1)
                        log_reward_visualization_to_wandb(
                            self.logger._wandb,
                            reward_preds['actual_rewards'],
                            reward_preds['predicted_rewards'],
                            step=self._step,
                            trajectory_length=self.cfg.horizon,
                            save_dir=self.logger._log_dir if hasattr(self.logger, '_log_dir') else None
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log reward visualization: {e}")

            self._step += 1

        self.logger.finish(self.agent)
