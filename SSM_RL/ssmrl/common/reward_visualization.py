"""
Reward prediction visualization utility for SSM-RL training.
Provides functions to create and log reward prediction plots to wandb.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _to_trajectory_matrix(rewards, trajectory_length=None):
    rewards = np.asarray(rewards).squeeze()
    if rewards.ndim == 0:
        return rewards.reshape(1, 1)
    if rewards.ndim == 2:
        return rewards

    rewards = rewards.reshape(-1)
    if trajectory_length and trajectory_length > 0:
        usable = (len(rewards) // trajectory_length) * trajectory_length
        if usable:
            # Predictions are collected time-major: t0 for all batch items,
            # then t1 for all batch items, etc.
            return rewards[:usable].reshape(trajectory_length, -1).T
    return rewards.reshape(1, -1)


def plot_reward_trajectory(
    actual_rewards,
    predicted_rewards,
    step=0,
    save_path=None,
    trajectory_length=None,
    max_trajectories=8,
):
    """
    Create a plot comparing actual vs predicted reward trajectories.

    Args:
        actual_rewards:     1D numpy array of actual rewards
        predicted_rewards:  1D numpy array of predicted rewards
        step:               training step (for title)
        save_path:          optional path to save the figure
        trajectory_length:  optional number of steps per sampled trajectory
        max_trajectories:   maximum trajectories to show in the first plot

    Returns:
        matplotlib figure object
    """
    actual_matrix = _to_trajectory_matrix(actual_rewards, trajectory_length)
    predicted_matrix = _to_trajectory_matrix(predicted_rewards, trajectory_length)
    num_trajectories = min(len(actual_matrix), len(predicted_matrix))
    num_plotted = min(max_trajectories, num_trajectories)
    horizon = min(actual_matrix.shape[1], predicted_matrix.shape[1])

    fig_height = max(5, 1.15 * num_plotted)
    fig = plt.figure(figsize=(14, fig_height))
    grid = fig.add_gridspec(
        num_plotted,
        2,
        width_ratios=[1.45, 1.0],
        hspace=0.18,
        wspace=0.28,
    )
    trajectory_axes = [fig.add_subplot(grid[row, 0]) for row in range(num_plotted)]
    scatter_ax = fig.add_subplot(grid[:, 1])

    plotted_actual = actual_matrix[:num_plotted, :horizon]
    plotted_predicted = predicted_matrix[:num_plotted, :horizon]
    finite_values = np.concatenate([
        plotted_actual[np.isfinite(plotted_actual)],
        plotted_predicted[np.isfinite(plotted_predicted)],
    ])
    if finite_values.size:
        y_min, y_max = finite_values.min(), finite_values.max()
        y_pad = max((y_max - y_min) * 0.08, 1e-6)
        y_limits = (y_min - y_pad, y_max + y_pad)
    else:
        y_limits = None

    steps = np.arange(horizon)
    for idx, ax in enumerate(trajectory_axes):
        ax.plot(
            steps,
            plotted_actual[idx],
            color='tab:blue',
            label='Actual' if idx == 0 else None,
            linewidth=1.4,
        )
        ax.plot(
            steps,
            plotted_predicted[idx],
            color='tab:red',
            linestyle='--',
            label='Predicted' if idx == 0 else None,
            linewidth=1.4,
        )
        ax.set_ylabel(f'Traj {idx + 1}', fontsize=9)
        ax.grid(True, alpha=0.25)
        if y_limits:
            ax.set_ylim(*y_limits)
        if idx < num_plotted - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time Step', fontsize=12)

    title = f'Reward Trajectories (Step {step})'
    if num_trajectories > num_plotted:
        title += f' - showing {num_plotted} of {num_trajectories}'
    trajectory_axes[0].set_title(title, fontsize=13, fontweight='bold')
    trajectory_axes[0].legend(fontsize=10, loc='upper right')

    # Plot 2: Scatter plot (actual vs predicted)
    ax = scatter_ax
    actual_flat = np.asarray(actual_rewards).reshape(-1)
    predicted_flat = np.asarray(predicted_rewards).reshape(-1)
    usable = min(len(actual_flat), len(predicted_flat))
    actual_flat = actual_flat[:usable]
    predicted_flat = predicted_flat[:usable]
    ax.scatter(actual_flat, predicted_flat, alpha=0.45, s=24, edgecolors='none')

    # Add diagonal line for perfect predictions
    min_val = min(actual_flat.min(), predicted_flat.min())
    max_val = max(actual_flat.max(), predicted_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('Actual Reward', fontsize=12)
    ax.set_ylabel('Predicted Reward', fontsize=12)
    ax.set_title(f'Actual vs Predicted Scatter Plot (Step {step})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def compute_reward_metrics(actual_rewards, predicted_rewards):
    """
    Compute metrics comparing actual and predicted rewards.

    Args:
        actual_rewards:     1D numpy array of actual rewards
        predicted_rewards:  1D numpy array of predicted rewards

    Returns:
        dict with metrics
    """
    actual_rewards = np.asarray(actual_rewards).reshape(-1)
    predicted_rewards = np.asarray(predicted_rewards).reshape(-1)
    usable = min(len(actual_rewards), len(predicted_rewards))
    actual_rewards = actual_rewards[:usable]
    predicted_rewards = predicted_rewards[:usable]

    mse = np.mean((actual_rewards - predicted_rewards) ** 2)
    mae = np.mean(np.abs(actual_rewards - predicted_rewards))

    # Correlation
    if len(actual_rewards) > 1 and np.std(actual_rewards) > 0 and np.std(predicted_rewards) > 0:
        correlation = np.corrcoef(actual_rewards, predicted_rewards)[0, 1]
    else:
        correlation = 0.0

    # R-squared
    ss_res = np.sum((actual_rewards - predicted_rewards) ** 2)
    ss_tot = np.sum((actual_rewards - np.mean(actual_rewards)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(correlation),
        'r_squared': float(r_squared),
    }


def log_reward_visualization_to_wandb(wandb, actual_rewards, predicted_rewards, step,
                                      save_dir=None, trajectory_length=None,
                                      max_trajectories=8):
    """
    Log reward prediction visualization to wandb.

    Args:
        wandb:              wandb module
        actual_rewards:     1D numpy array of actual rewards
        predicted_rewards:  1D numpy array of predicted rewards
        step:               training step
        save_dir:           optional directory to save figures locally
        trajectory_length:  optional number of steps per sampled trajectory
        max_trajectories:   maximum trajectories to show in the first plot
    """
    if wandb is None:
        return

    # Create figure
    fig = plot_reward_trajectory(
        actual_rewards,
        predicted_rewards,
        step=step,
        trajectory_length=trajectory_length,
        max_trajectories=max_trajectories,
        save_path=Path(save_dir) / f'reward_viz_step_{step}.png' if save_dir else None
    )

    # Log figure to wandb
    wandb.log({
        'reward_prediction/trajectory_plot': wandb.Image(fig),
    }, step=step)

    # Compute and log metrics
    metrics = compute_reward_metrics(actual_rewards, predicted_rewards)
    for metric_name, metric_value in metrics.items():
        wandb.log({
            f'reward_prediction/{metric_name}': metric_value,
        }, step=step)

    plt.close(fig)

    return metrics
