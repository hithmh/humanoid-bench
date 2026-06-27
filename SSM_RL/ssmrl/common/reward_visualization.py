"""
Reward prediction visualization utility for SSM-RL training.
Provides functions to create and log reward prediction plots to wandb.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_reward_trajectory(actual_rewards, predicted_rewards, step=0, save_path=None):
    """
    Create a plot comparing actual vs predicted reward trajectories.

    Args:
        actual_rewards:     1D numpy array of actual rewards
        predicted_rewards:  1D numpy array of predicted rewards
        step:               training step (for title)
        save_path:          optional path to save the figure

    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Trajectory comparison
    ax = axes[0]
    steps = np.arange(len(actual_rewards))
    ax.plot(steps, actual_rewards, 'b-o', label='Actual', linewidth=0.5, markersize=1)
    ax.plot(steps, predicted_rewards, 'r--s', label='Predicted', linewidth=0.5, markersize=1)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Reward Trajectory Comparison (Step {step})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter plot (actual vs predicted)
    ax = axes[1]
    ax.scatter(actual_rewards, predicted_rewards, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add diagonal line for perfect predictions
    min_val = min(actual_rewards.min(), predicted_rewards.min())
    max_val = max(actual_rewards.max(), predicted_rewards.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual Reward', fontsize=12)
    ax.set_ylabel('Predicted Reward', fontsize=12)
    ax.set_title(f'Actual vs Predicted Scatter Plot (Step {step})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

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
                                      save_dir=None):
    """
    Log reward prediction visualization to wandb.

    Args:
        wandb:              wandb module
        actual_rewards:     1D numpy array of actual rewards
        predicted_rewards:  1D numpy array of predicted rewards
        step:               training step
        save_dir:           optional directory to save figures locally
    """
    if wandb is None:
        return

    # Create figure
    fig = plot_reward_trajectory(
        actual_rewards,
        predicted_rewards,
        step=step,
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
