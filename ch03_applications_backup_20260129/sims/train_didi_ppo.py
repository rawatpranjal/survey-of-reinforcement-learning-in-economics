"""
Stable Baselines3 PPO Training for DiDi Dispatch Environment
Chapter 3: Applications of RL
Trains PPO agent on high-fidelity ride-hailing dispatch using SB3.
"""

import os
import sys
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from didi_dispatch_hifi import (
    DiDiDispatchEnv,
    optimal_hungarian_policy,
    nearest_driver_policy,
    greedy_fare_policy,
    random_policy,
    evaluate_policy,
)


class ProgressCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, log_interval=10000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Step {self.n_calls:>7d} | Mean reward: {mean_reward:>8.1f} | Mean length: {mean_length:.0f}")
        return True


def make_env(env_kwargs):
    """Factory for creating DiDi environment."""
    def _init():
        env = DiDiDispatchEnv(**env_kwargs)
        return env
    return _init


def train_ppo_sb3(
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    policy_kwargs: dict = None,
    **env_kwargs
):
    """
    Train PPO using Stable Baselines3.

    Parameters
    ----------
    total_timesteps : int
        Total training timesteps
    n_envs : int
        Number of parallel environments
    seed : int
        Random seed
    learning_rate : float
        Learning rate
    n_steps : int
        Steps per environment per update
    batch_size : int
        Minibatch size
    n_epochs : int
        Number of epochs per update
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda
    clip_range : float
        PPO clip range
    ent_coef : float
        Entropy coefficient
    vf_coef : float
        Value function coefficient
    policy_kwargs : dict
        Policy network kwargs
    **env_kwargs
        Environment configuration
    """
    print("=" * 60)
    print("DiDi Dispatch PPO Training (Stable Baselines3)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Num envs: {n_envs}")
    print(f"Steps per env: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Entropy coef: {ent_coef}")
    print("=" * 60)

    # Default policy kwargs
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

    # Create vectorized environment with true parallelism
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    def make_single_env(idx):
        def _init():
            e = DiDiDispatchEnv(seed=seed + idx, **env_kwargs)
            return e
        return _init

    env = SubprocVecEnv([make_single_env(i) for i in range(n_envs)], start_method='fork')
    env = VecMonitor(env)

    # Create eval environment
    eval_env = DiDiDispatchEnv(seed=seed + 1000, **env_kwargs)
    eval_env = Monitor(eval_env)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=None,
    )

    print(f"\nPolicy architecture: {policy_kwargs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Callbacks
    progress_callback = ProgressCallback(log_interval=10000)

    # Train
    print("\nTraining started...")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        progress_bar=True,
    )

    print("-" * 60)
    print("Training completed.")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    eval_env_fresh = DiDiDispatchEnv(seed=seed + 5000, **env_kwargs)
    results = {}

    # Evaluate PPO
    ppo_rewards = []
    ppo_revenues = []
    ppo_cancels = []

    for ep in range(20):
        obs, info = eval_env_fresh.reset(seed=seed + 5000 + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_fresh.step(action)
            ep_reward += reward
            done = terminated or truncated

        ppo_rewards.append(ep_reward)
        ppo_revenues.append(info["total_revenue"])
        ppo_cancels.append(info["total_cancellations"])

    results["PPO"] = {
        "mean_reward": np.mean(ppo_rewards),
        "std_reward": np.std(ppo_rewards),
        "mean_revenue": np.mean(ppo_revenues),
        "mean_cancellations": np.mean(ppo_cancels),
    }

    # Evaluate baselines
    for name, policy in [
        ("Hungarian", optimal_hungarian_policy),
        ("Nearest", nearest_driver_policy),
        ("Greedy Fare", greedy_fare_policy),
        ("Random", random_policy),
    ]:
        res = evaluate_policy(eval_env_fresh, policy, 20, seed + 6000)
        results[name] = res

    # Print results table
    print("\nResults (20 episodes each):")
    print("-" * 70)
    print(f"{'Method':<15} {'Reward':>12} {'Std':>10} {'Revenue':>12} {'Cancels':>10}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<15} {res['mean_reward']:>12.1f} {res['std_reward']:>10.1f} "
              f"{res['mean_revenue']:>12.1f} {res['mean_cancellations']:>10.1f}")
    print("-" * 70)

    # Save results
    save_results(results, model, env_kwargs)

    env.close()

    return model, results


def save_results(results, model, env_kwargs):
    """Save training results and generate figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(results.keys())
    rewards = [results[m]["mean_reward"] for m in methods]
    stds = [results[m]["std_reward"] for m in methods]
    colors = ['#2ecc71' if m == 'PPO' else '#3498db' for m in methods]

    bars = ax.bar(methods, rewards, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('DiDi Dispatch: PPO vs Heuristics')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'didi_hifi_learning_curve.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {fig_path}")

    # Save LaTeX table
    table_path = os.path.join(output_dir, 'didi_hifi_results.tex')
    with open(table_path, 'w') as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Reward & Std & Revenue & Cancellations \\\\\n")
        f.write("\\midrule\n")
        for name, res in results.items():
            f.write(f"{name} & {res['mean_reward']:.1f} & {res['std_reward']:.1f} & "
                    f"{res['mean_revenue']:.1f} & {res['mean_cancellations']:.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Saved results table: {table_path}")

    # Save model
    model_path = os.path.join(output_dir, 'didi_ppo_model.zip')
    model.save(model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    # Environment config
    env_kwargs = dict(
        num_drivers=20,
        grid_radius=2,
        max_batch_size=30,
        episode_length=288,
    )

    # Train with SB3 PPO
    model, results = train_ppo_sb3(
        total_timesteps=500_000,
        n_envs=8,
        seed=42,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        **env_kwargs
    )
