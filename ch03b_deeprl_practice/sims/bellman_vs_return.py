"""
bellman_vs_return.py
Chapter: The Empirics of Deep RL (ch03b_deeprl_practice)
Experiment: TD loss is a poor proxy for policy quality in deep RL.
A network can minimize Bellman residual while episode return stagnates or collapses.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import load_results, save_results, add_cache_args

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
TOTAL_STEPS     = 60000
EVAL_INTERVAL   = 500        # evaluate policy every N steps
BUFFER_SIZE     = 10000
BATCH_SIZE      = 64
LR              = 1e-3
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995
TARGET_UPDATE   = 100        # hard target network update every N steps
ROLLING_WINDOW  = 20         # rolling average for episode return
NUM_SEEDS       = 3
SEEDS           = [42, 123, 777]

SAVE_DIR        = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR       = os.path.join(SAVE_DIR, 'cache')
SCRIPT_NAME     = 'bellman_vs_return'
CONFIG = {
    'total_steps': TOTAL_STEPS,
    'eval_interval': EVAL_INTERVAL,
    'buffer_size': BUFFER_SIZE,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'gamma': GAMMA,
    'epsilon_start': EPSILON_START,
    'epsilon_end': EPSILON_END,
    'epsilon_decay': EPSILON_DECAY,
    'target_update': TARGET_UPDATE,
    'rolling_window': ROLLING_WINDOW,
    'seeds': SEEDS,
    'version': 1,
}

# ------------------------------------------------------------------
# Q-Network
# ------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_next)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------------
# DQN training loop (returns td_loss_log, return_log, step_log)
# ------------------------------------------------------------------
def train_dqn(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make('CartPole-v1')
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net      = QNetwork(obs_dim, n_actions)
    tgt_net    = QNetwork(obs_dim, n_actions)
    tgt_net.load_state_dict(q_net.state_dict())
    optimizer  = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer     = ReplayBuffer(BUFFER_SIZE)

    epsilon      = EPSILON_START
    total_steps  = 0
    episode_ret  = 0.0
    episode_rets = []

    td_loss_log  = []   # (step, td_loss)
    return_log   = []   # (step, rolling_avg_return)
    step_log     = []   # step checkpoints

    obs, _ = env.reset(seed=seed)

    while total_steps < TOTAL_STEPS:
        # e-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(obs).unsqueeze(0))
                action = q_vals.argmax().item()

        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, obs_next, float(done))

        episode_ret += reward
        total_steps += 1
        obs = obs_next
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if done:
            episode_rets.append(episode_ret)
            episode_ret = 0.0
            obs, _ = env.reset()

        # training step
        if len(buffer) >= BATCH_SIZE:
            s, a, r, s_next, d = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                tgt_q = tgt_net(s_next).max(1)[0]
                y     = r + GAMMA * tgt_q * (1 - d)

            q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss   = nn.functional.mse_loss(q_vals, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record at eval_interval checkpoints
            if total_steps % EVAL_INTERVAL == 0:
                rolling = (
                    np.mean(episode_rets[-ROLLING_WINDOW:])
                    if len(episode_rets) >= 1 else 0.0
                )
                td_loss_log.append(loss.item())
                return_log.append(rolling)
                step_log.append(total_steps)

        # hard target update
        if total_steps % TARGET_UPDATE == 0:
            tgt_net.load_state_dict(q_net.state_dict())

    env.close()
    return step_log, td_loss_log, return_log


# ------------------------------------------------------------------
# Supervised analogue: logistic regression on 2D toy data
# ------------------------------------------------------------------
def train_logistic():
    np.random.seed(42)
    torch.manual_seed(42)

    n = 800
    X1 = np.random.randn(n//2, 2) + np.array([1.5, 0])
    X2 = np.random.randn(n//2, 2) + np.array([-1.5, 0])
    X  = np.vstack([X1, X2])
    y  = np.array([1]*( n//2) + [0]*(n//2), dtype=float)

    idx  = np.random.permutation(n)
    X, y = X[idx], y[idx]
    Xt   = torch.FloatTensor(X)
    yt   = torch.FloatTensor(y).unsqueeze(1)

    model     = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS    = 600
    loss_log  = []
    acc_log   = []
    ep_log    = []

    for ep in range(1, EPOCHS + 1):
        logits = model(Xt)
        loss   = criterion(logits, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 5 == 0:
            with torch.no_grad():
                pred = (torch.sigmoid(model(Xt)) >= 0.5).float()
                acc  = (pred == yt).float().mean().item()
            loss_log.append(loss.item())
            acc_log.append(acc * 100)
            ep_log.append(ep)

    return ep_log, loss_log, acc_log


# ------------------------------------------------------------------
# Normalize for dual-axis plots
# ------------------------------------------------------------------
def normalize_01(arr):
    mn, mx = np.min(arr), np.max(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ------------------------------------------------------------------
# compute_data: run all computation, return dict
# ------------------------------------------------------------------
def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("=" * 60)
    print("Experiment: Bellman Residual vs. Episode Return Disconnect")
    print("=" * 60)
    print(f"Environment: CartPole-v1 | Steps: {TOTAL_STEPS} | Seeds: {SEEDS}")
    print(f"Batch: {BATCH_SIZE} | Buffer: {BUFFER_SIZE} | Target update: {TARGET_UPDATE}")
    print()

    all_steps     = []
    all_td_loss   = []
    all_ep_return = []

    for seed in SEEDS:
        print(f"Training DQN, seed={seed}...", flush=True)
        steps, td_loss, ep_ret = train_dqn(seed)
        all_steps.append(steps)
        all_td_loss.append(td_loss)
        all_ep_return.append(ep_ret)

        min_len = min(len(steps), len(td_loss), len(ep_ret))
        print(f"  Seed {seed}: {min_len} checkpoints logged")
        if min_len > 0:
            final_td  = td_loss[-1]
            final_ret = ep_ret[-1]
            print(f"  Final TD loss: {final_td:.4f} | Final rolling return: {final_ret:.1f}")

    print()

    # Align lengths across seeds (take minimum)
    min_len = min(len(s) for s in all_steps)
    steps_ref    = all_steps[0][:min_len]
    td_matrix    = np.array([x[:min_len] for x in all_td_loss])
    ret_matrix   = np.array([x[:min_len] for x in all_ep_return])

    td_mean   = td_matrix.mean(0)
    td_se     = td_matrix.std(0) / np.sqrt(NUM_SEEDS)
    ret_mean  = ret_matrix.mean(0)
    ret_se    = ret_matrix.std(0) / np.sqrt(NUM_SEEDS)

    # Print summary table
    print(f"{'Step':>8} {'TD Loss (mean±se)':>22} {'Ep Return (mean±se)':>22}")
    print("-" * 56)
    idx_print = list(range(0, min_len, max(1, min_len // 10)))
    for i in idx_print:
        print(f"{steps_ref[i]:>8d}  {td_mean[i]:>8.4f} ± {td_se[i]:>6.4f}   {ret_mean[i]:>8.1f} ± {ret_se[i]:>6.1f}")

    print()

    # Run supervised analogue
    print("Training logistic regression (supervised analogue)...", flush=True)
    ep_log_sup, loss_log_sup, acc_log_sup = train_logistic()
    print(f"  Final CE loss: {loss_log_sup[-1]:.4f} | Final accuracy: {acc_log_sup[-1]:.1f}%")
    print()

    # Detailed correlation analysis
    print()
    print("Correlation analysis (TD loss vs. episode return):")
    corr = np.corrcoef(td_mean, ret_mean)[0, 1]
    print(f"  Pearson correlation (TD loss, episode return): {corr:.4f}")
    print(f"  (In supervised learning the analogous metric is near +1.0 / -1.0)")
    print()

    # Phase analysis: early vs late training
    mid = min_len // 2
    corr_early = np.corrcoef(td_mean[:mid], ret_mean[:mid])[0, 1] if mid > 2 else float('nan')
    corr_late  = np.corrcoef(td_mean[mid:], ret_mean[mid:])[0, 1] if (min_len - mid) > 2 else float('nan')
    print(f"  Correlation (first half): {corr_early:.4f}")
    print(f"  Correlation (second half): {corr_late:.4f}")
    print()

    data = {
        'steps_ref': steps_ref,
        'td_mean': td_mean,
        'td_se': td_se,
        'ret_mean': ret_mean,
        'ret_se': ret_se,
        'ep_log_sup': ep_log_sup,
        'loss_log_sup': loss_log_sup,
        'acc_log_sup': acc_log_sup,
        'corr': corr,
        'corr_early': corr_early,
        'corr_late': corr_late,
    }

    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ------------------------------------------------------------------
# generate_outputs: figures and tables from data dict
# ------------------------------------------------------------------
def generate_outputs(data):
    steps_ref = data['steps_ref']
    td_mean = data['td_mean']
    td_se = data['td_se']
    ret_mean = data['ret_mean']
    ret_se = data['ret_se']
    ep_log_sup = data['ep_log_sup']
    loss_log_sup = data['loss_log_sup']
    acc_log_sup = data['acc_log_sup']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: supervised ----
    ax = axes[0]
    ax2 = ax.twinx()

    color_loss = '#2166ac'
    color_acc  = '#d6604d'

    ax.plot(ep_log_sup, loss_log_sup, color=color_loss, lw=1.8, label='CE loss')
    ax.set_xlabel('Training epochs', fontsize=11)
    ax.set_ylabel('Cross-entropy loss', color=color_loss, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color_loss)

    ax2.plot(ep_log_sup, acc_log_sup, color=color_acc, lw=1.8, linestyle='--', label='Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', color=color_acc, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_acc)

    ax.set_title('Supervised learning\n(loss $\\downarrow$ $\\Leftrightarrow$ accuracy $\\uparrow$)', fontsize=11)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

    # ---- Panel B: DQN ----
    ax = axes[1]
    ax2 = ax.twinx()

    ax.fill_between(steps_ref, td_mean - td_se, td_mean + td_se, color=color_loss, alpha=0.2)
    ax.plot(steps_ref, td_mean, color=color_loss, lw=1.8, label='TD loss')
    ax.set_xlabel('Environment steps', fontsize=11)
    ax.set_ylabel('TD loss (MSE)', color=color_loss, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color_loss)

    ax2.fill_between(steps_ref, ret_mean - ret_se, ret_mean + ret_se, color=color_acc, alpha=0.2)
    ax2.plot(steps_ref, ret_mean, color=color_acc, lw=1.8, linestyle='--', label='Episode return')
    ax2.set_ylabel('Episode return (rolling avg.)', color=color_acc, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_acc)

    ax.set_title('Deep Q-learning on CartPole-v1\n(TD loss and return decoupled)', fontsize=11)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout(pad=2.0)

    out_path = os.path.join(SAVE_DIR, 'bellman_vs_return.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_path}")

    plt.close(fig)

    print()
    print("Output files:")
    print(f"  {out_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()

    if not args.data_only:
        generate_outputs(data)
