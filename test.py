"""
Evaluation / analysis script reproducing the experiments in Section 5:

  --mode rollout   : run N episodes with a trained model, print per-episode
                      reward/steps/status (REACHED GOAL / COLLISION / STOPPED),
                      success rate, velocity oscillation; save a 3D trajectory
                      plot (Fig. 3 style).
  --mode alpha_sweep: reproduce Table 2 / Fig. 4-6 (effect of alpha on
                      velocity oscillation, reward, episode length).
  --mode compare    : reproduce Fig. 8 (PPO vs TRPO episode length across
                      training checkpoints).

Usage:
    env\\Scripts\\python.exe test.py --mode rollout --model models/ppo_uav_final_alpha0.01.zip --algo ppo --episodes 5
    env\\Scripts\\python.exe test.py --mode rollout --model models/ppo_uav_final_alpha0.01.zip --algo ppo --goal_x 131.94 --goal_y -275.53 --goal_z -10.0
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sb3_contrib import RecurrentPPO, TRPO

import config
from airsim_env import AirSimUAVEnv

ALGOS = {"ppo": RecurrentPPO, "trpo": TRPO}


def load_model(path, algo):
    return ALGOS[algo].load(path)


def episode_status(info) -> str:
    if info.get("success"):
        return "REACHED GOAL"
    if info.get("collided"):
        return "COLLISION"
    return "STOPPED"  # timeout / truncated without reaching the goal or colliding


def run_episode(env, model, deterministic=True, goal=None, verbose=True):
    obs, _ = env.reset(options={"goal": goal} if goal is not None else None)
    done = False
    total_reward = 0.0
    steps = 0
    commands = []
    info = {}

    # RecurrentPPO carries an LSTM hidden state across steps within an
    # episode (its "memory" of previous actions/observations); reset it at
    # episode_start and thread it through predict() each step. Non-recurrent
    # algorithms (e.g. TRPO) simply ignore state/episode_start.
    lstm_states = None
    episode_start = np.array([True])

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = np.array([done])
        total_reward += reward
        steps += 1
        commands.append(env._cmd_smooth.copy())

    commands = np.array(commands) if commands else np.zeros((1, env.action_space.shape[0]))
    osc = float(np.mean(np.linalg.norm(np.diff(commands, axis=0), axis=1))) if len(commands) > 1 else 0.0

    status = episode_status(info)
    if verbose:
        print(f"Reward: {total_reward:.2f}, Steps: {steps}, "
              f"Final distance: {info.get('distance', float('nan')):.2f} m, STATUS: {status}")

    return dict(
        reward=total_reward,
        steps=steps,
        success=info.get("success", False),
        collided=info.get("collided", False),
        status=status,
        oscillation=osc,
        trajectory=np.array(env.trajectory),
    )


def plot_trajectory(trajectory, goal, save_path="trajectory.png"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = trajectory[:, 0], trajectory[:, 1], -trajectory[:, 2]  # altitude = -z (NED)
    progression = np.linspace(0, 1, len(xs))
    sc = ax.scatter(xs, ys, zs, c=progression, cmap="plasma", s=8)
    ax.plot(xs, ys, zs, color="gray", alpha=0.4, linewidth=1)
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=80, label="Start")
    ax.scatter([goal[0]], [goal[1]], [-goal[2]], c="red", marker="x", s=80, label="Goal")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend()
    fig.colorbar(sc, label="Trajectory Progression")
    plt.title("UAV Trajectory")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory plot to {save_path}")


def rollout_mode(args):
    env = AirSimUAVEnv()
    model = load_model(args.model, args.algo)

    goal = None
    if args.goal_x is not None and args.goal_y is not None and args.goal_z is not None:
        goal = (args.goal_x, args.goal_y, args.goal_z)
        print(f"Flying to fixed goal: {goal}")

    results = []
    for ep in range(args.episodes):
        print(f"--- Episode {ep + 1}/{args.episodes} ---")
        results.append(run_episode(env, model, goal=goal))

    success_rate = np.mean([r["success"] for r in results])
    collision_rate = np.mean([r["collided"] for r in results])
    mean_reward = np.mean([r["reward"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])
    mean_osc = np.mean([r["oscillation"] for r in results])

    print()
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Collision rate: {collision_rate:.2f}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Mean episode length: {mean_steps:.1f}")
    print(f"Mean velocity oscillation: {mean_osc:.4f}")

    plot_trajectory(results[-1]["trajectory"], env._goal)
    env.close()


def alpha_sweep_mode(args):
    """Reproduces Table 2 / Fig. 4-6 for a fixed trained model family."""
    env = AirSimUAVEnv()
    alphas = args.alphas
    rows = []

    for alpha in alphas:
        config.ALPHA_SMOOTH = alpha
        model_path = args.model_template.format(alpha=alpha)
        if not os.path.exists(model_path):
            print(f"[skip] no model found for alpha={alpha} at {model_path}")
            continue
        model = load_model(model_path, args.algo)
        results = [run_episode(env, model, verbose=False) for _ in range(args.episodes)]

        rows.append(dict(
            alpha=alpha,
            oscillation=np.mean([r["oscillation"] for r in results]),
            reward=np.mean([r["reward"] for r in results]),
            episode_length=np.mean([r["steps"] for r in results]),
        ))
        print(rows[-1])

    env.close()
    if not rows:
        return

    alphas_x = [r["alpha"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(alphas_x, [r["oscillation"] for r in rows], marker="o")
    axes[0].set_title("Effect of Alpha on UAV Motion Stability")
    axes[0].set_xlabel("Alpha Value")
    axes[0].set_ylabel("Velocity Oscillation")

    axes[1].plot(alphas_x, [r["reward"] for r in rows], marker="o", color="tab:blue")
    axes[1].set_title("Effect of Alpha on PPO Reward")
    axes[1].set_xlabel("Alpha Value")
    axes[1].set_ylabel("Total Reward")

    axes[2].plot(alphas_x, [r["episode_length"] for r in rows], marker="o", color="tab:orange")
    axes[2].set_title("Effect of Alpha on Navigation Efficiency")
    axes[2].set_xlabel("Alpha Value")
    axes[2].set_ylabel("Episode Length")

    plt.tight_layout()
    plt.savefig("alpha_sweep.png", dpi=150)
    print("Saved alpha_sweep.png")


def compare_mode(args):
    """Reproduces Fig. 8: episode length vs training checkpoint for PPO and TRPO."""
    env = AirSimUAVEnv()

    plt.figure(figsize=(8, 5))
    for algo, pattern in [("ppo", args.ppo_glob), ("trpo", args.trpo_glob)]:
        checkpoints = sorted(glob.glob(pattern))
        steps_x, ep_len_y = [], []
        for ckpt in checkpoints:
            model = load_model(ckpt, algo)
            results = [run_episode(env, model, verbose=False) for _ in range(args.episodes)]
            ep_len_y.append(np.mean([r["steps"] for r in results]))
            base = os.path.basename(ckpt)
            digits = "".join(c for c in base if c.isdigit())
            steps_x.append(int(digits) if digits else len(steps_x))
        if steps_x:
            order = np.argsort(steps_x)
            plt.plot(np.array(steps_x)[order], np.array(ep_len_y)[order], marker="o", label=algo.upper())

    plt.xlabel("Training Steps")
    plt.ylabel("Navigation Episode Length")
    plt.title("TRPO vs PPO Navigation Efficiency Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_vs_trpo.png", dpi=150)
    print("Saved ppo_vs_trpo.png")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rollout", "alpha_sweep", "compare"], default="rollout")
    parser.add_argument("--model", type=str, help="path to a trained model .zip (rollout mode)")
    parser.add_argument("--model_template", type=str,
                         default=os.path.join(config.MODEL_DIR, "ppo_uav_final_alpha{alpha}.zip"),
                         help="path template with {alpha} placeholder (alpha_sweep mode)")
    parser.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--goal_x", type=float, default=None, help="fixed goal X (NED, meters), rollout mode")
    parser.add_argument("--goal_y", type=float, default=None, help="fixed goal Y (NED, meters), rollout mode")
    parser.add_argument("--goal_z", type=float, default=None, help="fixed goal Z (NED, meters, negative=up), rollout mode")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05, 0.1])
    parser.add_argument("--ppo_glob", type=str, default=os.path.join(config.MODEL_DIR, "ppo_uav_alpha*_*_steps.zip"))
    parser.add_argument("--trpo_glob", type=str, default=os.path.join(config.MODEL_DIR, "trpo_uav_alpha*_*_steps.zip"))
    args = parser.parse_args()

    if args.mode == "rollout":
        assert args.model, "--model is required for rollout mode"
        rollout_mode(args)
    elif args.mode == "alpha_sweep":
        alpha_sweep_mode(args)
    elif args.mode == "compare":
        compare_mode(args)


if __name__ == "__main__":
    main()
