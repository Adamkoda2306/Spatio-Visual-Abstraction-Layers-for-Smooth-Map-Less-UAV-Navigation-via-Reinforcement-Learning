"""
Train a TRPO policy (sb3-contrib) over the compact 28-dim spatio-visual
state space produced by airsim_env.AirSimUAVEnv (Section 3.4, Eq. 15: KL
trust-region constraint).

Usage:
    # random goal each episode (generalizing policy)
    env\\Scripts\\python.exe train_trpo.py --timesteps 1000000 --alpha 0.01

    # train to reach one fixed (x, y, z) NED destination => goal 131.94, -275.53, 0.5
    env\\Scripts\\python.exe train_trpo.py --timesteps 1000000 --alpha 0.01 --goal_x 131.94 --goal_y -275.53 --goal_z 0.5
"""

import argparse
import os

from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import config
from airsim_env import AirSimUAVEnv
from callbacks import EpisodeLogCallback


def make_env(goal=None):
    env = AirSimUAVEnv(goal=goal)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--alpha", type=float, default=config.ALPHA_SMOOTH,
                         help="velocity smoothing coefficient (Table 2)")
    parser.add_argument("--resume", type=str, default=None, help="path to a .zip checkpoint to resume from")
    parser.add_argument("--goal_x", type=float, default=None, help="fixed training goal X (NED, meters)")
    parser.add_argument("--goal_y", type=float, default=None, help="fixed training goal Y (NED, meters)")
    parser.add_argument("--goal_z", type=float, default=None, help="fixed training goal Z (NED, meters, negative=up)")
    args = parser.parse_args()

    config.ALPHA_SMOOTH = args.alpha

    goal = None
    if args.goal_x is not None and args.goal_y is not None and args.goal_z is not None:
        goal = (args.goal_x, args.goal_y, args.goal_z)
        print(f"Training toward fixed goal: {goal}")
    else:
        print("No fixed goal given — sampling a random goal each episode.")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    env = make_env(goal=goal)

    if args.resume:
        model = TRPO.load(args.resume, env=env, tensorboard_log=config.LOG_DIR)
    else:
        model = TRPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=0.01,          # Eq. 15, trust-region bound delta
            cg_max_steps=15,
            n_critic_updates=10,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=config.LOG_DIR,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=config.MODEL_DIR,
        name_prefix=f"trpo_uav_alpha{args.alpha}",
    )
    episode_log_cb = EpisodeLogCallback()
    callback = CallbackList([checkpoint_cb, episode_log_cb])

    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="trpo")

    final_path = os.path.join(config.MODEL_DIR, f"trpo_uav_final_alpha{args.alpha}.zip")
    model.save(final_path)
    print(f"Saved final TRPO model to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
