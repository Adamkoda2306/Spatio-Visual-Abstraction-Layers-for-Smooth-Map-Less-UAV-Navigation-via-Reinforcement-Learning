"""
Train a PPO policy (stable-baselines3) over the compact 28-dim spatio-visual
state space produced by airsim_env.AirSimUAVEnv (Section 3.4, Eq. 14).

Usage:
    # random goal each episode (generalizing policy)
    env\\Scripts\\python.exe train_ppo.py --timesteps 1000000 --alpha 0.01

    # train to reach one fixed (x, y, z) NED destination => goal 131.94, -275.53, 0.5
    env\\Scripts\\python.exe train_ppo.py --timesteps 1000000 --alpha 0.01 --goal_x 131.94 --goal_y -275.53 --goal_z 0.5
"""

import argparse
import os

from stable_baselines3 import PPO
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
        model = PPO.load(args.resume, env=env, tensorboard_log=config.LOG_DIR)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,          # Eq. 14, epsilon = 0.2
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=config.LOG_DIR,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=config.MODEL_DIR,
        name_prefix=f"ppo_uav_alpha{args.alpha}",
    )
    episode_log_cb = EpisodeLogCallback()
    callback = CallbackList([checkpoint_cb, episode_log_cb])

    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="ppo")

    final_path = os.path.join(config.MODEL_DIR, f"ppo_uav_final_alpha{args.alpha}.zip")
    model.save(final_path)
    print(f"Saved final PPO model to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
