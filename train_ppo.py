"""
Train a PPO policy (sb3-contrib RecurrentPPO, i.e. PPO + an LSTM) over the
compact 28-dim spatio-visual state space produced by airsim_env.AirSimUAVEnv
(Section 3.4, Eq. 14).

The LSTM gives the policy "memory": its hidden state carries information
about the sequence of states/actions/outcomes seen earlier in the current
episode (e.g. "I already tried forward here and clipped an obstacle"),
instead of the plain MLP policy picking each action from the current 28-dim
state alone.

Usage:
    # random goal each episode (generalizing policy)
    env\\Scripts\\python.exe train_ppo.py --timesteps 1000000 --alpha 0.01

    # train to reach one fixed (x, y, z) NED destination => goal 131.94, -275.53, -10.0 or 127.32, 25.28, -0.15 (AIRSIMNH)
    env\\Scripts\\python.exe train_ppo.py --timesteps 1000000 --alpha 0.01 --goal_x 131.94 --goal_y -275.53 --goal_z -10.0
"""

import argparse
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import config
from airsim_env import AirSimUAVEnv
from callbacks import EpisodeLogCallback


def make_env(goal=None):
    env = AirSimUAVEnv(goal=goal)
    return Monitor(env)


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linearly anneal a hyperparameter (e.g. learning rate) over training,
    as a function of the fraction of training remaining (1.0 -> 0.0)."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


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
        model = RecurrentPPO.load(args.resume, env=env, tensorboard_log=config.LOG_DIR)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=linear_schedule(3e-4, 1e-5),   # anneal LR instead of a fixed 3e-4 for the whole run
            n_steps=256,              # shorter rollout buffer suits BPTT through the LSTM
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,          # Eq. 14, epsilon = 0.2
            ent_coef=0.01,           # keep some exploration going instead of collapsing to a greedy policy early
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                lstm_hidden_size=128,
                n_lstm_layers=1,
                # std starts smaller than SB3's default (1.0) so early rollouts explore with
                # gentle velocity commands instead of constantly saturating at +/-1 (bang-bang
                # flight), which was previously the main cause of early-training collisions.
                log_std_init=-1.0,
            ),
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
