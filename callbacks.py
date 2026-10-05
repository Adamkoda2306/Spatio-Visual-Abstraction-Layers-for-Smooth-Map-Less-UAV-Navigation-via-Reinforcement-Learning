"""Shared SB3 training callbacks."""

import csv
import os

from stable_baselines3.common.callbacks import BaseCallback

import config


class EpisodeLogCallback(BaseCallback):
    """Prints per-episode outcome and keeps a rolling GOAL SUCCESS RATE.

    Relies on the `Monitor` wrapper, which injects an `episode` key
    (`{"r": total_reward, "l": length}`) into `info` on the terminal step,
    and on airsim_env.AirSimUAVEnv.step() setting `success`, `collided`,
    `timed_out`, and `distance` in `info` on every step.

    Every episode is appended to logs/episode_metrics.csv, and every
    `bucket_size` episodes (default 100) a summary -- success/collision/
    timeout rate and mean final/minimum distance over that block -- is
    printed and appended to logs/goal_success_rate.csv, so the trend in
    GOAL SUCCESS RATE across training can be read straight from the
    console or plotted later from the CSV.
    """

    def __init__(self, verbose=0, bucket_size=100, log_dir=None):
        super().__init__(verbose)
        self.episode_count = 0
        self.bucket_size = bucket_size
        self.log_dir = log_dir or config.LOG_DIR

        self._min_dist = {}  # env_idx -> running minimum distance this episode
        self._bucket = []    # list of dicts, one per episode, cleared every bucket_size

        os.makedirs(self.log_dir, exist_ok=True)
        self._episode_csv_path = os.path.join(self.log_dir, "episode_metrics.csv")
        self._bucket_csv_path = os.path.join(self.log_dir, "goal_success_rate.csv")
        self._init_csv(self._episode_csv_path,
                        ["episode", "reward", "steps", "final_distance", "minimum_distance",
                         "success", "collided", "timed_out"])
        self._init_csv(self._bucket_csv_path,
                        ["episode_start", "episode_end", "success_rate", "collision_rate",
                         "timeout_rate", "mean_final_distance", "mean_minimum_distance"])

    @staticmethod
    def _init_csv(path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _on_step(self) -> bool:
        for idx, info in enumerate(self.locals.get("infos", [])):
            dist = info.get("distance")
            if dist is not None:
                self._min_dist[idx] = min(self._min_dist.get(idx, dist), dist)

            episode = info.get("episode")
            if episode is None:
                continue

            self.episode_count += 1
            final_dist = dist if dist is not None else float("nan")
            min_dist = self._min_dist.pop(idx, final_dist)
            success = bool(info.get("success"))
            collided = bool(info.get("collided"))
            timed_out = bool(info.get("timed_out"))

            print(f"Episode {self.episode_count}, Reward: {episode['r']:.2f}, Steps: {episode['l']}")
            print(f"Final distance: {final_dist:.2f} m, Minimum distance: {min_dist:.2f} m")
            print(f"STATUS => SUCCESS: {success}, COLLISION: {collided}, TIMEOUT: {timed_out}")
            print()

            self._append_csv(self._episode_csv_path, [
                self.episode_count, f"{episode['r']:.4f}", episode["l"],
                f"{final_dist:.4f}", f"{min_dist:.4f}", success, collided, timed_out,
            ])

            self._bucket.append(dict(
                final_dist=final_dist, min_dist=min_dist,
                success=success, collided=collided, timed_out=timed_out,
            ))
            if len(self._bucket) >= self.bucket_size:
                self._flush_bucket()

        return True

    def _flush_bucket(self):
        n = len(self._bucket)
        start = self.episode_count - n + 1
        end = self.episode_count

        success_rate = 100.0 * sum(e["success"] for e in self._bucket) / n
        collision_rate = 100.0 * sum(e["collided"] for e in self._bucket) / n
        timeout_rate = 100.0 * sum(e["timed_out"] for e in self._bucket) / n
        mean_final = sum(e["final_dist"] for e in self._bucket) / n
        mean_min = sum(e["min_dist"] for e in self._bucket) / n

        print("=" * 60)
        print(f"Episodes {start}-{end}: "
              f"{success_rate:.0f}% success  {collision_rate:.0f}% collision  "
              f"{timeout_rate:.0f}% timeout  "
              f"mean_final={mean_final:.2f}m  mean_min={mean_min:.2f}m")
        print("=" * 60)

        self._append_csv(self._bucket_csv_path, [
            start, end, f"{success_rate:.2f}", f"{collision_rate:.2f}", f"{timeout_rate:.2f}",
            f"{mean_final:.4f}", f"{mean_min:.4f}",
        ])

        self._bucket.clear()

    @staticmethod
    def _append_csv(path, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _episode_status(info) -> str:
        if info.get("success"):
            return "REACHED GOAL"
        if info.get("collided"):
            return "COLLISION"
        return "STOPPED"  # timeout / truncated without reaching the goal or colliding
