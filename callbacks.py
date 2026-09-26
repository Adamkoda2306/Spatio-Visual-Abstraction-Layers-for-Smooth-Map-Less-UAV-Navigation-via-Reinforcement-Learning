"""Shared SB3 training callbacks."""

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """Prints total reward and step count every time an episode finishes.

    Relies on the `Monitor` wrapper, which injects an `episode` key
    (`{"r": total_reward, "l": length}`) into `info` on the terminal step.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episode_count += 1
                status = self._episode_status(info)
                print(f"[Episode {self.episode_count}] reward={episode['r']:.2f}  "
                      f"steps={episode['l']}  status={status}")
        return True

    @staticmethod
    def _episode_status(info) -> str:
        if info.get("success"):
            return "REACHED GOAL"
        if info.get("collided"):
            return "COLLISION"
        return "STOPPED"  # timeout / truncated without reaching the goal or colliding
