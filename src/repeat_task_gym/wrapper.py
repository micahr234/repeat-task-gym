"""Repeat episodes within a task using Gymnasium's standard API."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any, Literal, TypeVar

import gymnasium as gym
from gymnasium.utils import seeding

ActType = TypeVar("ActType")


class RepeatTaskEnv(
    gym.Wrapper[dict[str, Any], ActType, Any, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Present repeated episodes as one Gymnasium task.

    ``reset()`` starts a task. ``step()`` returns the standard Gymnasium tuple,
    but ``terminated`` and ``truncated`` describe the task. Observations contain
    the original ``observation`` plus the boolean ``episode_terminated`` and
    ``episode_truncated`` fields, each in a ``Discrete(2)`` space. Both episode
    flags are preserved when the underlying env sets both, and both are false
    on initial observations and episode reset frames.

    After a nonfinal episode ends, the next ``step()`` ignores its action and
    returns a reset frame with reward 0.0 and both task flags false. The final
    episode's observation is always preserved. After a task ends, callers must
    call ``reset()`` before stepping again.

    Args:
        env: An existing Gymnasium environment, including any episode wrappers.
        max_task_episodes: Completed episodes before task truncation. Zero
            (the default) means unlimited.
        terminate_task: Optional predicate called only on episode-end steps
            with keyword arguments ``step_index``, ``episode_index``, ``state``,
            ``action``, ``reward``, ``episode_terminated``,
            ``episode_truncated``, and ``next_state``. Indices are zero-based
            for episodes; the first actual step has step_index=1. States are
            the original observations; episode flags are booleans.
            A true result terminates the task. If the episode budget is also
            reached, both task flags are true.
        episode_reset_options: Options passed to every underlying reset.
            Public ``reset(options=...)`` adds or overrides options for that
            task's first episode only; later episodes use these defaults.
            A ``seed`` entry is not allowed; use public ``reset(seed=...)``.
        episode_seed_mode: Controls when to draw from the wrapper's single
            seed generator: ``"per_episode"`` (the default) draws at every
            episode start; ``"per_task"`` draws at every public reset;
            ``"constant"`` draws only on the first reset or after reseeding.
            Between draws, every underlying reset reuses the last drawn seed.

    Every public ``reset()``, including ``reset(seed=None)``, starts a new task.
    An explicit seed restarts the seed generator and draws a new episode seed
    in every mode. The environment receives the drawn seed, not the input seed.
    Generated episode seeds are independent of earlier episode lengths and
    environment RNG use. In per_episode mode, extra episodes advance the shared
    stream and therefore affect the next task's seed. Seedless resets continue
    the stream, or retain the same seed in constant mode. Preserving a task's
    map, goal, or dynamics depends on the environment; use ``reset(options=...)``
    when it supports a separate request to generate a new instance. The caller
    manages agent memory: retain it between episodes and clear it for a new task.
    """

    def __init__(
        self,
        env: gym.Env[Any, ActType],
        *,
        max_task_episodes: int = 0,
        terminate_task: Callable[..., bool] | None = None,
        episode_reset_options: dict[str, Any] | None = None,
        episode_seed_mode: Literal["per_episode", "per_task", "constant"] = "per_episode",
    ) -> None:
        if isinstance(max_task_episodes, bool) or not isinstance(max_task_episodes, Integral):
            raise TypeError("max_task_episodes must be an integer (0 = unlimited).")
        if max_task_episodes < 0:
            raise ValueError("max_task_episodes must be >= 0 (0 = unlimited).")
        if terminate_task is not None and not callable(terminate_task):
            raise TypeError("terminate_task must be callable or None.")
        if episode_seed_mode not in ("per_episode", "per_task", "constant"):
            raise ValueError("episode_seed_mode must be 'per_episode', 'per_task', or 'constant'.")
        episode_options = dict(episode_reset_options or {})
        if "seed" in episode_options:
            raise ValueError("Pass seed to reset(seed=...), not episode_reset_options.")

        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_task_episodes=max_task_episodes,
            terminate_task=terminate_task,
            episode_reset_options=episode_reset_options,
            episode_seed_mode=episode_seed_mode,
        )
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "episode_terminated": gym.spaces.Discrete(2),
                "episode_truncated": gym.spaces.Discrete(2),
            }
        )
        self._max_task_episodes = int(max_task_episodes)
        self._terminate_task = terminate_task
        self._episode_reset_options = episode_options
        self._episode_seed_mode = episode_seed_mode
        self._seed_rng, _ = seeding.np_random()
        self._episode_seed: int | None = None
        self._needs_reset = True
        self._episode_reset_pending = False
        self._episode_index = 0
        self._step_index = 0
        self._state: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new task, even when seed is None.

        An integer seed reseeds the wrapper's generator and draws a new seed
        in every mode. With None, per_episode/per_task draw the next seed and
        constant reuses the last seed (drawing one if none exists yet).
        The environment receives the drawn seed, not the supplied seed itself.
        Options override episode_reset_options for this initial reset only.
        """
        self._needs_reset = True
        if seed is not None:
            self._seed_rng, _ = seeding.np_random(seed)
            self._episode_seed = None
        if self._episode_seed_mode != "constant" or self._episode_seed is None:
            self._episode_seed = int(self._seed_rng.integers(0, 2**31))
        reset_options = dict(self._episode_reset_options)
        reset_options.update(options or {})
        observation, info = self.env.reset(seed=self._episode_seed, options=reset_options or None)
        self._episode_index = 0
        self._step_index = 0
        self._state = observation
        self._episode_reset_pending = False
        self._needs_reset = False
        return self._observation(observation, False, False), info

    def step(self, action: ActType) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the current episode, or emit its pending reset frame."""
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping and after a task ends.")

        if self._episode_reset_pending:
            if self._episode_seed_mode == "per_episode":
                self._episode_seed = int(self._seed_rng.integers(0, 2**31))
            observation, info = self.env.reset(
                seed=self._episode_seed,
                options=dict(self._episode_reset_options) or None,
            )
            self._episode_index += 1
            self._step_index = 0
            self._state = observation
            self._episode_reset_pending = False
            return self._observation(observation, False, False), 0.0, False, False, info

        observation, reward, terminated, truncated, info = self.env.step(action)
        episode_terminated = bool(terminated)
        episode_truncated = bool(truncated)
        self._step_index += 1
        reward = float(reward)
        task_terminated = False
        task_truncated = False

        if episode_terminated or episode_truncated:
            if self._terminate_task is not None:
                task_terminated = bool(
                    self._terminate_task(
                        step_index=self._step_index,
                        episode_index=self._episode_index,
                        state=self._state,
                        action=action,
                        reward=reward,
                        episode_terminated=episode_terminated,
                        episode_truncated=episode_truncated,
                        next_state=observation,
                    )
                )
            task_truncated = (
                self._max_task_episodes > 0
                and self._episode_index + 1 >= self._max_task_episodes
            )
            self._needs_reset = task_terminated or task_truncated
            self._episode_reset_pending = not self._needs_reset

        self._state = observation
        return (
            self._observation(observation, episode_terminated, episode_truncated),
            reward,
            task_terminated,
            task_truncated,
            info,
        )

    @staticmethod
    def _observation(
        observation: Any, episode_terminated: bool, episode_truncated: bool,
    ) -> dict[str, Any]:
        return {
            "observation": observation,
            "episode_terminated": episode_terminated,
            "episode_truncated": episode_truncated,
        }
