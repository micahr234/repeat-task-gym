"""Task boundaries adapted from mouse-gym's episode repetition protocol."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any, TypeVar

import gymnasium as gym
from gymnasium.utils import seeding

ActType = TypeVar("ActType")

DONE_RUNNING = 0
DONE_TERMINATED = 1
DONE_TRUNCATED = 2


class RepeatTaskEnv(
    gym.Wrapper[dict[str, Any], ActType, Any, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Present repeated episodes as one Gymnasium task.

    ``reset()`` starts a task. ``step()`` returns the standard Gymnasium tuple,
    but ``terminated`` and ``truncated`` describe the task. Observations are
    ``{"observation": original_observation, "episode_done": code}``, where the
    code is 0 (running/reset), 1 (terminated), or 2 (truncated). Termination
    takes precedence if the underlying env sets both flags.

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
            ``action``, ``reward``, ``done``, and ``next_state``. Indices are
            zero-based for episodes; the first actual step has step_index=1.
            States are the original observations and done is the episode code.
            A true result terminates the task and takes precedence over the
            episode budget, matching mouse-gym.
        episode_reset_options: Options passed to every underlying reset.
        task_reset_options: Additional options applied only at task starts.
            Options passed to ``reset(options=...)`` override these for that
            task-start reset only.

    ``reset(seed=...)`` seeds a stream which draws one underlying env seed per
    task. Subsequent ``reset()`` calls advance that stream. Internal episode
    resets use ``seed=None`` so the env's RNG continues within the task.
    """

    def __init__(
        self,
        env: gym.Env[Any, ActType],
        *,
        max_task_episodes: int = 0,
        terminate_task: Callable[..., bool] | None = None,
        episode_reset_options: dict[str, Any] | None = None,
        task_reset_options: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(max_task_episodes, bool) or not isinstance(max_task_episodes, Integral):
            raise TypeError("max_task_episodes must be an integer (0 = unlimited).")
        if max_task_episodes < 0:
            raise ValueError("max_task_episodes must be >= 0 (0 = unlimited).")
        if terminate_task is not None and not callable(terminate_task):
            raise TypeError("terminate_task must be callable or None.")

        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_task_episodes=max_task_episodes,
            terminate_task=terminate_task,
            episode_reset_options=episode_reset_options,
            task_reset_options=task_reset_options,
        )
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "episode_done": gym.spaces.Discrete(3),
            }
        )
        self._max_task_episodes = int(max_task_episodes)
        self._terminate_task = terminate_task
        self._episode_reset_options = dict(episode_reset_options or {})
        self._task_reset_options = dict(task_reset_options or {})
        self._seed_rng, _ = seeding.np_random()
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
        """Start a new task, optionally restarting its reproducible seed stream."""
        self._needs_reset = True
        if seed is not None:
            self._seed_rng, _ = seeding.np_random(seed)
        task_seed = int(self._seed_rng.integers(0, 2**31))
        reset_options = dict(self._episode_reset_options)
        reset_options.update(self._task_reset_options)
        reset_options.update(options or {})
        observation, info = self.env.reset(seed=task_seed, options=reset_options or None)
        self._episode_index = 0
        self._step_index = 0
        self._state = observation
        self._episode_reset_pending = False
        self._needs_reset = False
        return self._observation(observation, DONE_RUNNING), info

    def step(self, action: ActType) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the current episode, or emit its pending reset frame."""
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping and after a task ends.")

        if self._episode_reset_pending:
            observation, info = self.env.reset(
                seed=None,
                options=dict(self._episode_reset_options) or None,
            )
            self._episode_index += 1
            self._step_index = 0
            self._state = observation
            self._episode_reset_pending = False
            return self._observation(observation, DONE_RUNNING), 0.0, False, False, info

        observation, reward, terminated, truncated, info = self.env.step(action)
        episode_done = (
            DONE_TERMINATED if terminated else DONE_TRUNCATED if truncated else DONE_RUNNING
        )
        self._step_index += 1
        reward = float(reward)
        task_terminated = False
        task_truncated = False

        if episode_done != DONE_RUNNING:
            if self._terminate_task is not None:
                task_terminated = bool(
                    self._terminate_task(
                        step_index=self._step_index,
                        episode_index=self._episode_index,
                        state=self._state,
                        action=action,
                        reward=reward,
                        done=episode_done,
                        next_state=observation,
                    )
                )
            task_truncated = (
                not task_terminated
                and self._max_task_episodes > 0
                and self._episode_index + 1 >= self._max_task_episodes
            )
            self._needs_reset = task_terminated or task_truncated
            self._episode_reset_pending = not self._needs_reset

        self._state = observation
        return (
            self._observation(observation, episode_done),
            reward,
            task_terminated,
            task_truncated,
            info,
        )

    @staticmethod
    def _observation(observation: Any, episode_done: int) -> dict[str, Any]:
        return {"observation": observation, "episode_done": episode_done}
