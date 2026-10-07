"""Behavioral tests for the episode/task boundary contract."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

from repeat_task_gym import RepeatTaskEnv


class EpisodeEnv(gym.Env):
    """A deterministic episode with observable resets and terminal states."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, length: int = 2, *, terminated: bool = True, truncated: bool = False):
        self.observation_space = gym.spaces.Discrete(length + 1)
        self.action_space = gym.spaces.Discrete(2)
        self.render_mode = "ansi"
        self.length = length
        self.terminated = terminated
        self.truncated = truncated
        self.position = 0
        self.reset_calls: list[tuple[int | None, dict[str, Any] | None]] = []
        self.actions: list[int] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls.append((seed, options))
        self.position = 0
        return 0, {"reset": True}

    def step(self, action):
        assert self.position < self.length, "Ended episode was stepped without reset"
        self.actions.append(action)
        self.position += 1
        done = self.position == self.length
        return (
            self.position,
            float(self.position),
            done and self.terminated,
            done and self.truncated,
            {"position": self.position},
        )

    def render(self):
        return f"position={self.position}"

    def close(self):
        self.closed = True


@pytest.mark.parametrize("terminated,truncated,code", [(True, False, 1), (False, True, 2), (True, True, 1)])
def test_episode_boundaries_preserve_terminal_observation(terminated, truncated, code):
    base = EpisodeEnv(terminated=terminated, truncated=truncated)
    env = RepeatTaskEnv(base, max_task_episodes=2)
    observation, info = env.reset(seed=7)
    assert observation == {"observation": 0, "episode_done": 0}
    assert observation in env.observation_space
    assert info == {"reset": True}

    assert env.step(0) == ({"observation": 1, "episode_done": 0}, 1.0, False, False, {"position": 1})
    terminal = env.step(1)
    assert terminal == ({"observation": 2, "episode_done": code}, 2.0, False, False, {"position": 2})
    assert len(base.reset_calls) == 1

    reset_frame = env.step(0)
    assert reset_frame == ({"observation": 0, "episode_done": 0}, 0.0, False, False, {"reset": True})
    assert base.actions == [0, 1]
    assert base.reset_calls[-1][0] is None
    assert terminal[0] == {"observation": 2, "episode_done": code}

    env.step(1)
    final = env.step(0)
    assert final == ({"observation": 2, "episode_done": code}, 2.0, False, True, {"position": 2})
    assert final[0] in env.observation_space
    assert len(base.reset_calls) == 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    assert len(base.actions) == 4
    assert len(base.reset_calls) == 2

    observation, _ = env.reset()
    assert observation["episode_done"] == 0
    assert base.reset_calls[-1][0] is not None
    assert env.step(0)[2:4] == (False, False)


def test_reset_required_before_first_step():
    base = EpisodeEnv()
    env = RepeatTaskEnv(base)
    with pytest.raises(gym.error.ResetNeeded, match="reset"):
        env.step(0)
    assert base.actions == []
    assert base.reset_calls == []


def test_unlimited_task_keeps_repeating():
    base = EpisodeEnv(length=1)
    env = RepeatTaskEnv(base)
    env.reset(seed=0)
    for _ in range(20):
        assert env.step(0)[0]["episode_done"] == 1
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation["episode_done"] == 0
        assert (reward, terminated, truncated) == (0.0, False, False)
    assert len(base.reset_calls) == 21


def test_single_episode_task_has_no_reset_frame_before_boundary():
    base = EpisodeEnv(length=1)
    env = RepeatTaskEnv(base, max_task_episodes=1)
    env.reset()
    assert env.step(0)[2:4] == (False, True)
    assert len(base.reset_calls) == 1


@pytest.mark.parametrize("budget", [0, 2, 5])
def test_task_predicate_only_runs_at_episode_end_and_wins_over_budget(budget):
    calls = []

    def terminate_task(**transition):
        calls.append(transition)
        return transition["episode_index"] == 1

    env = RepeatTaskEnv(EpisodeEnv(), max_task_episodes=budget, terminate_task=terminate_task)
    env.reset()
    env.step(0)
    assert calls == []
    assert env.step(1)[2:4] == (False, False)
    assert calls == [{
        "step_index": 2, "episode_index": 0, "state": 1, "action": 1,
        "reward": 2.0, "done": 1, "next_state": 2,
    }]
    env.step(0)
    env.step(0)
    assert len(calls) == 1
    assert env.step(1)[2:4] == (True, False)
    assert calls[-1] == dict(calls[0], episode_index=1)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset()
    env.step(0)
    assert env.step(1)[2:4] == (False, False)
    assert calls[-1]["episode_index"] == 0


def test_predicate_can_terminate_after_underlying_truncation():
    env = RepeatTaskEnv(
        EpisodeEnv(length=1, terminated=False, truncated=True),
        terminate_task=lambda *, done, **kwargs: done == 2,
    )
    env.reset()
    observation, _, terminated, truncated, _ = env.step(0)
    assert observation["episode_done"] == 2
    assert (terminated, truncated) == (True, False)


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_manual_reset_discards_pending_episode_and_task_progress(steps):
    base = EpisodeEnv()
    env = RepeatTaskEnv(base, max_task_episodes=2)
    env.reset(seed=5)
    for _ in range(steps):
        env.step(0)
    env.reset(seed=5)
    resets = len(base.reset_calls)
    assert env.step(1)[0] == {"observation": 1, "episode_done": 0}
    assert env.step(1)[2:4] == (False, False)
    assert len(base.reset_calls) == resets


def test_reset_option_precedence_and_episode_options_are_separate():
    episode_options = {"shared": "episode", "episode": 1}
    task_options = {"shared": "task", "task": 2}
    explicit_options = {"shared": "explicit", "explicit": 3}
    base = EpisodeEnv(length=1)
    env = RepeatTaskEnv(
        base, episode_reset_options=episode_options, task_reset_options=task_options,
    )
    env.reset(options=explicit_options)
    assert base.reset_calls[-1][1] == {
        "shared": "explicit", "episode": 1, "task": 2, "explicit": 3,
    }
    received_options = base.reset_calls[-1][1]
    assert received_options is not None
    received_options["mutated"] = True
    env.step(0)
    env.step(0)
    assert base.reset_calls[-1] == (None, episode_options)
    received_options = base.reset_calls[-1][1]
    assert received_options is not None
    received_options["mutated"] = True
    env.step(0)
    env.step(0)
    assert base.reset_calls[-1][1] == episode_options
    env.reset()
    assert base.reset_calls[-1][1] == {"shared": "task", "episode": 1, "task": 2}
    assert episode_options == {"shared": "episode", "episode": 1}
    assert task_options == {"shared": "task", "task": 2}
    assert explicit_options == {"shared": "explicit", "explicit": 3}


class SeededTaskEnv(gym.Env):
    """Create a new task only on explicit seeds; vary starts on every reset."""

    def __init__(self):
        self.observation_space = gym.spaces.MultiDiscrete([2**31, 2**31])
        self.action_space = gym.spaces.Discrete(2)
        self.seeds = []
        self.task = 0
        self.start = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        if seed is not None:
            self.task = int(self.np_random.integers(2**31))
        self.start = int(self.np_random.integers(2**31))
        return np.array([self.task, self.start]), {}

    def step(self, action):
        return np.array([self.task, self.start]), 1.0, True, False, {}


def test_task_seed_stream_and_varying_episode_resets():
    base = SeededTaskEnv()
    env = RepeatTaskEnv(base, max_task_episodes=2)

    def run(seed):
        first, _ = env.reset(seed=seed)
        env.step(0)
        second, _, _, _, _ = env.step(0)
        assert env.step(0)[2:4] == (False, True)
        third, _ = env.reset()
        return [first["observation"], second["observation"], third["observation"]]

    initial = run(42)
    seeds = base.seeds.copy()
    assert seeds[0] is not None and seeds[1] is None and seeds[2] is not None
    assert seeds[0] != seeds[2]
    assert initial[0][0] == initial[1][0]
    assert initial[0][1] != initial[1][1]
    assert initial[0][0] != initial[2][0]
    assert data_equivalence(initial, run(42))
    assert base.seeds[3:] == seeds
    assert not data_equivalence(initial, run(43))


def test_action_sampling_is_independent_of_task_seed_stream():
    def task_seeds(sample_actions):
        base = SeededTaskEnv()
        env = RepeatTaskEnv(base)
        env.reset(seed=123)
        env.action_space.seed(7)
        if sample_actions:
            for _ in range(100):
                env.action_space.sample()
        env.reset()
        return base.seeds

    assert task_seeds(False) == task_seeds(True)


@pytest.mark.parametrize("space", [
    gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
    gym.spaces.Discrete(3),
    gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.MultiBinary(3))),
    gym.spaces.Dict({
        "episode_done": gym.spaces.Discrete(10),
        "observation": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float64),
    }),
])
def test_original_spaces_observations_actions_and_info_are_preserved(space):
    class PassthroughEnv(gym.Env):
        observation_space = space
        action_space = gym.spaces.Dict({"move": gym.spaces.Box(-1, 1, shape=(2,))})

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.value = space.sample()
            self.info = {"payload": object()}
            return self.value, self.info

        def step(self, action):
            self.action = action
            return self.value, 0.25, False, False, self.info

    base = PassthroughEnv()
    env = RepeatTaskEnv(base)
    assert isinstance(env, gym.Env)
    assert env.action_space is base.action_space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.observation_space["observation"] is base.observation_space
    observation, info = env.reset(seed=0)
    assert observation in env.observation_space
    assert observation["observation"] is base.value
    assert info is base.info
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation in env.observation_space
    assert observation["observation"] is base.value
    assert base.action is action
    assert info is base.info
    assert (reward, terminated, truncated) == (0.25, False, False)


def test_render_metadata_and_close_delegate():
    base = EpisodeEnv()
    env = RepeatTaskEnv(base)
    assert env.unwrapped is base
    assert env.metadata is base.metadata
    assert env.render_mode == "ansi"
    env.reset()
    env.step(0)
    assert env.render() == "position=1"
    env.close()
    assert base.closed


@pytest.mark.parametrize("value,error", [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("2", TypeError), (None, TypeError)])
def test_invalid_episode_budgets(value, error):
    with pytest.raises(error, match="max_task_episodes"):
        RepeatTaskEnv(EpisodeEnv(), max_task_episodes=value)


def test_numpy_integer_budget():
    env = RepeatTaskEnv(EpisodeEnv(length=1), max_task_episodes=cast(Any, np.int64(1)))
    env.reset()
    assert env.step(0)[3] is True


def test_invalid_predicate():
    with pytest.raises(TypeError, match="terminate_task"):
        RepeatTaskEnv(EpisodeEnv(), terminate_task=12)  # type: ignore[arg-type]


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "FrozenLake-v1"])
def test_gymnasium_checker_and_real_environment_rollout(env_id):
    env = RepeatTaskEnv(gym.make(env_id), max_task_episodes=2)
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=0)
        assert observation in env.observation_space
        for _ in range(450):
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert observation in env.observation_space
            if terminated or truncated:
                env.reset()
    finally:
        env.close()
