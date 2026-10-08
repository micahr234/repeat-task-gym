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


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_episode_boundaries_preserve_terminal_observation(terminated, truncated):
    base = EpisodeEnv(terminated=terminated, truncated=truncated)
    env = RepeatTaskEnv(base, max_task_episodes=2)
    observation, info = env.reset(seed=7)
    initial = {"observation": 0, "episode_terminated": False, "episode_truncated": False}
    running = dict(initial, observation=1)
    ended = {"observation": 2, "episode_terminated": terminated, "episode_truncated": truncated}
    assert observation == initial
    assert isinstance(env.observation_space, gym.spaces.Dict)
    termination_space = env.observation_space["episode_terminated"]
    truncation_space = env.observation_space["episode_truncated"]
    assert isinstance(termination_space, gym.spaces.Discrete)
    assert isinstance(truncation_space, gym.spaces.Discrete)
    assert termination_space.n == 2
    assert truncation_space.n == 2
    assert observation in env.observation_space
    assert info == {"reset": True}

    assert env.step(0) == (running, 1.0, False, False, {"position": 1})
    terminal = env.step(1)
    assert terminal == (ended, 2.0, False, False, {"position": 2})
    assert terminal[0] in env.observation_space
    assert terminal[0]["episode_terminated"] is terminated
    assert terminal[0]["episode_truncated"] is truncated
    assert len(base.reset_calls) == 1

    reset_frame = env.step(0)
    assert reset_frame == (initial, 0.0, False, False, {"reset": True})
    assert base.actions == [0, 1]
    assert isinstance(base.reset_calls[-1][0], int)
    assert base.reset_calls[-1][0] != base.reset_calls[0][0]
    assert terminal[0] == ended

    env.step(1)
    final = env.step(0)
    assert final == (ended, 2.0, False, True, {"position": 2})
    assert final[0] in env.observation_space
    assert len(base.reset_calls) == 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    assert len(base.actions) == 4
    assert len(base.reset_calls) == 2

    observation, _ = env.reset()
    assert observation["episode_terminated"] is False
    assert observation["episode_truncated"] is False
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
        observation, _, _, _, _ = env.step(0)
        assert observation["episode_terminated"] is True
        assert observation["episode_truncated"] is False
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation["episode_terminated"] is False
        assert observation["episode_truncated"] is False
        assert (reward, terminated, truncated) == (0.0, False, False)
    assert len(base.reset_calls) == 21


def test_single_episode_task_has_no_reset_frame_before_boundary():
    base = EpisodeEnv(length=1)
    env = RepeatTaskEnv(base, max_task_episodes=1)
    env.reset()
    assert env.step(0)[2:4] == (False, True)
    assert len(base.reset_calls) == 1


@pytest.mark.parametrize("budget", [0, 2, 5])
@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_callbacks_run_every_step_and_budget_is_independent(budget, terminated, truncated):
    reward_calls = []
    terminate_calls = []

    def reward_transform(**transition):
        reward_calls.append(dict(transition))
        return transition["reward"] + 10.0

    def terminate_task(**transition):
        terminate_calls.append(dict(transition))
        return transition["episode_index"] == 1 and (
            transition["episode_terminated"] or transition["episode_truncated"]
        )

    env = RepeatTaskEnv(
        EpisodeEnv(terminated=terminated, truncated=truncated),
        max_task_episodes=budget,
        reward_transform=reward_transform,
        terminate_task=terminate_task,
    )
    env.reset()
    assert reward_calls == [] and terminate_calls == []

    assert env.step(0)[1:4] == (11.0, False, False)
    assert env.step(1)[1:4] == (12.0, False, False)
    reset_frame = env.step(9)
    assert reset_frame[1:4] == (10.0, False, False)
    assert env.step(0)[1:4] == (11.0, False, False)
    assert env.step(1)[1:4] == (12.0, True, budget == 2)

    assert [call["step_index"] for call in terminate_calls] == [1, 2, 0, 1, 2]
    assert reward_calls == [
        {
            "step_index": 1, "episode_index": 0, "state": 0, "action": 0,
            "reward": 1.0, "episode_terminated": False, "episode_truncated": False, "next_state": 1,
        },
        {
            "step_index": 2, "episode_index": 0, "state": 1, "action": 1,
            "reward": 2.0, "episode_terminated": terminated, "episode_truncated": truncated, "next_state": 2,
        },
        {
            "step_index": 0, "episode_index": 1, "state": 2, "action": None,
            "reward": 0.0, "episode_terminated": False, "episode_truncated": False, "next_state": 0,
        },
        {
            "step_index": 1, "episode_index": 1, "state": 0, "action": 0,
            "reward": 1.0, "episode_terminated": False, "episode_truncated": False, "next_state": 1,
        },
        {
            "step_index": 2, "episode_index": 1, "state": 1, "action": 1,
            "reward": 2.0, "episode_terminated": terminated, "episode_truncated": truncated, "next_state": 2,
        },
    ]
    assert terminate_calls == [dict(call, reward=call["reward"] + 10.0) for call in reward_calls]
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset()
    assert env.step(0)[1:4] == (11.0, False, False)
    assert terminate_calls[-1]["episode_index"] == 0


def test_terminate_task_can_end_mid_episode_or_on_a_reset_frame():
    mid = RepeatTaskEnv(
        EpisodeEnv(length=3),
        terminate_task=lambda *, step_index, **kwargs: step_index == 1,
    )
    mid.reset()
    observation, _, terminated, truncated, _ = mid.step(0)
    assert observation["episode_terminated"] is False
    assert observation["episode_truncated"] is False
    assert (terminated, truncated) == (True, False)
    with pytest.raises(gym.error.ResetNeeded):
        mid.step(0)

    reset_frame = RepeatTaskEnv(
        EpisodeEnv(length=1),
        max_task_episodes=5,
        terminate_task=lambda *, step_index, episode_index, **kwargs: (
            episode_index == 1 and step_index == 0
        ),
    )
    reset_frame.reset()
    assert reset_frame.step(0)[2:4] == (False, False)
    observation, reward, terminated, truncated, _ = reset_frame.step(7)
    assert observation == {"observation": 0, "episode_terminated": False, "episode_truncated": False}
    assert (reward, terminated, truncated) == (0.0, True, False)


def test_predicate_can_terminate_after_underlying_truncation():
    env = RepeatTaskEnv(
        EpisodeEnv(length=1, terminated=False, truncated=True),
        terminate_task=lambda *, episode_truncated, **kwargs: episode_truncated,
    )
    env.reset()
    observation, _, terminated, truncated, _ = env.step(0)
    assert observation["episode_terminated"] is False
    assert observation["episode_truncated"] is True
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
    assert env.step(1)[0] == {"observation": 1, "episode_terminated": False, "episode_truncated": False}
    assert env.step(1)[2:4] == (False, False)
    assert len(base.reset_calls) == resets


@pytest.mark.parametrize("mode", ["per_episode", "per_task", "constant"])
def test_reset_option_precedence_and_episode_options_are_separate(mode):
    episode_options = {"shared": "episode", "episode": 1}
    env_options = {"shared": "episode", "episode": 1}
    explicit_options = {"shared": "explicit", "explicit": 3}
    base = EpisodeEnv(length=1)
    env = RepeatTaskEnv(
        base, episode_reset_options=episode_options, episode_seed_mode=mode,
    )
    env.reset(options=explicit_options)
    first_seed = base.reset_calls[-1][0]
    assert base.reset_calls[-1][1] == {
        "shared": "explicit", "episode": 1, "explicit": 3,
    }
    received_options = base.reset_calls[-1][1]
    assert received_options is not None
    received_options["mutated"] = True
    env.step(0)
    env.step(0)
    assert isinstance(base.reset_calls[-1][0], int)
    if mode != "per_episode":
        assert base.reset_calls[-1][0] == first_seed
    assert base.reset_calls[-1][1] == env_options
    received_options = base.reset_calls[-1][1]
    assert received_options is not None
    received_options["mutated"] = True
    env.step(0)
    env.step(0)
    assert base.reset_calls[-1][1] == env_options
    env.reset()
    assert base.reset_calls[-1][1] == env_options
    assert episode_options == env_options
    assert explicit_options == {"shared": "explicit", "explicit": 3}


class SeededTaskEnv(gym.Env):
    """Generate tasks on request, and sample episode starts using the reset seed."""

    def __init__(self, length: int = 1):
        self.observation_space = gym.spaces.MultiDiscrete([2**31, 2**31])
        self.action_space = gym.spaces.Discrete(2)
        self.seeds = []
        self.task = None
        self.start = 0
        self.length = length
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        if self.task is None or (options or {}).get("regenerate_task", False):
            self.task = int(self.np_random.integers(2**31))
        self.start = int(self.np_random.integers(2**31))
        self.steps = 0
        return np.array([self.task, self.start]), {}

    def step(self, action):
        self.np_random.random(int(action) + 1)
        self.steps += 1
        return np.array([self.task, self.start]), 1.0, self.steps == self.length, False, {}


@pytest.mark.parametrize("reset_kwargs", [{}, {"seed": None}], ids=["implicit-none", "explicit-none"])
def test_seed_stream_and_per_episode_resets(reset_kwargs):
    base = SeededTaskEnv()
    env = RepeatTaskEnv(base, max_task_episodes=2)

    def run(seed):
        first, _ = env.reset(seed=seed, options={"regenerate_task": True})
        env.step(0)
        second, _, _, _, _ = env.step(0)
        assert env.step(0)[2:4] == (False, True)
        third, _ = env.reset(**reset_kwargs, options={"regenerate_task": True})
        return [first["observation"], second["observation"], third["observation"]]

    initial = run(42)
    seeds = base.seeds.copy()
    assert all(isinstance(seed, int) for seed in seeds)
    assert len(set(seeds)) == 3
    assert initial[0][0] == initial[1][0]
    assert initial[0][1] != initial[1][1]
    assert initial[0][0] != initial[2][0]
    assert data_equivalence(initial, run(42))
    assert base.seeds[3:] == seeds
    assert not data_equivalence(initial, run(43))


@pytest.mark.parametrize("reset_kwargs", [{}, {"seed": None}], ids=["implicit-none", "explicit-none"])
def test_per_task_seed_replays_procedural_episodes_and_changes_on_new_tasks(reset_kwargs):
    base = SeededTaskEnv(length=2)
    env = RepeatTaskEnv(
        base, max_task_episodes=2, episode_seed_mode="per_task",
        episode_reset_options={"regenerate_task": True},
    )

    def run(seed):
        first, _ = env.reset(seed=seed)
        env.step(0)
        env.step(1)
        second, _, _, _, _ = env.step(0)
        env.step(1)
        assert env.step(1)[2:4] == (False, True)
        third, _ = env.reset(**reset_kwargs)
        env.step(1)
        env.step(0)
        fourth, _, _, _, _ = env.step(0)
        env.step(0)
        assert env.step(0)[2:4] == (False, True)
        return [obs["observation"] for obs in (first, second, third, fourth)]

    initial = run(42)
    seeds = base.seeds.copy()
    assert isinstance(seeds[0], int)
    assert seeds[0] == seeds[1]
    assert seeds[0] != seeds[2]
    assert seeds[2] == seeds[3]
    np.testing.assert_array_equal(initial[0], initial[1])
    np.testing.assert_array_equal(initial[2], initial[3])
    assert not data_equivalence(initial[0], initial[2])
    assert data_equivalence(initial, run(42))
    assert base.seeds[4:] == seeds
    assert not data_equivalence(initial, run(43))


def test_constant_seed_persists_across_tasks_until_public_reseed():
    base = SeededTaskEnv()
    env = RepeatTaskEnv(
        base, max_task_episodes=2, episode_seed_mode="constant",
        episode_reset_options={"regenerate_task": True},
    )

    def run(**reset_kwargs):
        first, _ = env.reset(**reset_kwargs)
        env.step(0)
        second, _, _, _, _ = env.step(0)
        assert env.step(0)[3] is True
        return [first["observation"], second["observation"]], base.seeds[-2:]

    # Constant mode must also initialize itself without an explicit seed.
    initial, seeds = run()
    assert isinstance(seeds[0], int)
    assert seeds[0] == seeds[1]
    np.testing.assert_array_equal(initial[0], initial[1])
    replay, replay_seeds = run(seed=None)
    assert data_equivalence(initial, replay)
    assert seeds == replay_seeds

    reseeded, reseeded_seeds = run(seed=42)
    assert reseeded_seeds[0] == reseeded_seeds[1]
    continued, continued_seeds = run()
    assert data_equivalence(reseeded, continued)
    assert reseeded_seeds == continued_seeds

    changed, changed_seeds = run(seed=7)
    assert changed_seeds != reseeded_seeds
    assert not data_equivalence(reseeded, changed)
    changed_continued, changed_continued_seeds = run(seed=None)
    assert data_equivalence(changed, changed_continued)
    assert changed_seeds == changed_continued_seeds

    replay, replay_seeds = run(seed=42)
    assert data_equivalence(reseeded, replay)
    assert reseeded_seeds == replay_seeds


@pytest.mark.parametrize("mode", ["per_episode", "per_task", "constant"])
def test_episode_resets_are_independent_of_previous_steps_and_random_draws(mode):
    def run(length, action):
        base = SeededTaskEnv(length=length)
        env = RepeatTaskEnv(base, episode_seed_mode=mode)
        first, _ = env.reset(seed=42, options={"regenerate_task": True})
        for _ in range(length):
            env.step(action)
        second, _, _, _, _ = env.step(0)
        return [first["observation"], second["observation"]], base.seeds

    short, short_seeds = run(1, 0)
    long, long_seeds = run(10, 1)
    assert data_equivalence(short, long)
    assert short_seeds == long_seeds


@pytest.mark.parametrize("mode", ["per_episode", "per_task", "constant"])
def test_episode_count_advances_seed_stream_only_in_per_episode_mode(mode):
    def run(episodes):
        base = SeededTaskEnv()
        env = RepeatTaskEnv(base, episode_seed_mode=mode)
        env.reset(seed=42, options={"regenerate_task": True})
        for _ in range(episodes):
            env.step(0)
            env.step(0)
        next_task, _ = env.reset(options={"regenerate_task": True})
        return next_task["observation"], base.seeds[-1]

    short, short_seed = run(1)
    long, long_seed = run(10)
    assert data_equivalence(short, long) == (mode != "per_episode")
    assert (short_seed == long_seed) == (mode != "per_episode")


@pytest.mark.parametrize("seed", [0, 42])
def test_modes_draw_from_the_same_stream_at_their_scheduled_boundaries(seed):
    def recorded_seeds(mode, task_lengths):
        base = SeededTaskEnv()
        env = RepeatTaskEnv(base, episode_seed_mode=mode)
        for task_index, episodes in enumerate(task_lengths):
            env.reset(seed=seed if task_index == 0 else None)
            for _ in range(episodes - 1):
                env.step(0)  # End this episode.
                env.step(0)  # Reset the next episode.
            env.step(0)
        return base.seeds

    draws = recorded_seeds("per_task", [1] * 6)
    assert len(set(draws)) == 6
    assert draws[0] != seed  # The public seed seeds the RNG, not the environment.
    assert recorded_seeds("per_episode", [3, 1, 2]) == draws
    assert recorded_seeds("per_episode", [2, 4]) == draws
    assert recorded_seeds("per_task", [3, 1, 2]) == [
        draws[0], draws[0], draws[0], draws[1], draws[2], draws[2],
    ]
    assert recorded_seeds("constant", [3, 1, 2]) == [draws[0]] * 6


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
        "episode_terminated": gym.spaces.Discrete(10),
        "episode_truncated": gym.spaces.MultiBinary(2),
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


def test_invalid_reward_transform():
    with pytest.raises(TypeError, match="reward_transform"):
        RepeatTaskEnv(EpisodeEnv(), reward_transform=12)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["unknown", "varying", "sequence", "", None, True, 1])
def test_invalid_episode_seed_mode(mode):
    with pytest.raises(ValueError, match="episode_seed_mode"):
        RepeatTaskEnv(EpisodeEnv(), episode_seed_mode=mode)


@pytest.mark.parametrize("seed", [None, 0, 123])
def test_seed_in_episode_reset_options_is_rejected(seed):
    with pytest.raises(ValueError, match="Pass seed to reset"):
        RepeatTaskEnv(EpisodeEnv(), episode_reset_options={"seed": seed})


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
