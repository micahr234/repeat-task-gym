"""Check composition with Gymnasium's own wrappers and vector API."""

import gymnasium as gym
import numpy as np
import pytest

from repeat_task_gym import RepeatTaskEnv


def make_task():
    return RepeatTaskEnv(
        gym.make("CartPole-v1", max_episode_steps=2), max_task_episodes=2,
    )


def test_episode_and_task_statistics_at_their_respective_levels():
    inner = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1", max_episode_steps=2))
    env = gym.wrappers.RecordEpisodeStatistics(
        RepeatTaskEnv(inner, max_task_episodes=2), stats_key="task",
    )
    try:
        env.reset(seed=0)
        for _ in range(4):
            env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        assert (terminated, truncated) == (False, True)
        assert list(inner.return_queue) == [2.0, 2.0]
        assert list(inner.length_queue) == [2, 2]
        assert info["episode"]["r"] == 2.0
        assert info["task"]["r"] == 4.0
        assert info["task"]["l"] == 5
    finally:
        env.close()


def test_flatten_observation():
    env = gym.wrappers.FlattenObservation(make_task())
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (8,)  # Four state values + two one-hot binary fields.
        for _ in range(5):
            observation, _, _, _, _ = env.step(0)
            assert observation in env.observation_space
    finally:
        env.close()


def test_simultaneous_episode_and_task_flags_with_time_limit():
    base = gym.make("FrozenLake-v1", desc=["SG"], is_slippery=False, max_episode_steps=1)
    env = RepeatTaskEnv(
        base, max_task_episodes=1,
        terminate_task=lambda *, reward, episode_terminated, **kwargs: episode_terminated and reward > 0,
    )
    try:
        env.reset(seed=0)
        observation, reward, terminated, truncated, _ = env.step(2)  # Goal at the time limit.
        assert observation["episode_terminated"] is True
        assert observation["episode_truncated"] is True
        assert observation in env.observation_space
        assert (reward, terminated, truncated) == (1.0, True, True)
        with pytest.raises(gym.error.ResetNeeded):
            env.step(2)
        observation, _ = env.reset()
        assert observation["episode_terminated"] is False
        assert observation["episode_truncated"] is False
    finally:
        env.close()


def test_outer_time_limit_can_interrupt_a_task():
    env = gym.wrappers.TimeLimit(make_task(), max_episode_steps=1)
    try:
        env.reset(seed=0)
        observation, _, terminated, truncated, _ = env.step(0)
        assert observation["episode_terminated"] is False
        assert observation["episode_truncated"] is False
        assert (terminated, truncated) == (False, True)
        env.reset()
        observation, _, _, _, _ = env.step(0)
        assert observation["episode_terminated"] is False
        assert observation["episode_truncated"] is False
    finally:
        env.close()


def test_outer_autoreset_starts_new_tasks():
    env = gym.wrappers.Autoreset(make_task())
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(0)
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation["episode_terminated"] is False
        assert observation["episode_truncated"] is False
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert env.step(0)[1] == 1.0
    finally:
        env.close()


def test_sync_vector_task_boundaries_and_automatic_task_reset():
    env = gym.vector.SyncVectorEnv([make_task, make_task])
    try:
        observation, _ = env.reset(seed=10)
        assert observation in env.observation_space
        for step in range(5):
            observation, rewards, terminated, truncated, _ = env.step(np.array([0, 1]))
            assert observation in env.observation_space
            np.testing.assert_array_equal(terminated, [False, False])
            np.testing.assert_array_equal(truncated, [step == 4, step == 4])
            np.testing.assert_array_equal(observation["episode_terminated"], [False, False])
            np.testing.assert_array_equal(observation["episode_truncated"], [step in (1, 4)] * 2)
            np.testing.assert_array_equal(rewards, [0.0 if step == 2 else 1.0] * 2)
        observation, rewards, terminated, truncated, _ = env.step(np.array([0, 1]))
        np.testing.assert_array_equal(observation["episode_terminated"], [False, False])
        np.testing.assert_array_equal(observation["episode_truncated"], [False, False])
        np.testing.assert_array_equal(rewards, [0.0, 0.0])
        assert not terminated.any() and not truncated.any()
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["per_episode", "per_task", "constant"])
def test_spec_recreates_the_wrapper(mode):
    original = RepeatTaskEnv(
        gym.make("CartPole-v1", max_episode_steps=2),
        max_task_episodes=2, episode_seed_mode=mode,
        episode_reset_options={"low": -0.025, "high": 0.025},
    )
    try:
        assert original.spec is not None
        recreated = gym.make(original.spec)
        try:
            assert isinstance(recreated, RepeatTaskEnv)
            left, _ = original.reset(seed=7)
            right, _ = recreated.reset(seed=7)
            np.testing.assert_array_equal(left["observation"], right["observation"])
            for step in range(5):
                left_step = original.step(0)
                right_step = recreated.step(0)
                np.testing.assert_array_equal(left_step[0]["observation"], right_step[0]["observation"])
                assert left_step[0]["episode_terminated"] == right_step[0]["episode_terminated"]
                assert left_step[0]["episode_truncated"] == right_step[0]["episode_truncated"]
                assert left_step[1:4] == right_step[1:4]
                assert right_step[3] is (step == 4)
            left, _ = original.reset()
            right, _ = recreated.reset()
            np.testing.assert_array_equal(left["observation"], right["observation"])
        finally:
            recreated.close()
    finally:
        original.close()


@pytest.mark.parametrize("mode", ["per_episode", "per_task", "constant"])
def test_cartpole_episode_starts_follow_seed_mode(mode):
    env = RepeatTaskEnv(
        gym.make("CartPole-v1", max_episode_steps=1),
        max_task_episodes=2, episode_seed_mode=mode,
    )
    try:
        first, _ = env.reset(seed=42)
        env.step(0)
        second, _, _, _, _ = env.step(0)
        if mode != "per_episode":
            np.testing.assert_array_equal(first["observation"], second["observation"])
        else:
            assert not np.array_equal(first["observation"], second["observation"])
        assert env.step(1)[3] is True
        next_task, _ = env.reset()
        assert np.array_equal(first["observation"], next_task["observation"]) == (mode == "constant")
        replay, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first["observation"], replay["observation"])
        env.step(0)
        replay_second, _, _, _, _ = env.step(0)
        np.testing.assert_array_equal(second["observation"], replay_second["observation"])
    finally:
        env.close()


@pytest.mark.parametrize("seed", [-1, 1.5, "123"])
def test_invalid_reset_seeds_use_gymnasium_validation(seed):
    env = make_task()
    try:
        with pytest.raises(gym.error.Error):
            env.reset(seed=seed)
    finally:
        env.close()
