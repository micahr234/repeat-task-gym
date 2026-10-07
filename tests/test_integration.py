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
        assert observation.shape == (7,)
        for _ in range(5):
            observation, _, _, _, _ = env.step(0)
            assert observation in env.observation_space
    finally:
        env.close()


def test_outer_time_limit_can_interrupt_a_task():
    env = gym.wrappers.TimeLimit(make_task(), max_episode_steps=1)
    try:
        env.reset(seed=0)
        observation, _, terminated, truncated, _ = env.step(0)
        assert observation["episode_done"] == 0
        assert (terminated, truncated) == (False, True)
        env.reset()
        assert env.step(0)[0]["episode_done"] == 0
    finally:
        env.close()


def test_outer_autoreset_starts_new_tasks():
    env = gym.wrappers.Autoreset(make_task())
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(0)
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation["episode_done"] == 0
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
            np.testing.assert_array_equal(observation["episode_done"], [2 if step in (1, 4) else 0] * 2)
            np.testing.assert_array_equal(rewards, [0.0 if step == 2 else 1.0] * 2)
        observation, rewards, terminated, truncated, _ = env.step(np.array([0, 1]))
        np.testing.assert_array_equal(observation["episode_done"], [0, 0])
        np.testing.assert_array_equal(rewards, [0.0, 0.0])
        assert not terminated.any() and not truncated.any()
    finally:
        env.close()


def test_spec_recreates_the_wrapper():
    original = make_task()
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
                assert left_step[1:4] == right_step[1:4]
                assert right_step[3] is (step == 4)
        finally:
            recreated.close()
    finally:
        original.close()


@pytest.mark.parametrize("seed", [-1, 1.5, "123"])
def test_invalid_reset_seeds_use_gymnasium_validation(seed):
    env = make_task()
    try:
        with pytest.raises(gym.error.Error):
            env.reset(seed=seed)
    finally:
        env.close()
