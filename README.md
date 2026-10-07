# Repeat Task Gym

`repeat-task-gym` concatenates episodes of an existing Gymnasium environment
into one task. This lets us measure learning progress with meta-RL using many
existing environments with little additional code.

**Steps make up an episode. Episodes make up a task.** The question is whether
experience in an early episode helps the agent improve in later episodes.

Imagine commuting to a new workplace. You try different routes on the first
few mornings, then use what you learned to arrive on time. Each commute is an
episode; the succession of commutes is a task. One journey shows how you did
that day. The task shows whether you learned. The
[first notebook](examples/00_episodes_to_tasks.ipynb) models this situation.

## From an MDP to a partially observable learning task

An MDP with an unknown goal, map, or dynamics becomes a POMDP learning problem
when the agent must infer that information from experience. Repeating episodes
of the same instance lets it discover and reuse that information. Existing
POMDP environments can also be repeated. The wrapper preserves the original
observations; the environment determines what remains hidden and learnable.

## Install

Requires Python 3.11+ and Gymnasium 1.0+.

```bash
pip install -e .
pip install -e ".[examples]"  # Include notebook dependencies.
```

## Quick start

```python
import gymnasium as gym
from repeat_task_gym import RepeatTaskEnv

env = RepeatTaskEnv(
    gym.make("CartPole-v1", max_episode_steps=50), max_task_episodes=3
)
try:
    observation, info = env.reset()
    episode_return = 0.0
    while True:
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        episode_return += reward
        if observation["episode_terminated"] or observation["episode_truncated"]:
            print("Episode return:", episode_return)
            episode_return = 0.0
        if terminated or truncated:
            break
finally:
    env.close()
```

This random policy traces three episodes of one task; it does not learn.

## Using the wrapper

`step()` returns the usual Gymnasium tuple:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

The **episode flags are inside the observation dictionary**, alongside the
original environment observation at `observation["observation"]`.
The **task flags are the separate `terminated` and `truncated` return values**.

| Value | Meaning |
| --- | --- |
| `observation["episode_terminated"]` | The underlying episode terminated. |
| `observation["episode_truncated"]` | The underlying episode truncated. |
| `terminated` | The task's termination callback fired. |
| `truncated` | The task reached `max_task_episodes`. |

Keep stepping between episodes. An episode-ending step returns its final
observation. If the task continues, the following step returns a reset frame
with zero reward and ignores its action. Call `reset()` before stepping and
after the task ends.

`max_task_episodes=0` allows unlimited episodes. An optional `terminate_task`
callback can end the task at an episode boundary. See
[`RepeatTaskEnv`](src/repeat_task_gym/wrapper.py) for callback arguments and reset options.

## Resets

Every public `reset()` starts a new task. In every mode, `reset(seed=42)`
reseeds one wrapper RNG and draws a seed to pass to the environment.
`reset()` and `reset(seed=None)` leave the RNG running. The mode controls when
to draw again; otherwise, each episode reuses the last drawn seed.

```python
# Draw a seed for each episode (default).
env = RepeatTaskEnv(base_env, episode_seed_mode="per_episode")
env.reset(seed=42)
# Keep stepping through the task, then:
env.reset()  # Draw the next seed from the same RNG for this task's first episode.

# Reuse one seed within each task; draw a fresh seed for the next task.
env = RepeatTaskEnv(base_env, episode_seed_mode="per_task")
env.reset(seed=42)
# Keep stepping through the task, then:
env.reset()  # Draw the next task's seed.

# Reuse one seed across episodes and tasks until explicitly reseeded.
env = RepeatTaskEnv(base_env, episode_seed_mode="constant")
env.reset(seed=42)
env.reset()         # Start a new task with the same seed.
env.reset(seed=7)   # Restart the generator and choose a new constant seed.
```

In `"per_episode"` mode, extra episodes consume extra draws, affecting the next
task's seed too.

`episode_reset_options` supplies default options for the base environment.
Public `reset(options=...)` merges over these defaults for the task's first
episode only; later episodes use the defaults. Supply seeds through
`reset(seed=...)`; a `seed` entry in `episode_reset_options` is rejected.

Reset behavior and supported options depend on the base environment. Carefully
verify which conditions stay fixed or change in your configuration.

## Examples

Run `jupyter lab examples/` after installing the examples extra.

- [00 — Episodes to tasks](examples/00_episodes_to_tasks.ipynb): learning within one task.
- [01 — Random rollout](examples/01_random_rollout.ipynb): inputs, outputs, and boundaries.
- [02 — Task termination](examples/02_task_termination.ipynb): success conditions and episode budgets.
- [03 — Seeding](examples/03_rng_seeding_control.ipynb): seed modes and reset options. Requires Python 3.12+.
- [04 — Composition](examples/04_gymnasium_composition.ipynb): statistics, flattening, and vectorization.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and [CHANGELOG.md](CHANGELOG.md)
for changes. Licensed under [GPL-3.0-or-later](LICENSE).
