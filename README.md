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

Requires Python 3.11+ and Gymnasium 1.4+. Development uses free-threaded
Python 3.15 (`3.15t`).

```bash
pip install -e .
pip install -e ".[examples]"  # Include notebook dependencies.
```

## Quick start

FrozenLake makes the learning problem concrete: an agent must find a route
across a lake without knowing which tiles are safe and which are holes. The
example generates a map with a path to the goal, but does not give that map
to the policy. The agent observes its current cell and the outcomes of its
actions. Movement is deterministic here, so a failed attempt can teach it
which route to avoid on the next attempt.

**One map is a task; each attempt on that map is an episode.** An episode ends
at the goal, in a hole, or after 50 steps. Repeat Task Gym keeps these attempts
inside one task, with an explicit episode boundary where the agent can retain
what it learned. This formalizes the question: **how many attempts does the
agent need on a new, unknown task to meet a success criterion?**

Below, the criterion is reaching the goal once. `terminate_task` ends the task
on success, while `max_task_episodes=10` caps the number of attempts. The caller
counts completed episodes to measure attempts used. If the budget runs out
first, record the task as unsolved within that budget.

```python
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from repeat_task_gym import RepeatTaskEnv


def reached_goal(*, reward, episode_terminated, **kwargs):
    return episode_terminated and reward > 0


env = RepeatTaskEnv(
    gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=4, p=0.8, seed=7),
        is_slippery=False,
        max_episode_steps=50,
    ),
    max_task_episodes=10,
    terminate_task=reached_goal,
)
try:
    observation, _ = env.reset()
    attempts = 0
    episode_return = 0.0
    while True:
        # An episode boundary requires an explicit reset-frame step.
        action = env.action_space.sample() if env.action_required else None
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        if observation["episode_terminated"] or observation["episode_truncated"]:
            attempts += 1
            print(f"Attempt {attempts} return: {episode_return}")
            episode_return = 0.0
        if terminated or truncated:
            if terminated:
                print(f"Reached the goal after {attempts} attempts.")
            else:
                print(f"Goal not reached within {attempts} attempts.")
            break
finally:
    env.close()
```

The random action sampler only illustrates the protocol; it does not learn.
An adapting agent would update its memory from real transitions, keep that
memory across `step(None)` episode resets, and clear it for a new map. The
number of attempts to first success is a simple measure; consistent success
over later episodes is stronger evidence that the agent learned the task.

The generated map stays fixed for all attempts. The map seed makes the layout
reproducible; it does not reveal the map to the policy. To evaluate another
unknown task, generate another map and construct a new FrozenLake environment
with it. Calling public `reset()` on this environment starts another task on
the same map; it does not generate a new layout. Across sampled maps, compare
attempts to success and the fraction of tasks solved under the same budget.

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
observation. If the task continues, call **`step(None)`** next to receive the
reset frame. Passing any other action on that step raises `ValueError` without
advancing the environment. Then resume ordinary actions. This applies to both
episode termination and episode truncation. Call `reset()` before stepping
and after the task ends.

`env.action_required` is false while that reset frame is pending (and before a
task is reset). `None` is not an ordinary environment action. Reset frames
return reward 0 before any reward transform and should not be used as action
transitions when updating a learner.

The wrapper returns the base environment's `info` entries in a new dictionary
without adding metadata or mutating the original dictionary.

`max_task_episodes=0` allows unlimited episodes. That budget is checked only
when an episode ends. `reward_transform` and `terminate_task` run on every
`step()`, including mid-episode steps and the reset frame between episodes.
They are not called from `reset()`. `reward_transform` replaces the step
reward, and `terminate_task` sees that reward. A true result ends the task on
that step. A reset frame starts at reward 0 with `action=None`. See
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
- [01 — Random rollout](examples/01_random_rollout.ipynb): CartPole inputs, outputs, and explicit reset frames.
- [02 — Task termination](examples/02_task_termination.ipynb): success conditions and episode budgets.
- [03 — Seeding](examples/03_rng_seeding_control.ipynb): seed modes and reset options. Requires Python 3.12+.
- [04 — Composition](examples/04_gymnasium_composition.ipynb): statistics, flattening, and vectorization.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and [CHANGELOG.md](CHANGELOG.md)
for changes. Licensed under [GPL-3.0-or-later](LICENSE).
