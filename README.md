# Repeat Task Gym

`repeat-task-gym` wraps a Gymnasium environment so an agent can experience
multiple episodes of the same task before the task ends. It takes a
`gymnasium.Env` and returns a `gymnasium.Env` with the usual `reset()` and
five-value `step()` API.

The episode repetition and task seeding follow
[mouse-gym](https://github.com/micahr234/mouse-gym). This package focuses on that
wrapper: use Gymnasium's existing wrappers for reward transforms, statistics,
and vector environments.

## Install

Requires Python 3.11+ and Gymnasium 1.0+.

```bash
pip install -e .
```

To include Jupyter and the Python kernel for running the example notebooks:

```bash
pip install -e ".[examples]"
```

For development, with notebook support:

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The script installs the package, development tools, and example dependencies.
It uses Python 3.14 by default; set `REPEAT_TASK_PYTHON=3.11` to choose another
supported interpreter. Free-threaded Python is not required.

## Quick start

```python
import gymnasium as gym
from repeat_task_gym import RepeatTaskEnv

env = RepeatTaskEnv(gym.make("CartPole-v1"), max_task_episodes=5)
observation, info = env.reset(seed=0)
env.action_space.seed(0)

try:
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        state = observation["observation"]
        episode_done = observation["episode_done"]
        if terminated or truncated:
            observation, info = env.reset()
finally:
    env.close()
```

## Episode and task boundaries

One outer Gymnasium episode is a **task** containing one or more inner episodes.
The standard `terminated` and `truncated` outputs describe the task:

| Output | Meaning |
| --- | --- |
| `terminated=True` | The `terminate_task` callback ended the task. |
| `truncated=True` | `max_task_episodes` episodes completed. |
| Both `False` | Continue stepping within the current task. |

Each observation is a Gymnasium `Dict` with two fields:

| Field | Space | Meaning |
| --- | --- | --- |
| `observation` | Original observation space | The original observation, including nested dictionaries or tuples. |
| `episode_done` | `Discrete(3)` | `0`: running/reset, `1`: episode terminated, `2`: episode truncated. |

If the underlying env sets both flags, `episode_done=1` takes precedence, as in
mouse-gym. This encoding intentionally does not distinguish termination alone
from simultaneous termination and truncation. Rewards, actions, and `info`
keep their original meaning; the action space is unchanged.

An episode-end step returns the **final observation and reward** of that
episode. If the task continues, the next `step(action)` resets the underlying
environment and returns its initial observation, `episode_done=0`, reward
`0.0`, and both task flags `False`. That reset-frame action is ignored. This
preserves mouse-gym's separate terminal and reset frames.

For two one-step episodes and `max_task_episodes=2`:

| Call | Observation | `episode_done` | `terminated` | `truncated` |
| --- | --- | --- | --- | --- |
| `reset()` | First episode's initial state | 0 | - | - |
| `step(action)` | First episode's final state | 1 | False | False |
| `step(action)` | Second episode's initial state | 0 | False | False |
| `step(action)` | Second episode's final state | 1 | False | True |
| `reset()` | Next task's initial state | 0 | - | - |

Call `reset()` before the first step and after either task flag is true.
Stepping without doing so raises `gymnasium.error.ResetNeeded`. A public
`reset()` always starts a new task, including when called partway through one.

## Task settings

```python
env = RepeatTaskEnv(
    existing_env,
    max_task_episodes=5,
    terminate_task=None,
    episode_reset_options=None,
    task_reset_options=None,
)
```

`max_task_episodes=0` (the default) allows unlimited episodes. A task can still
end through `terminate_task`. The predicate runs only on episode-end steps:

```python
def solved(*, reward, done, **kwargs):
    return done == 1 and reward > 0

env = RepeatTaskEnv(
    gym.make("FrozenLake-v1", is_slippery=False),
    max_task_episodes=10,
    terminate_task=solved,
)
```

The callback receives `step_index`, `episode_index`, `state`, `action`,
`reward`, `done`, and `next_state`, matching mouse-gym's task callback.
`episode_index` starts at zero, `step_index` at one on the first actual step;
`state` and `next_state` are original observations, and `done` is the episode
code. `reward` is the current step's reward, not the episode return. If the
predicate fires on the last allowed episode, task termination takes precedence
and `truncated=False`.

## Seeding and resets

`reset(seed=123)` initializes a seed stream and draws a seed for the first task.
Each subsequent public `reset()` draws the next task seed. Repeating
`reset(seed=123)` reproduces the task sequence for the same actions and a
deterministic underlying environment. Action sampling is independently seeded
with `env.action_space.seed(...)`.

Internal episode resets pass `seed=None`: the underlying RNG continues
within the task. Environments that generate a task instance only when given an
explicit seed retain it across those resets. The wrapper cannot impose this
behavior on environments that regenerate their task on every reset.

Every underlying reset receives a fresh dictionary of `episode_reset_options`.
At task starts, `task_reset_options` are overlaid, then options supplied to
`reset(options=...)`. Task-start options are not reused for internal episode
resets. These copies are shallow; option values should be treated as immutable.

## Gymnasium composition

Set up the base environment and episode wrappers before passing it in.
`TimeLimit` inside `RepeatTaskEnv` limits each inner episode; outside it, it
limits the task, including reset frames. Similarly, `RecordEpisodeStatistics`
inside records inner episodes and outside records tasks. When using both,
set a different `stats_key` on the outer wrapper (for example, `"task"`) to
avoid colliding with the inner `info["episode"]`. Apply any automatic task-reset
wrapper outside `RepeatTaskEnv`.

The wrapper delegates rendering, closing, metadata, and the action space to
the wrapped environment. Standard `SyncVectorEnv` or `AsyncVectorEnv` can
batch independently constructed wrappers. `FlattenObservation` can flatten
the augmented observation when supported by the original observation space.
See the [Gymnasium wrapper documentation](https://gymnasium.farama.org/api/wrappers/)
for composition conventions.

## Examples

The numbered notebooks in [`examples/`](examples/) cover:

- **01 - Random rollout:** inspect episode boundaries and standard task resets.
- **02 - Task termination:** end a task on success or its episode budget.
- **03 - RNG seeding control:** reproduce task sequences with independent action seeds.
- **04 - Gymnasium composition:** collect task statistics, flatten observations, and vectorize.

Install the `examples` extra, then run `jupyter lab examples/` from the
environment where the package is installed.

## Development

The repository follows mouse-gym's `src/`, `tests/`, `examples/`, and
`scripts/` layout. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and
[CHANGELOG.md](CHANGELOG.md) for changes.

## License

GNU General Public License v3.0 or later, matching mouse-gym. See [LICENSE](LICENSE).
