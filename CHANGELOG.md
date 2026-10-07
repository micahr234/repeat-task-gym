# Changelog

## Unreleased

- Copy Procedural FrozenLake info in the seeding example so nested map metadata
  is independent across resets and steps, keeping Gymnasium's checker enabled.
- Control episode reset seeds with `episode_seed_mode`: `"per_episode"`
  (default) draws from a single wrapper RNG at every episode start;
  `"per_task"` reuses a seed within each task; `"constant"` reuses one across
  episodes and tasks. Public `reset(seed=...)` restarts the generator in every
  mode. Seeds must be supplied there, not in `episode_reset_options`.
  Episode lengths and environment RNG use do not affect later episode seeds.
  In `"per_episode"` mode, extra episodes advance the shared seed stream and
  affect the next task's seed.
- Remove `task_reset_options`. Public `reset(options=...)` adds or overrides
  `episode_reset_options` for the first episode of that task only; subsequent
  episodes use the configured episode defaults.
- Replace the `episode_done` observation code with boolean
  `episode_terminated` and `episode_truncated` fields in `Discrete(2)` spaces.
- Replace the task callback's `done` argument with `episode_terminated` and
  `episode_truncated`; preserve both flags when both are true.
- Report task termination and episode-budget truncation independently, allowing
  both task flags to be true on the same step.

## 0.1.0 - 2026-10-07

- Introduce `RepeatTaskEnv`, accepting and returning a Gymnasium environment.
- Represent task termination/truncation through Gymnasium's standard step flags.
- Include underlying episode boundaries in the `episode_done` observation code.
- Preserve separate episode reset frames, task seed draws, episode budgets,
  and task termination predicate.
- Add examples, Gymnasium integration tests, and development CI.
- Provide Jupyter and the Python kernel through the `examples` extra; the setup
  script installs all development and example dependencies by default.
- Use Procedural FrozenLake in the seeding example. The `examples` extra
  installs that package on Python 3.12+.
