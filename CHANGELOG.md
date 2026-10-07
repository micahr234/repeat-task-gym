# Changelog

## 0.1.0 - 2026-10-07

- Introduce `RepeatTaskEnv`, accepting and returning a Gymnasium environment.
- Represent task termination/truncation through Gymnasium's standard step flags.
- Include underlying episode boundaries in the `episode_done` observation code.
- Preserve mouse-gym's episode reset frames, task seed stream, episode budget,
  and task termination predicate.
- Add examples, Gymnasium integration tests, and development CI.
- Provide Jupyter and the Python kernel through the `examples` extra; the setup
  script installs all development and example dependencies by default.
