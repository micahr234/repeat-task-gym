"""Repeat Gymnasium episodes within a task."""

from importlib.metadata import version

from repeat_task_gym.wrapper import RepeatTaskEnv

__version__ = version("repeat-task-gym")

__all__ = ["RepeatTaskEnv", "__version__"]
