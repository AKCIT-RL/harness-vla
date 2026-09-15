"""Environment adapters that expose the Harness VLA agent to external benchmarks.

Each sub-package binds the environment-agnostic core (planner loop, Task
Memory, Global Memory, phase policy) to one benchmark's contract. The
environment itself is never implemented here.
"""
