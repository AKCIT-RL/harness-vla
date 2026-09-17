"""Smoke client for the inverted-control env API (``PartnrEnvClient``).

Classification: beta-only (smoke baseline; no LLM). Drives ``reset``/``step``
from the agent side; the env side (``partnr_harness.external``) owns the loop
inversion.
"""

import random

_ROOM_ARG = {"room"}
# Tools whose skills are safe to fire with an arbitrary room/furniture argument.
# State-changing skills (Pour, PowerOn/Off, Fill, Clean, Pick/Place, Open/Close)
# on a random furniture can crash the simulator skill (observed:
# ``Pour[bench_48]`` -> "'NoneType' object has no attribute 'object_id'").
_SAFE_TOOLS = {"navigate", "explore", "wait"}


def RandomLoopAgent(env, max_steps: int = 8, seed: int = 0) -> None:
    """Random navigation/exploration with room/furniture args, then Done.

    ``env`` is a ``partnr_harness.external.PartnrEnvClient`` (duck-typed:
    ``reset()``, ``step()``, ``action_space()``, ``done``).
    """
    rng = random.Random(seed)
    obs = env.reset()
    space = env.action_space()
    candidates = [
        entry
        for entry in space
        if entry["name"].lower() in _SAFE_TOOLS and len(entry.get("arg_names") or []) <= 1
    ]
    for _ in range(max_steps):
        if env.done or not candidates:
            break
        state = obs.get("state") or {}
        rooms = list(state.get("rooms") or [])
        furniture = list((state.get("furniture") or {}).keys())
        entry = rng.choice(candidates)
        arg_names = entry.get("arg_names") or []
        if not arg_names:
            args = ""
        elif arg_names[0] in _ROOM_ARG and rooms:
            args = rng.choice(rooms)
        elif furniture:
            args = rng.choice(furniture)
        elif rooms:
            args = rng.choice(rooms)
        else:
            args = ""
        obs, _, done, _ = env.step({"action": entry["name"], "args": args})
        if done:
            return
    if not env.done:
        env.step("Done")
