"""PARTNR primitive vocabulary, bracket-call codec and instance symbolization.

Classification: **paper-compatible**. The paper's Task Memory stores symbolic,
instance-free command skeletons; PARTNR names entities as ``<category>_<id>``
(``cup_0``, ``table_13``, ``kitchen_1``), so symbolization strips the numeric
suffix and tags the entity kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PARTNR_PRIMITIVE_NAMES",
    "PARTNR_PRIMITIVE_SIGNATURES",
    "PARTNR_VOCABULARY",
    "Vocabulary",
    "active_primitive_names",
    "active_vocabulary",
    "use_vocabulary",
    "extract_json_object",
    "format_action",
    "parse_action",
    "symbolize_partnr_action",
    "symbolize_entity",
]

# Vocabulary agreed for the primitive harness (no Rearrange/Explore macros).
PARTNR_PRIMITIVE_SIGNATURES: Dict[str, List[str]] = {
    "Navigate": ["target"],
    "Pick": ["object"],
    "Place": ["object", "relation", "furniture", "spatial_constraint", "reference"],
    "Open": ["furniture"],
    "Close": ["furniture"],
    "PowerOn": ["object"],
    "PowerOff": ["object"],
    "Clean": ["object"],
    "Fill": ["object"],
    "Pour": ["object"],
    "Wait": [],
    "Done": [],
}
PARTNR_PRIMITIVE_NAMES: Tuple[str, ...] = tuple(PARTNR_PRIMITIVE_SIGNATURES)


@dataclass(frozen=True)
class Vocabulary:
    """Primitive names (plus domain wording) the memory machinery works with.

    PARTNR is the default. A new environment installs its own with
    :func:`use_vocabulary` once per process, so Task/Global Memory keep only
    that environment's primitives and the rule extractor describes the right
    domain; nothing else in the controller is environment specific.
    """

    names: Tuple[str, ...]
    domain: str = "a household robot"
    instance_examples: str = "(cup_0, table_13, kitchen_1)"
    scope: str = "houses"


PARTNR_VOCABULARY = Vocabulary(PARTNR_PRIMITIVE_NAMES)
_ACTIVE_VOCABULARY = PARTNR_VOCABULARY


def use_vocabulary(vocabulary: Optional[Vocabulary]) -> Vocabulary:
    """Install ``vocabulary`` (``None`` restores PARTNR) and return the previous one."""
    global _ACTIVE_VOCABULARY
    previous = _ACTIVE_VOCABULARY
    _ACTIVE_VOCABULARY = vocabulary or PARTNR_VOCABULARY
    return previous


def active_vocabulary() -> Vocabulary:
    return _ACTIVE_VOCABULARY


def active_primitive_names() -> Tuple[str, ...]:
    return _ACTIVE_VOCABULARY.names

# Place keywords that are never entities and must stay literal.
_LITERAL_ARGS = {"on", "within", "next_to", "none", ""}

_BRACKET_CALL = re.compile(r"([A-Za-z_]+)\s*\[([^\]]*)\]", re.DOTALL)
_INSTANCE = re.compile(r"^(?P<category>[A-Za-z][A-Za-z_\-]*?)_(?P<index>\d+)$")
_SYMBOL = re.compile(r"^<(?P<kind>obj|furn|room|ent):(?P<category>[^>]+)>$")
_UNICODE_BRACKETS = str.maketrans({"【": "{", "】": "}", "［": "[", "］": "]"})


# ------------------------------------------------------------- json codec

def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model output."""
    if not text:
        return None
    cleaned = text.strip().translate(_UNICODE_BRACKETS)
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    candidates = [cleaned]
    brace = first_balanced_object(cleaned)
    if brace is not None:
        candidates.append(brace)

    for cand in candidates:
        for attempt in (cand, _strip_trailing_commas(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def first_balanced_object(text: str) -> Optional[str]:
    """Substring of the first ``{...}`` with balanced braces (string-aware)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


# ----------------------------------------------------------- bracket codec


def format_action(name: str, args: Sequence[str] | str | None) -> str:
    """Render ``Name[a, b]``; ``args`` may already be the raw bracket string."""
    if args is None:
        rendered = ""
    elif isinstance(args, str):
        rendered = args.strip()
    else:
        rendered = ", ".join(str(item).strip() for item in args)
    return f"{name}[{rendered}]"


def split_args(raw: str) -> List[str]:
    """Split a raw bracket argument string on commas (empty -> [])."""
    raw = raw.strip()
    if not raw:
        return []
    return [part.strip().strip("'\"") for part in raw.split(",")]


def parse_action(text: str) -> Optional[Tuple[str, List[str]]]:
    """Parse ``Name[a, b]`` or a JSON object carrying an action.

    Accepted JSON shapes: ``{"action": "Name[a, b]"}`` (paper-style, with an
    optional ``thought``) and the legacy ``{"action": "Name", "args": "a, b"}``
    (``args`` may also be a list). Returns ``None`` when nothing parses.
    """
    if not text or not text.strip():
        return None
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        action = parsed.get("action", parsed.get("tool"))
        args = parsed.get("args")
        if isinstance(action, str):
            bracket = _BRACKET_CALL.search(action)
            if bracket is not None:
                embedded = split_args(bracket.group(2))
                if embedded or not args:
                    return bracket.group(1), embedded
                action = bracket.group(1)
            if isinstance(args, list):
                return action.strip(), [
                    "None" if item is None else str(item).strip() for item in args
                ]
            if isinstance(args, str):
                stripped = args.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    stripped = stripped[1:-1]
                return action.strip(), split_args(stripped)
            if args is None:
                return action.strip(), []
        return None
    bracket = _BRACKET_CALL.search(text)
    if bracket is None:
        return None
    return bracket.group(1), split_args(bracket.group(2))


def symbolize_entity(
    token: str,
    *,
    objects: Iterable[str] = (),
    furniture: Iterable[str] = (),
    rooms: Iterable[str] = (),
) -> str:
    """``cup_0`` -> ``<obj:cup>`` using the vocabulary sets; unknown -> ``<ent:cup>``."""
    stripped = token.strip()
    if stripped.lower() in _LITERAL_ARGS or _SYMBOL.match(stripped):
        return stripped
    match = _INSTANCE.match(stripped)
    if match is None:
        return stripped
    category = match.group("category")
    if stripped in set(objects):
        kind = "obj"
    elif stripped in set(furniture):
        kind = "furn"
    elif stripped in set(rooms):
        kind = "room"
    else:
        kind = "ent"
    return f"<{kind}:{category}>"


def symbolize_partnr_action(
    action_str: str,
    *,
    objects: Iterable[str] = (),
    furniture: Iterable[str] = (),
    rooms: Iterable[str] = (),
) -> str:
    """Replace instance ids in ``Name[a, b]`` by ``<kind:category>`` symbols."""
    parsed = parse_action(action_str)
    if parsed is None:
        return action_str.strip()
    name, args = parsed
    objects, furniture, rooms = set(objects), set(furniture), set(rooms)
    symbols = [
        symbolize_entity(arg, objects=objects, furniture=furniture, rooms=rooms)
        for arg in args
    ]
    return format_action(name, symbols)


def parse_symbol(token: str) -> Optional[Tuple[str, str]]:
    """``<obj:cup>`` -> ``("obj", "cup")``; ``None`` for literal tokens."""
    match = _SYMBOL.match(token.strip())
    if match is None:
        return None
    return match.group("kind"), match.group("category")


def instance_category(name: str) -> Optional[str]:
    """``table_13`` -> ``table``; ``None`` when ``name`` is not an instance id."""
    match = _INSTANCE.match(name.strip())
    return match.group("category") if match else None
