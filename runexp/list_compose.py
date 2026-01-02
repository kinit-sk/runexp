from typing import Any, List, Mapping, Sequence
from omegaconf import OmegaConf

# list concatenation resolver
OmegaConf.register_new_resolver("concat", lambda *xs: sum((list(x) for x in xs), []))

# compose_list function
def compose_list(
    base: Sequence[Any],
    edits: Sequence[Mapping[str, Any]],
    on_missing: str = "error",  # "error" | "skip"
) -> List[Any]:
    out = list(base)

    for i, edit in enumerate(edits):
        spec = dict(edit)  # defensive copy

        payload = _extract_payload(spec, i)
        mode, ref = _extract_mode(spec, i)

        if mode == "at":
            idx = _normalize_index(ref, len(out), i)
            out[idx:idx] = payload
            continue

        # before / after (anchor-based)
        try:
            idx = out.index(ref)
        except ValueError:
            if on_missing == "skip":
                continue
            raise ValueError(f"Edit #{i}: anchor {ref!r} not found for '{mode}'")

        insert_at = idx if mode == "before" else idx + 1
        out[insert_at:insert_at] = payload

    return out

def _extract_payload(spec: dict, i: int) -> List[Any]:
    has_item = "item" in spec
    has_items = "items" in spec
    if has_item == has_items:
        raise ValueError(f"Edit #{i}: specify exactly one of 'item' or 'items'")
    if has_item:
        return [spec.pop("item")]
    items = spec.pop("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError(f"Edit #{i}: 'items' must be a list")
    return list(items)

def _extract_mode(spec: dict, i: int):
    modes = [k for k in ("before", "after", "at") if k in spec]
    if len(modes) != 1:
        raise ValueError(f"Edit #{i}: specify exactly one of before/after/at")
    mode = modes[0]
    ref = spec.pop(mode)
    if spec:
        raise ValueError(f"Edit #{i}: unknown keys: {sorted(spec.keys())}")
    return mode, ref

def _normalize_index(idx: Any, n: int, i: int) -> int:
    if not isinstance(idx, int):
        raise TypeError(f"Edit #{i}: 'at' must be an integer")

    # Chosen semantics: -1 means end (append), -2 means just before end, etc.
    if idx < 0:
        idx = n + idx + 1

    if idx < 0:
        idx = 0
    if idx > n:
        idx = n
    return idx
