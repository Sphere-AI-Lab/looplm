from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class AttentionBackendReport:
    requested: Optional[str]
    attempted: List[str]
    selected: str
    actual: Optional[str]
    fallback_from: Optional[str] = None
    fallback_reason: str = ""
    applied_via: str = ""


def normalize_attn_implementation(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "model.attn_implementation must be a string if provided, "
            f"got {type(value).__name__}."
        )
    value = value.strip()
    return value or None


def resolve_model_torch_dtype(value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    name = value.strip().lower()
    if not name:
        return None

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Resolving model.torch_dtype requires torch to be installed.") from exc

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise ValueError(
            "Unsupported model.torch_dtype value "
            f"{value!r}. Expected one of: {', '.join(sorted(mapping))}."
        )
    return mapping[name]


def build_dtype_factory(
    build_fn: Callable[..., Any],
    model_dtype: Any,
) -> Callable[..., Any]:
    def wrapped(cfg: Any, attn_implementation: Optional[str] = None):
        kwargs = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if model_dtype is None:
            return build_fn(cfg, **kwargs)

        try:
            return build_fn(cfg, dtype=model_dtype, **kwargs)
        except TypeError as exc:
            if not _is_dtype_kwarg_type_error(exc):
                raise
        return build_fn(cfg, torch_dtype=model_dtype, **kwargs)

    return wrapped


def _backend_label(value: Optional[str]) -> str:
    return value if value is not None else "default"


def _clean_reason(text: str) -> str:
    return " ".join(str(text).strip().split())


def apply_model_dtype_to_config(cfg: Any, value: Any) -> None:
    if value is None:
        return
    resolved = resolve_model_torch_dtype(value)
    try:
        setattr(cfg, "dtype", resolved)
    except Exception:
        pass


def _set_config_attn_implementation(cfg: Any, attn_implementation: Optional[str]) -> None:
    if attn_implementation is None:
        return
    try:
        setattr(cfg, "attn_implementation", attn_implementation)
    except Exception:
        pass
    try:
        setattr(cfg, "_attn_implementation", attn_implementation)
    except Exception:
        pass


def get_model_attn_implementation(model: Any) -> Optional[str]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    actual = getattr(cfg, "_attn_implementation", None)
    if actual is None:
        actual = getattr(cfg, "attn_implementation", None)
    if actual is None:
        return None
    if isinstance(actual, str):
        actual = actual.strip() or None
    return actual


def _is_attn_kwarg_type_error(exc: TypeError) -> bool:
    msg = str(exc).lower()
    return "attn_implementation" in msg and "unexpected keyword" in msg


def _is_dtype_kwarg_type_error(exc: TypeError) -> bool:
    msg = str(exc).lower()
    return (
        ("dtype" in msg and "unexpected keyword" in msg)
        or ("torch_dtype" in msg and "unexpected keyword" in msg)
    )


def _is_attention_backend_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    attention_terms = (
        "attn_implementation",
        "flash_attention",
        "flash attention",
        "flash-attention",
        "flash_attn",
        "sdpa",
        "scaled dot product attention",
        "scaled_dot_product_attention",
        "attention backend",
        "does not support",
    )
    capability_terms = (
        "not available",
        "not installed",
        "not supported",
        "unsupported",
        "requires",
        "unexpected keyword",
        "only supports",
        "cannot be used",
        "failed to import",
    )
    return any(term in msg for term in attention_terms) and any(term in msg for term in capability_terms)


def _build_candidate_chain(requested: Optional[str]) -> List[Optional[str]]:
    if requested is None:
        return [None]
    if requested == "flash_attention_2":
        return ["flash_attention_2", "sdpa", None]
    if requested == "flash_attention_3":
        return ["flash_attention_3", "flash_attention_2", "sdpa", None]
    if requested == "sdpa":
        return ["sdpa", None]
    return [requested, None]


def _invoke_builder_with_candidate(
    build_fn: Callable[..., Any],
    cfg: Any,
    attn_implementation: Optional[str],
) -> Tuple[Any, str]:
    if attn_implementation is None:
        return build_fn(cfg), "default"

    _set_config_attn_implementation(cfg, attn_implementation)
    try:
        return build_fn(cfg, attn_implementation=attn_implementation), "from_config_kwarg"
    except TypeError as exc:
        if not _is_attn_kwarg_type_error(exc):
            raise
    return build_fn(cfg), "config_attr"


def _maybe_apply_runtime_attention_backend(
    model: Any,
    requested: Optional[str],
    applied_via: str,
) -> Tuple[Any, str]:
    if requested is None or not hasattr(model, "set_attn_implementation"):
        return model, applied_via

    actual = get_model_attn_implementation(model)
    if actual == requested:
        return model, applied_via

    model.set_attn_implementation(requested)
    applied_via = f"{applied_via}+set_attn_implementation"
    return model, applied_via


def attach_attention_backend_report(model: Any, report: AttentionBackendReport) -> None:
    setattr(model, "_attention_backend_report", asdict(report))


def get_attention_backend_report(model: Any) -> Dict[str, Any]:
    report = getattr(model, "_attention_backend_report", None)
    if report is not None:
        return dict(report)
    actual = get_model_attn_implementation(model)
    return {
        "requested": None,
        "attempted": ["default"],
        "selected": _backend_label(actual) if actual is not None else "default",
        "actual": actual,
        "fallback_from": None,
        "fallback_reason": "",
        "applied_via": "",
    }


def format_attention_backend_report(report: Dict[str, Any]) -> str:
    parts = [
        f"requested={report.get('requested') or 'default'}",
        f"selected={report.get('selected') or 'default'}",
        f"actual={report.get('actual') or 'unknown'}",
    ]

    fallback_from = report.get("fallback_from")
    if fallback_from:
        parts.append(f"fallback_from={fallback_from}")

    fallback_reason = str(report.get("fallback_reason") or "").strip()
    if fallback_reason:
        parts.append(f"reason={_clean_reason(fallback_reason)}")

    applied_via = str(report.get("applied_via") or "").strip()
    if applied_via:
        parts.append(f"via={applied_via}")

    return ", ".join(parts)


def build_model_with_attention_fallback(
    build_fn: Callable[..., Any],
    cfg: Any,
    requested_attn_implementation: Optional[str],
    default_attn_implementation: Optional[str] = "sdpa",
) -> Any:
    requested = normalize_attn_implementation(requested_attn_implementation)
    default_backend = normalize_attn_implementation(default_attn_implementation)
    # Keep flash attention opt-in only. When the config does not explicitly
    # request a backend, prefer a known non-flash backend instead of letting
    # upstream model defaults silently select FlashAttention.
    candidates = _build_candidate_chain(requested if requested is not None else default_backend)

    attempted: List[str] = []
    fallback_from: Optional[str] = None
    fallback_reason = ""

    for idx, candidate in enumerate(candidates):
        attempted.append(_backend_label(candidate))
        cfg_try = copy.deepcopy(cfg)

        try:
            model, applied_via = _invoke_builder_with_candidate(build_fn, cfg_try, candidate)
            model, applied_via = _maybe_apply_runtime_attention_backend(model, candidate, applied_via)
            actual = get_model_attn_implementation(model)

            # If the backend request was silently ignored, prefer falling back to
            # an explicitly known-safe backend when possible.
            if candidate is not None and actual is not None and actual != candidate and idx < (len(candidates) - 1):
                if fallback_from is None:
                    fallback_from = candidate
                fallback_reason = (
                    f"requested backend '{candidate}' but model reported '{actual}'"
                )
                continue

            report = AttentionBackendReport(
                requested=requested,
                attempted=list(attempted),
                selected=_backend_label(candidate),
                actual=actual,
                fallback_from=fallback_from,
                fallback_reason=fallback_reason,
                applied_via=applied_via,
            )
            attach_attention_backend_report(model, report)
            return model
        except Exception as exc:
            if candidate is None or not _is_attention_backend_error(exc):
                raise
            if fallback_from is None:
                fallback_from = candidate
            fallback_reason = _clean_reason(str(exc))

    raise RuntimeError("Unreachable attention backend fallback state.")
