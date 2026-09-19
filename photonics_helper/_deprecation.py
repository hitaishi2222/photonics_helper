"""Deprecation helpers for the public API.

These are the runtime half of the stability contract (see
``docs/stability.md``). The policy is:

- a deprecated symbol keeps working and warns **once per process**, naming the
  replacement and the release in which it will be removed;
- it stays for at least one minor release before removal.

Nothing in the library is deprecated yet — these helpers exist so the first
rename or retirement follows the policy instead of breaking callers.

Usage
-----

Decorator (functions and classes)::

    from photonics_helper._deprecation import deprecated

    @deprecated("Use new_thing() instead.", replacement="new_thing",
                since="0.1.7", removed_in="0.3.0")
    def old_thing(*args, **kwargs):
        ...

Hand-written module-level shim::

    from photonics_helper._deprecation import warn_deprecated

    def legacy_name(*args, **kwargs):
        warn_deprecated("legacy_name", replacement="new_name",
                        since="0.1.7", removed_in="0.3.0")
        return new_name(*args, **kwargs)
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

__all__ = ["deprecated", "warn_deprecated"]

T = TypeVar("T", bound=Callable[..., object])

# Names already warned about in this process (once-per-symbol semantics).
_warned: set[str] = set()


def _reset_warnings() -> None:
    """Clear the once-per-process registry (for tests only)."""
    _warned.clear()


def _warn_once(
    name: str,
    *,
    replacement: str | None,
    since: str | None,
    removed_in: str | None,
    stacklevel: int,
) -> None:
    if name in _warned:
        return
    _warned.add(name)

    message = f"{name} is deprecated"
    if since:
        message += f" since {since}"
    if replacement:
        message += f"; use {replacement} instead"
    if removed_in:
        message += f". It will be removed in {removed_in}"
    message += "."

    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def warn_deprecated(
    name: str,
    *,
    replacement: str | None = None,
    since: str | None = None,
    removed_in: str | None = None,
) -> None:
    """Emit a once-per-process :class:`DeprecationWarning` for ``name``.

    Parameters
    ----------
    name : the deprecated symbol's public name.
    replacement : what callers should use instead.
    since : version in which it became deprecated.
    removed_in : version in which it will be removed.
    """
    _warn_once(
        name,
        replacement=replacement,
        since=since,
        removed_in=removed_in,
        stacklevel=3,
    )


def deprecated(
    reason: str | None = None,
    *,
    replacement: str | None = None,
    since: str | None = None,
    removed_in: str | None = None,
) -> Callable[[T], T]:
    """Mark a function or class as deprecated.

    The decorated callable keeps its signature and behaviour; the first use in a
    process emits a :class:`DeprecationWarning`. Deprecating a class warns when
    it is instantiated and returns the class unchanged, so ``isinstance`` and
    ``issubclass`` keep working.

    Parameters
    ----------
    reason : optional extra explanation appended to the warning.
    replacement : name of the replacement symbol.
    since : version in which it became deprecated.
    removed_in : version in which it will be removed.
    """

    def decorator(obj: T) -> T:
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))

        def emit(stacklevel: int) -> None:
            suffix = f" ({reason})" if reason else ""
            _warn_once(
                name + suffix,
                replacement=replacement,
                since=since,
                removed_in=removed_in,
                stacklevel=stacklevel,
            )

        if isinstance(obj, type):
            original_init = getattr(obj, "__init__")

            @functools.wraps(original_init)
            def __init__(self: object, *args: object, **kwargs: object) -> None:
                emit(stacklevel=3)
                original_init(self, *args, **kwargs)

            # setattr avoids mypy's unsound-instance-__init__ check while
            # preserving the class object (so isinstance/issubclass still work).
            setattr(obj, "__init__", __init__)
            return obj

        @functools.wraps(obj)
        def wrapper(*args: object, **kwargs: object) -> object:
            emit(stacklevel=3)
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
