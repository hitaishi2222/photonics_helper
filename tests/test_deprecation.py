"""Tests for the deprecation machinery (stability contract, runtime half)."""

from __future__ import annotations

import inspect
import warnings

import pytest

from photonics_helper._deprecation import (
    _reset_warnings,
    deprecated,
    warn_deprecated,
)


@pytest.fixture(autouse=True)
def _fresh_warning_registry():
    _reset_warnings()
    yield
    _reset_warnings()


def _deprecated_function():
    @deprecated(
        "Use new_add instead.",
        replacement="new_add",
        since="0.1.7",
        removed_in="0.3.0",
    )
    def old_add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    return old_add


def test_deprecated_function_still_works_and_warns():
    old_add = _deprecated_function()
    with pytest.warns(DeprecationWarning) as record:
        assert old_add(2, 3) == 5
    assert len(record) == 1


def test_warning_message_names_replacement_and_removal():
    old_add = _deprecated_function()
    with pytest.warns(DeprecationWarning) as record:
        old_add(1)
    message = str(record[0].message)
    assert "old_add" in message
    assert "new_add" in message
    assert "0.3.0" in message
    assert "0.1.7" in message


def test_warning_is_emitted_once_per_process():
    old_add = _deprecated_function()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old_add(1)
        old_add(2)
        old_add(3)
    assert len(record) == 1


def test_reset_warnings_re_enables():
    old_add = _deprecated_function()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old_add(1)
        _reset_warnings()
        old_add(1)
    assert len(record) == 2


def test_signature_is_preserved():
    old_add = _deprecated_function()
    params = inspect.signature(old_add).parameters
    assert list(params) == ["a", "b"]
    assert params["b"].default == 1
    assert params["a"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert old_add.__name__ == "old_add"
    assert old_add.__doc__ == "Add two numbers."


def test_keyword_call_is_preserved():
    old_add = _deprecated_function()
    with pytest.warns(DeprecationWarning):
        assert old_add(a=5, b=6) == 11


def test_warn_deprecated_helper():
    with pytest.warns(DeprecationWarning) as record:
        warn_deprecated("legacy", replacement="modern", since="0.1.7", removed_in="0.3.0")
    message = str(record[0].message)
    assert "legacy" in message and "modern" in message and "0.3.0" in message


def test_deprecated_class_warns_on_instantiation_and_keeps_identity():
    @deprecated(replacement="NewThing", since="0.1.7", removed_in="0.3.0")
    class OldThing:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    with pytest.warns(DeprecationWarning):
        obj = OldThing(7)

    assert obj.value == 7
    assert isinstance(obj, OldThing)  # class identity preserved
    params = inspect.signature(OldThing).parameters
    assert list(params) == ["value"]
    assert params["value"].default == 0
