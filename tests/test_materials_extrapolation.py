"""Tagged extrapolation for index lookups, and the documented AsSe2 absence.

The default is strict, so ``nk_data``'s "out-of-range evaluation is rejected"
contract and every catalogue audit keep their behaviour. ``extrapolate="warn"``
returns the value with one ``UserWarning`` naming the window (and the material,
through the wrapper); ``"allow"`` is silent and the caller tags the value with
``valid_range_um``.
"""

import warnings

import numpy as np
import pytest

from photonics_helper.base import WavelengthArray
from photonics_helper.core.materials import Material, OpticalMaterial, material
from photonics_helper.materials import RefractiveIndex


def _index() -> RefractiveIndex:
    wl = WavelengthArray(np.linspace(1.0, 5.0, 9), "um")
    n = 1.5 + 0.1 / wl.as_um
    return RefractiveIndex(n=n, k=np.zeros_like(n), wl=wl)


def _material() -> Material:
    return Material(name="TestGlass", index=_index(), source="synthetic")


# ─── 3.2 policies ───────────────────────────────────────────────────────────


def test_strict_default_raises_naming_the_window():
    idx = _index()
    with pytest.raises(
        ValueError, match=r"Index valid only between 1\.0 μm and 5\.0 μm"
    ):
        idx.n_func(9.0)
    # The wrapper adds the material name; the bare index cannot know it.
    with pytest.raises(ValueError, match=r"TestGlass: Index valid only between"):
        _material().n_func(9.0)


def _sweep(record: bool, fn, points):
    with warnings.catch_warnings(record=True) as caught:
        # The suite's ini ignores UserWarning and "always" fires per call; the
        # default registry is what collapses the sweep.
        warnings.simplefilter("default")
        values = [fn(w, extrapolate="warn") for w in points]
    return (caught if record else None), values


def test_warn_returns_the_value_and_warns_once_per_sweep():
    mat = _material()
    caught, values = _sweep(True, mat.n_func, np.linspace(6.0, 9.0, 20))
    assert len(caught) == 1, f"expected one warning, got {len(caught)}"
    text = str(caught[0].message)
    assert "TestGlass" in text and "Index valid only between" in text
    assert caught[0].category is UserWarning
    # The warning points at the caller, not at library internals.
    assert caught[0].filename == __file__, caught[0].filename
    assert all(np.isfinite(values))
    assert values[-1] == pytest.approx(mat.n_func(9.0, extrapolate="allow"))
    # "always" shows every call: that is why the test pins "default".
    with warnings.catch_warnings(record=True) as every:
        warnings.simplefilter("always")
        for w in np.linspace(6.0, 9.0, 20):
            mat.n_func(w, extrapolate="warn")
    assert len(every) == 20


def test_bare_index_warning_is_attributed_to_the_caller():
    idx = _index()
    caught, values = _sweep(True, idx.n_func, np.linspace(6.0, 9.0, 20))
    assert len(caught) == 1
    assert caught[0].filename == __file__, caught[0].filename
    assert "TestGlass" not in str(caught[0].message)  # it cannot know the name
    assert all(np.isfinite(values))


def test_allow_is_silent_and_tagged_by_the_window():
    mat = _material()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = mat.n_func(9.0, extrapolate="allow")
        mat.k_func(9.0, extrapolate="allow")
        mat.nk_func(9.0, extrapolate="allow")
        mat.group_index(9.0, extrapolate="allow")
        assert np.isfinite(mat.group_velocity(9.0, extrapolate="allow"))
    assert caught == []
    assert np.isfinite(value)
    lo, hi = mat.valid_range_um
    assert (lo, hi) == (1.0, 5.0)
    assert mat.index.valid_range_um == (1.0, 5.0)
    # The caller tags the value: 9 µm is outside the window, so the returned
    # number is an extrapolation.
    assert not (lo <= 9.0 <= hi)


def test_inside_the_window_every_policy_agrees():
    mat = _material()
    for wavelength in (1.0, 2.5, 5.0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for policy in ("raise", "warn", "allow"):
                assert mat.n_func(wavelength, extrapolate=policy) == pytest.approx(
                    mat.index.n_func(wavelength)
                )
        assert caught == []


def test_unknown_policy_is_rejected():
    mat = _material()
    for fn in (mat.n_func, _index().n_func):
        with pytest.raises(ValueError, match="extrapolate must be one of"):
            fn(2.0, extrapolate="silent")


def test_k_func_and_nk_func_share_the_policy():
    idx = _index()
    with pytest.raises(ValueError, match="Index valid only between"):
        idx.k_func(0.5)
    with pytest.raises(ValueError, match="Index valid only between"):
        idx.nk_func(0.5)
    with pytest.raises(ValueError, match="Index valid only between"):
        idx.dn_dlambda(0.5)
    with pytest.raises(ValueError, match="Index valid only between"):
        idx.group_index(0.5)
    value = idx.nk_func(0.5, extrapolate="allow")
    assert np.isfinite(value.real) and np.isfinite(value.imag)


def test_material_satisfies_the_protocol_with_the_new_member():
    mat = _material()
    assert isinstance(mat, OpticalMaterial)
    assert isinstance(mat.valid_range_um, tuple)


class _DuckIndex:
    """A minimal ``OpticalMaterial`` index: ``n_func(wavelength)`` and no policy.

    This is exactly what ``docs/building-on-core.md`` and
    ``examples/33_build_on_core.py`` show an external tool writing, so the
    ``Material`` wrapper must not require its ``extrapolate=`` keyword.
    """

    name = "DuckGlass"

    def __init__(self) -> None:
        self._idx = _index()

    @property
    def wl(self):
        return self._idx.wl

    @property
    def n(self):
        return self._idx.n

    @property
    def k(self):
        return self._idx.k

    @property
    def valid_range_um(self):
        return self._idx.valid_range_um

    def n_func(self, wavelength: float) -> float:
        return self._idx.n_func(wavelength, extrapolate="allow")

    def k_func(self, wavelength: float) -> float:
        return self._idx.k_func(wavelength, extrapolate="allow")


def test_wrapper_works_for_a_protocol_only_index():
    wrapped = Material(name="DuckGlass", index=_DuckIndex())
    assert isinstance(wrapped, OpticalMaterial)
    assert wrapped.n_func(2.0) == pytest.approx(_index().n_func(2.0))
    # The wrapper still owns the policy and the naming for such an index.
    with pytest.raises(ValueError, match=r"DuckGlass: Index valid only between"):
        wrapped.n_func(9.0)
    assert np.isfinite(wrapped.n_func(9.0, extrapolate="allow"))


def test_group_velocity_takes_the_policy():
    idx = _index()
    with pytest.raises(ValueError, match="Index valid only between"):
        idx.group_velocity(9.0)
    v = idx.group_velocity(9.0, extrapolate="allow")
    assert np.isfinite(v)
    assert v == pytest.approx(_material().group_velocity(9.0, extrapolate="allow"))


# ─── 3.3 a missing material is a documented absence ─────────────────────────


def test_absence_rule_for_an_unsourced_composition():
    """No ``AsSe2`` row: the lookup fails by name, with near matches."""
    with pytest.raises(ValueError) as excinfo:
        material("AsSe2")
    message = str(excinfo.value)
    assert "AsSe2" in message
    # Near matches are listed, and no fitted index is returned silently.
    assert "Did you mean:" in message
    assert "As2Se3" in message
    # The absence is stated, not just hinted.
    assert "primary index source" in message
    assert "untraceable fit" in message
    # The catalogue the lookup searches really has no row for it.
    from photonics_helper.core.data import MaterialsDatabase

    db = MaterialsDatabase()
    assert db.get_sellmeier("AsSe2") is None
    assert "AsSe2" not in db.list_materials()
    assert "As2Se3" in db.list_materials()
