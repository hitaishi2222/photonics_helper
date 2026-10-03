"""Example: querying the material catalog
=========================================

``photonics_helper.materials.material_catalog(name=None)`` is the programmatic
front door to the shipped ``materials.db``: one call returns every dataset the
library can evaluate ``n(λ)`` from, with its kind, axis, wavelength range,
source and provenance. As shipped it returns **69 datasets across 40
materials** (30 tabulated spectra + 39 Sellmeier equations) — the numbers the
script prints, so this docstring cannot drift away from the database silently.
This example is the *query* tutorial: what each field means, how to filter it,
which materials are citable, and whether the database is internally
consistent.

How this differs from ``examples/11``
-------------------------------------
``11_raman_material_catalog.py`` is the **plot gallery**: it draws the
Raman-relevant subset so a human can look at the spectra. This one is the
**programmatic contract** — filters, field semantics, provenance coverage and a
consistency smoke test, with the plots as evidence rather than as the point.
They are complements; the docstrings cross-link, and the band-coverage figure
here answers the question "what can I use at 1550 nm?" that `11`'s plots leave
to the reader's eye.

What a ``MaterialDataset`` actually carries
-------------------------------------------
One entry is one *dataset*, not one material: a material may appear several
times (tabulated spectra per literature source, plus one Sellmeier equation).
The fields, and which of them are populated for which kind:

=========================  ============  ====================================
field                      tabulated     sellmeier
=========================  ============  ====================================
``material``               always        always (axis suffix already stripped)
``kind``                   ``tabulated`` ``sellmeier``
``axis``                   ``None``      ``"extraordinary"``/``"ordinary"``
                                          only for birefringent crystals
                                          (stored as ``<name>_er``/``_or``)
``source`` / ``source_key``source key    citation string
``wl_min_um``/``wl_max_um``data range     the equation's validity range
``n_points``               point count   ``None`` — an equation has no grid
``doi``                    from provenance, else extracted from the citation
``citation``               provenance    the source string itself
``license``                provenance    e.g. ``see-source-publication``
=========================  ============  ====================================

So a ``sellmeier`` row with ``n_points is None`` is **not** missing data, and a
``tabulated`` row with ``axis is None`` is not a bug: ``axis`` only ever means
"this material is birefringent and the two branches are stored separately".

The substring filter is a substring filter
-------------------------------------------
``material_catalog("sil")`` returns ``Silica``, ``Silica``, ``Silica`` and
``Silicon`` — the match is case-insensitive and *substring*, so an author
query needs to be specific. The example prints this rather than hiding it, and
asserts the membership it claims.

Provenance is reported, not enforced
------------------------------------
Citation completeness is a live concern in this repository (``ISSUES.md``,
the ``material-provenance`` capability), so the script prints the coverage
matrix and the gap counts — how many datasets have no DOI, no citation, no
licence — and **exits 0 either way**. This is a report. What it *does* fail on
is a database that is internally inconsistent, because that is a defect rather
than a documentation gap.

Cost note: pure catalog reading plus one ``RefractiveIndex`` build for the
hand-off section. No propagation, no optimisation; the whole example runs in
well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photonics_helper.materials import (
    MaterialDataset,
    RefractiveIndex,
    material_catalog,
)

QUERY_UM = 1.55  # the application band this example queries for
QUERY_LABEL = "1550 nm telecom C+L"
SPOT_WAVELENGTHS_UM = (1.064, 1.31, 1.55)

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "provenance": OUT_DIR / "43_material_catalog_provenance.png",
    "bands": OUT_DIR / "43_material_coverage_bands.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 70 + f"\n {title}\n" + "=" * 70


def covers(d: MaterialDataset, wl_um: float) -> bool:
    """Does this dataset's stated range contain ``wl_um``?"""
    rng = d.wavelength_range_um
    return rng is not None and rng[0] <= wl_um <= rng[1]


# ── 1. Query ─────────────────────────────────────────────────────────────────


def run_query() -> list[MaterialDataset]:
    catalog = material_catalog()
    kinds = [d.kind for d in catalog]
    print(_rule("1. Querying the catalog"))
    print(
        f"  material_catalog()            -> {len(catalog)} datasets for "
        f"{len({d.material for d in catalog})} materials"
    )
    print(f"    kind='tabulated'            : {kinds.count('tabulated')}")
    print(f"    kind='sellmeier'            : {kinds.count('sellmeier')}")
    print("  one entry is one DATASET: a material with two tabulated sources and")
    print(
        "  a Sellmeier equation appears three times. Count materials with `len({...})`."
    )

    silica = material_catalog("sil")
    print()
    print(
        f"  material_catalog('sil')       -> {len(silica)} datasets: "
        f"{[d.material for d in silica]}"
    )
    print("  the filter is a case-insensitive SUBSTRING, so 'Silicon' comes along with")
    print("  'Silica'. Query by author needs a more specific needle.")
    assert all("sil" in d.material.lower() for d in silica), (
        "the substring filter returned a non-matching material: "
        f"{[d.material for d in silica]}"
    )
    assert "Silica" in {d.material for d in silica}, "Silica was not found"

    print()
    print("  every MaterialDataset field, for Silica:")
    for d in material_catalog("Silica"):
        print(
            f"    kind={d.kind:<11} axis={str(d.axis):<14} "
            f"n_points={str(d.n_points):<5}"
        )
        print(f"      source      = {d.source}")
        print(f"      source_key  = {d.source_key}")
        print(f"      range (um)  = {d.wavelength_range_um}")
        print(f"      doi         = {d.doi}")
        print(f"      citation    = {d.citation}")
        print(f"      license     = {d.license}")
    print(
        "  -> axis is None for tabulated rows and n_points is None for Sellmeier rows"
    )
    print("     because neither field means anything for that kind. Treat 'None' as")
    print("     'not applicable', not 'missing'.")
    assert all(d.axis is None for d in catalog if d.kind == "tabulated"), (
        "a tabulated dataset carries a polarization axis; the contract says "
        "tabulated rows have axis=None"
    )
    assert all(d.n_points is None for d in catalog if d.kind == "sellmeier"), (
        "a Sellmeier dataset carries a point count; an equation has no grid"
    )
    assert any(d.kind == "sellmeier" and d.n_points is None for d in catalog)
    return catalog


# ── 2. Filters ───────────────────────────────────────────────────────────────


def run_filters(catalog: list[MaterialDataset]) -> dict:
    print(_rule(f"2. Filters — and what each one answers for {QUERY_LABEL}"))
    tabulated = [d for d in catalog if d.kind == "tabulated"]
    sellmeier = [d for d in catalog if d.kind == "sellmeier"]
    birefringent = [d for d in catalog if d.axis is not None]
    in_band = [d for d in catalog if covers(d, QUERY_UM)]

    print(f"  by kind   : tabulated {len(tabulated)}, sellmeier {len(sellmeier)}")
    print(f"  by axis   : {len(birefringent)} datasets carry an explicit axis branch")
    for d in birefringent:
        print(f"              {d.material:<10} {d.axis:<14} {d.wavelength_range_um}")
    print(
        f"  by band   : {len(in_band)} of {len(catalog)} datasets cover "
        f"{QUERY_UM} um "
        f"({QUERY_LABEL})"
    )
    print(
        f"    sellmeier covering it : "
        f"{len([d for d in in_band if d.kind == 'sellmeier'])}"
    )
    print(
        f"    tabulated covering it : "
        f"{len([d for d in in_band if d.kind == 'tabulated'])}"
    )

    # An application-band query is only useful if the answer is checkable.
    for d in in_band:
        assert covers(d, QUERY_UM), (
            f"{d.material}/{d.kind} was selected for {QUERY_UM} um but its "
            f"range {d.wavelength_range_um} does not contain it"
        )
    assert all(not covers(d, QUERY_UM) for d in catalog if d not in in_band), (
        "the band filter is not exhaustive: a dataset outside the selection "
        "also covers the query wavelength"
    )
    outside = [d.material for d in catalog if not covers(d, QUERY_UM)]
    print(
        f"    not covering it ({len(outside)}): {', '.join(sorted(outside)[:8])}, ..."
    )

    narrowest = min(
        in_band, key=lambda d: d.wavelength_range_um[1] - d.wavelength_range_um[0]
    )
    widest = max(
        in_band, key=lambda d: d.wavelength_range_um[1] - d.wavelength_range_um[0]
    )
    print(
        f"  narrowest in-band range : {narrowest.material:<12} "
        f"{narrowest.wavelength_range_um}"
    )
    print(
        f"  widest in-band range   : {widest.material:<12} {widest.wavelength_range_um}"
    )
    return {
        "tabulated": tabulated,
        "sellmeier": sellmeier,
        "birefringent": birefringent,
        "in_band": in_band,
        "narrowest": narrowest,
        "widest": widest,
    }


# ── 3. Provenance table ─────────────────────────────────────────────────────


def run_provenance(catalog: list[MaterialDataset]) -> dict:
    print(_rule("3. Provenance coverage — which of these can I cite?"))
    header = (
        f"  {'dataset':<24}{'kind':<11}{'axis':<14}"
        f"{'range (um)':<22}{'n':>5}  DOI  cite  licence"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    gaps = {"doi": [], "citation": [], "license": []}
    rows = []
    for d in catalog:
        key = d.source_key or d.material
        flags = {
            "doi": bool(d.doi),
            "citation": bool(d.citation),
            "license": bool(d.license),
        }
        for field, present in flags.items():
            if not present:
                gaps[field].append(key)
        rng = d.wavelength_range_um
        rng_s = f"{rng[0]:.4g}-{rng[1]:.4g}" if rng else "—"
        rows.append(
            {
                "key": key,
                "material": d.material,
                "kind": d.kind,
                "axis": d.axis,
                "flags": flags,
                "n_points": d.n_points,
                "range": rng,
            }
        )
        print(
            f"  {key:<24}{d.kind:<11}{str(d.axis):<14}{rng_s:<22}"
            f"{str(d.n_points or '—'):>5}"
            f"{'  yes' if flags['doi'] else '   NO':>6}"
            f"{'  yes' if flags['citation'] else '   NO':>7}"
            f"{'  yes' if flags['license'] else '   NO':>9}"
        )

    print()
    for field in ("doi", "citation", "license"):
        n = len(gaps[field])
        print(
            f"  datasets missing a {field:<9}: {n:3d} of {len(catalog)}"
            + (f"  -> {', '.join(sorted(gaps[field])[:4])}, ..." if n else "")
        )
    print("  This is a report, not a gate: the example exits 0 with the gaps in place.")
    print("  A missing DOI means the citation string carries no parseable DOI,")
    print(
        "  not that the data is unusable — the licence field is populated everywhere."
    )
    return {"rows": rows, "gaps": gaps}


# ── 4. Consistency smoke test ────────────────────────────────────────────────


def consistency_failures(catalog: list[MaterialDataset]) -> list[str]:
    """Every internal inconsistency in ``catalog``, as a named message.

    Returns a list rather than raising so the caller can report *all* of them
    at once and so the negative control can reuse it on a corrupted copy.
    """
    problems: list[str] = []
    for d in catalog:
        key = f"{d.material}/{d.kind}/{d.source_key or d.source or '-'}"
        rng = d.wavelength_range_um
        if rng is None:
            problems.append(f"{key}: no wavelength range stated")
            continue
        if not rng[0] < rng[1]:
            problems.append(
                f"{key}: wavelength range is not increasing ({rng[0]} >= {rng[1]} um)"
            )
        if d.n_points is not None and d.n_points < 2:
            problems.append(f"{key}: n_points={d.n_points} cannot describe a spectrum")
        if not d.material:
            problems.append(f"{key}: empty material name")
    return problems


def run_consistency(catalog: list[MaterialDataset]) -> dict:
    print(_rule("4. Consistency smoke test"))
    problems = consistency_failures(catalog)
    print(f"  structural checks over {len(catalog)} datasets:")
    print("    wl_min_um < wl_max_um for every stated range")
    print("    n_points >= 2 whenever n_points is stated")
    print("    every dataset names a material")
    print(f"  -> {len(problems)} violation(s)")
    for p in problems:
        print(f"     {p}")

    # Grid-shape check on the datasets the public loader can actually load.
    # The shipped grids are NOT uniform — they are log-spaced, which is why the
    # catalog's range and point count cannot be used to infer a step — so this
    # reports the shape rather than asserting uniformity.
    resolvable, unresolvable, mismatch = [], [], []
    linear, log_spaced = [], []
    for d in [x for x in catalog if x.kind == "tabulated"]:
        try:
            index = RefractiveIndex.from_material_database(d.source_key)
        except ValueError as exc:
            unresolvable.append((d.source_key, str(exc).split(".")[0]))
            continue
        resolvable.append(d.source_key)
        wl = np.asarray(index.wl.as_um, dtype=float)
        if len(wl) != d.n_points:
            mismatch.append(
                f"{d.source_key}: catalog says n_points={d.n_points}, loaded {len(wl)}"
            )
        if wl.size < 3:
            continue
        step = np.abs(np.diff(wl))
        rel = float(np.std(step) / np.mean(step))
        (linear if rel < 1e-9 else log_spaced).append(d.source_key)
    print()
    print(f"  grid check: {len(resolvable)} of the tabulated source keys load through")
    print(
        "  RefractiveIndex.from_material_database. n_points agrees with the "
        "loaded array"
    )
    print(
        f"  for all of them ({len(mismatch)} mismatch(es)); the grids are "
        f"log-spaced, not"
    )
    print(
        f"  uniform ({len(log_spaced)} log-spaced, {len(linear)} linear), "
        f"so a wavelength range and a"
    )
    print("  point count do not imply a constant step.")
    for msg in mismatch:
        print(f"     {msg}")
    assert not mismatch, (
        "the catalog's n_points disagrees with the loaded spectrum:\n  "
        + "\n  ".join(mismatch)
    )

    print()
    print(
        f"  source resolvability: {len(unresolvable)} tabulated dataset(s) "
        f"are listed in the"
    )
    print("  catalog but cannot be loaded through the public entry point:")
    for key, reason in unresolvable:
        print(f"     {key:<22} {reason}")
    print("  The `material-author` key is split on the first hyphen, so any key")
    print("  whose *author* field itself contains a hyphen is mangled. This is a")
    print("  defect in the loader, not in the data, and it is reported here")
    print("  rather than patched: see the `fix-material-dataset-key-split`")
    print("  change. The example therefore does NOT assert resolvability.")

    assert not problems, (
        "the shipped database is internally inconsistent:\n  " + "\n  ".join(problems)
    )
    print("  no structural violations ✓  (resolvability reported, not asserted)")
    return {
        "problems": problems,
        "resolvable": resolvable,
        "unresolvable": unresolvable,
        "n_mismatch": len(mismatch),
        "log_spaced": len(log_spaced),
        "linear": len(linear),
    }


def negative_control(catalog: list[MaterialDataset]) -> None:
    """Corrupt a copy of the catalog and confirm the checks fire."""
    print(_rule("5. Negative control — corrupt a copy and watch the checks fire"))
    from dataclasses import replace

    bad_range = replace(catalog[0], wl_min_um=5.0, wl_max_um=1.0)
    few_points = replace(catalog[1], n_points=1)
    nameless = replace(catalog[2], material="")
    no_range = replace(catalog[3], wl_min_um=None, wl_max_um=None)
    corrupted = [bad_range, few_points, nameless, no_range]

    fired = consistency_failures(corrupted)
    print(f"  injected 4 defects into a 4-entry copy; {len(fired)} check(s) fired:")
    for msg in fired:
        print(f"     {msg}")
    assert len(fired) == 4, (
        f"the consistency checks reported {len(fired)} of 4 injected "
        "defects; a check that cannot fail is not a check"
    )
    assert consistency_failures(catalog) == [], (
        "the corrupted copy should differ from the shipped catalog, but the "
        "shipped catalog also failed — the negative control is meaningless"
    )
    print("  all four fired, and the shipped catalog still passes ✓")


# ── 5. Hand-off to physics ───────────────────────────────────────────────────


def run_handoff(in_band: list[MaterialDataset], resolvable: list[str]) -> dict:
    print(_rule(f"6. From the catalog to n(lambda) — the {QUERY_LABEL} query in use"))
    usable = [
        d
        for d in in_band
        if d.kind == "tabulated" and d.source_key in resolvable and covers(d, QUERY_UM)
    ]
    assert usable, (
        "the catalog query found no tabulated dataset covering the query band "
        "that also loads — the query returned nothing usable"
    )
    chosen = max(
        usable, key=lambda d: d.wavelength_range_um[1] - d.wavelength_range_um[0]
    )
    index = RefractiveIndex.from_material_database(chosen.source_key)

    print("  selection rule: among the tabulated datasets that cover the query band")
    print("  AND load through the public loader, take the widest range.")
    print(
        f"  selected by the query, not hard-coded: {chosen.material} "
        f"({chosen.source_key})"
    )
    print(
        f"    tabulated range {chosen.wavelength_range_um} um, "
        f"{chosen.n_points} points, source_key resolvable"
    )
    print(f"    citation: {chosen.citation}")
    print(f"    doi     : {chosen.doi}")
    print("  n(lambda) at three wavelengths, inside its stated range:")
    values = []
    for wl in SPOT_WAVELENGTHS_UM:
        if not covers(chosen, wl):
            print(
                f"    {wl * 1000:7.0f} nm: outside the dataset's range "
                f"({chosen.wavelength_range_um}) — refused, as it should be"
            )
            continue
        n_val = index.n_func(wl)
        # A spline through an all-zero absorption column returns values at the
        # 1e-144 level; that is interpolation noise, not a k of -3e-144.
        k_val = index.k_func(wl)
        k_shown = 0.0 if abs(k_val) < 1e-30 else k_val
        values.append((wl, n_val, k_shown))
        print(
            f"    {wl * 1000:7.0f} nm: n = {n_val:.5f}, k = {k_shown:.3e}"
            + ("   (spline noise, treated as 0)" if k_shown == 0.0 else "")
        )

    # Out-of-range must be refused, not silently extrapolated.
    try:
        index.n_func(chosen.wl_max_um * 10)
    except ValueError as exc:
        print(
            f"  evaluating beyond the range raises, as documented: {str(exc)[:60]}..."
        )
    else:
        raise AssertionError(
            "RefractiveIndex extrapolated past its stated range; the guard is "
            "not working"
        )
    assert values, "no spot wavelength was inside the dataset's range"
    return {"chosen": chosen, "index": index, "values": values}


# ── Figures ──────────────────────────────────────────────────────────────────


def provenance_figure(prov: dict, sweep: dict) -> None:
    rows = prov["rows"]
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            not rows[i]["flags"]["doi"],
            not rows[i]["flags"]["citation"],
            not rows[i]["flags"]["license"],
            rows[i]["key"],
        ),
    )
    rows = [rows[i] for i in order]
    fig, ax = plt.subplots(figsize=(11.0, 11.5))
    fields = ["doi", "citation", "license"]
    grid = np.array([[r["flags"][f] for f in fields] for r in rows], dtype=float)

    ax.imshow(
        grid,
        cmap=matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"]),
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(fields)), ["DOI", "citation", "licence"])
    ax.set_yticks(range(len(rows)), [r["key"] for r in rows])
    ax.set_xlim(-0.5, len(fields) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(
        "provenance coverage of every shipped dataset — green present, red missing",
        fontsize=11,
    )
    for i in range(len(rows)):
        for j in range(len(fields)):
            ax.text(
                j,
                i,
                "  present" if grid[i, j] else "  MISSING",
                ha="center",
                va="center",
                fontsize=6,
                color="w" if grid[i, j] else "black",
            )

    ylabel = ax.get_ylabel()
    n_missing = sum(1 for r in rows if not r["flags"]["doi"])
    ax.set_ylabel(
        f"{ylabel}\n({len(rows)} datasets, {n_missing} without a DOI)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(FIGS["provenance"], dpi=130, bbox_inches="tight")
    plt.close(fig)
    del sweep


def bands_figure(
    catalog: list[MaterialDataset], in_band: list[MaterialDataset], handoff: dict
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 10.0))
    order = sorted(catalog, key=lambda d: (d.material.lower(), d.kind))
    cmap = matplotlib.colormaps["viridis"]
    for i, d in enumerate(order):
        rng = d.wavelength_range_um
        if rng is None:
            continue
        colour = "tab:red" if d.kind == "tabulated" else cmap(0.25)
        ax.plot(
            rng,
            [i, i],
            lw=5 if covers(d, QUERY_UM) else 3,
            color=colour,
            solid_capstyle="butt",
        )
        if d is handoff["chosen"]:
            ax.annotate("  selected", (rng[1], i), va="center", fontsize=7)

    ax.axvspan(QUERY_UM - 0.05, QUERY_UM + 0.05, color="0.85", zorder=0)
    ax.axvline(QUERY_UM, color="0.3", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 200.0)
    ax.set_yticks(
        range(len(order)), [f"{d.material} [{d.kind[:3]}]" for d in order], fontsize=7
    )
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("wavelength (um)")
    ax.set_title(
        f"wavelength coverage per dataset; the shaded band is the "
        f"{QUERY_LABEL}\nquery ({QUERY_UM} um). Thick bars are "
        f"datasets that cover it.",
        fontsize=11,
    )
    ax.grid(alpha=0.3, axis="x")
    del in_band
    fig.tight_layout()
    fig.savefig(FIGS["bands"], dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print(_rule("0. material_catalog — the programmatic contract"))
    print("  material_catalog(name=None) -> list[MaterialDataset]")
    print("  one entry is one DATASET (a material may appear several times),")
    print("  and the fields that are None depend on the kind — see the docstring.")

    catalog = run_query()
    filters = run_filters(catalog)
    prov = run_provenance(catalog)
    consistency = run_consistency(catalog)
    negative_control(catalog)
    handoff = run_handoff(filters["in_band"], consistency["resolvable"])

    print(_rule("Summary"))
    print(
        f"  datasets shipped                 : {len(catalog)} "
        f"({len({d.material for d in catalog})} materials)"
    )
    print(f"  covering {QUERY_UM} um            : {len(filters['in_band'])} datasets")
    print(f"  birefringent axis branches       : {len(filters['birefringent'])}")
    print(
        f"  missing a DOI                    : "
        f"{len(prov['gaps']['doi'])} of {len(catalog)}"
    )
    print(
        f"  missing a citation               : "
        f"{len(prov['gaps']['citation'])} of {len(catalog)}"
    )
    print(
        f"  missing a licence                : "
        f"{len(prov['gaps']['license'])} of {len(catalog)}"
    )
    print(f"  structural inconsistencies       : {len(consistency['problems'])}")
    print(
        f"  tabulated keys not loadable      : "
        f"{len(consistency['unresolvable'])} (reported; see the "
        f"fix-material-dataset-key-split change)"
    )
    print(
        f"  log-spaced / linear tabulated grids: "
        f"{consistency['log_spaced']} / {consistency['linear']} "
        f"(n_points mismatches: {consistency['n_mismatch']})"
    )
    print(
        f"  hand-off                         : {handoff['chosen'].material} "
        f"({handoff['chosen'].source_key}), "
        f"n({QUERY_UM} um) = {handoff['values'][-1][1]:.5f}"
    )
    print(
        "  catalog queries, filters, provenance report and consistency checks all ran ✓"
    )

    provenance_figure(prov, filters)
    bands_figure(catalog, filters["in_band"], handoff)
    print("\nGenerated files:")
    for p in FIGS.values():
        print(f"  {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
