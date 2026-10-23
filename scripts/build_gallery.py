#!/usr/bin/env python3
"""Generate the example gallery under docs/ from examples/*.py.

Writes one page per script plus the gallery index, and downscaled thumbnails of
each script's figures. Every script under examples/ appears exactly once.

    python scripts/build_gallery.py          # write the pages and thumbnails
    python scripts/build_gallery.py --check   # fail if the committed pages are stale

The index and per-script pages are committed so that the navigation and the link
checks work in a fresh clone. Thumbnails are generated and not committed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
IMAGES = EXAMPLES / "images"
DOCS = ROOT / "docs"
OUT = DOCS / "examples"
THUMBS = OUT / "img"

THUMB_WIDTH = 640

# Ordered: the first rule whose pattern matches the script stem wins. A script
# that matches nothing lands in "Other examples", so a new example still appears
# in the gallery without anyone editing this file.
TOPICS: list[tuple[str, str]] = [
    ("Pulses, units and dispersion", r"^(0[1-3]|16|17|33)_|^pulse_visualization"),
    ("Raman response and material data", r"^(0[5-9]|11)_"),
    ("GNLSE propagation", r"^(12|13|14|15|20|21|23|26|27|34|39|40|45)_"),
    ("Solitons and breathers", r"^(18|30)_"),
    ("Few-mode, polarization and structured light", r"^(28|29|31|37|38)_|^generate_"),
    ("Devices: FROG, TMM, phase matching, inverse design", r"^(04|22|24|25|42)_"),
    ("Tapers, validation, noise and cascades", r"^(19|35|36|41|43|44)_"),
    ("Interactive dashboards", r"^(10|32)_"),
]
FALLBACK = "Other examples"

FRONT = "---\nquadrant: reference\n---\n"
PNG = re.compile(r"""["']([A-Za-z0-9_./ -]+\.png)["']""")


def title_of(stem: str, src: str) -> str:
    """First docstring line, with the boilerplate prefix removed."""
    if '"""' not in src:
        return stem
    first = src.split('"""')[1].strip().splitlines()
    text = first[0].strip() if first else stem
    text = re.sub(r"^Example:\s*", "", text).rstrip(".")
    text = text.replace(" — ", ": ").replace("—", ",")
    if len(text) > 96:
        text = text[:96].rsplit(" ", 1)[0] + "..."
    return text


def figures_of(stem: str, src: str, available: set[str]) -> list[str]:
    """Figure files a script writes, from its string literals or its number prefix."""
    named = {Path(n).name for n in PNG.findall(src)}
    found = sorted(n for n in named if n in available)
    if found:
        return found
    prefix = stem.split("_")[0]
    if prefix.isdigit():
        return sorted(n for n in available if n.startswith(prefix + "_"))
    return []


def collect() -> list[dict]:
    available = {p.name for p in IMAGES.glob("*.png")}
    entries = []
    for path in sorted(EXAMPLES.glob("*.py")):
        src = path.read_text(encoding="utf-8", errors="ignore")
        entries.append(
            {
                "stem": path.stem,
                "title": title_of(path.stem, src),
                "figures": figures_of(path.stem, src, available),
                "source": src,
                "lines": len(src.splitlines()),
            }
        )
    return entries


def topic_of(stem: str) -> str:
    for name, pattern in TOPICS:
        if re.search(pattern, stem):
            return name
    return FALLBACK


def group(entries: list[dict]) -> dict[str, list[dict]]:
    order = [name for name, _ in TOPICS] + [FALLBACK]
    buckets: dict[str, list[dict]] = {name: [] for name in order}
    for e in entries:
        buckets[topic_of(e["stem"])].append(e)
    return {k: v for k, v in buckets.items() if v}


def write_thumbs(entries: list[dict]) -> None:
    from PIL import Image

    THUMBS.mkdir(parents=True, exist_ok=True)
    for e in entries:
        for fig in e["figures"]:
            dst = THUMBS / fig
            if dst.exists() and dst.stat().st_mtime >= (IMAGES / fig).stat().st_mtime:
                continue
            with Image.open(IMAGES / fig) as im:
                im = im.convert("RGB")
                if im.width > THUMB_WIDTH:
                    im = im.resize(
                        (THUMB_WIDTH, round(im.height * THUMB_WIDTH / im.width)),
                        Image.Resampling.LANCZOS,
                    )
                im.save(dst, "PNG", optimize=True)


def render_entry(e: dict) -> str:
    """The per-script page: figures, then the source."""
    out = [FRONT, f"# {e['title']}\n"]
    out.append(f"`examples/{e['stem']}.py` ({e['lines']} lines)\n")
    if e["figures"]:
        out.append("## What it produces\n")
        for fig in e["figures"]:
            out.append(f"![{fig}](img/{fig})\n")
    else:
        out.append(
            "This script produces no figure. It prints its result, launches a "
            "server, or writes data for another example.\n"
        )
    out.append("## Source\n")
    out.append(f"```python\n{e['source'].rstrip()}\n```\n")
    return "\n".join(out)


def render_index(buckets: dict[str, list[dict]], total: int) -> str:
    out = [FRONT, "# Examples\n"]
    out.append(
        f"Every runnable script in `examples/`, {total} in total, grouped by what "
        "they demonstrate. Each entry shows the figure it produces and its source.\n"
    )
    out.append(
        "Run one with `python examples/<name>.py`. They write into "
        "`examples/images/` and share no state.\n"
    )
    for topic, entries in buckets.items():
        out.append(f"## {topic}\n")
        for e in entries:
            figs = f" {len(e['figures'])} figure(s)." if e["figures"] else ""
            out.append(f"- [{e['title']}](examples/{e['stem']}.md).{figs}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    check = "--check" in sys.argv
    entries = collect()
    buckets = group(entries)

    written: dict[Path, str] = {}
    written[DOCS / "gallery.md"] = render_index(buckets, len(entries))
    for e in entries:
        written[OUT / f"{e['stem']}.md"] = render_entry(e)

    stale = []
    for path, text in written.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    if check:
        if stale:
            print("Gallery is stale. Run: python scripts/build_gallery.py")
            for p in stale:
                print(f"  {p}")
            return 1
        print(f"OK: gallery matches examples/ ({len(entries)} scripts)")
        return 0

    write_thumbs(entries)
    other = buckets.get(FALLBACK, [])
    print(f"OK: {len(entries)} scripts, {len(buckets)} topics")
    for name, items in buckets.items():
        print(f"  {name}: {len(items)}")
    if other:
        print(f"note: {len(other)} script(s) have no topic rule; add one in TOPICS:")
        for e in other:
            print(f"  {e['stem']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
