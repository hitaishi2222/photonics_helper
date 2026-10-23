---
quadrant: reference
---

# Documentation style

These rules exist because the docs read like a design record rather than
instructions. They are short on purpose. Run `python scripts/check_docs.py`
before you push; it enforces the mechanical ones.

## Four kinds of page

Every page declares one quadrant in its front matter and lives in the matching
navigation section. The checker enforces both.

| Quadrant | Answers | Shape |
|---|---|---|
| `tutorial` | "teach me by doing" | one continuous run with a result the reader checks |
| `how-to` | "how do I do X" | prerequisites, numbered steps, "you know it worked when" |
| `reference` | "what exists" | signatures, tables, constraints |
| `explanation` | "why does this work" | prose, derivation, trade-offs |

`home` exists only for `index.md`.

A how-to guide links to an explanation page instead of restating the physics. If
you find yourself deriving something in a how-to, stop and link.

A reference page may contain fragments that do not run. A how-to may not: every
snippet on a how-to or tutorial page is executed by
`scripts/check_docs_code.py`, and where the page records the output, the output
is compared.

## Audience

Write to one person who has a physics problem and a deadline. Assume photonics,
assume no knowledge of this library's internals.

Do not write to a reviewer. Do not justify design decisions, explain what was
deliberately excluded, or argue that a choice was correct. That belongs in a code
comment, `notes/`, or the changelog.

## Every page

- Opens with what the reader can do when they finish reading it. Two sentences.
- Has a first code block that runs as written. No `...` placeholders, no
  commented-out alternatives inside the main example.
- Shows real output for any block that produces numbers. Copy it from an actual
  run, not from memory.
- When it claims a check passes, gives the number and names the test or the file
  that proves it.

## Prose

- One idea per paragraph, three sentences maximum.
- Second person, present tense: "you launch a pulse", not "a pulse is launched".
- No em dashes. Use a comma, a colon, a semicolon, or start a new sentence.
- Never end a sentence or a table cell with an ellipsis. Finish the claim or cut
  it.
- Bold at most three times per page, and only for a name the reader will type.
  Bold is not emphasis.
- Say the number. "the peak rose 0.02 %", not "the peak rose only slightly".

## Tables

Tables are for reference data: unit conventions, capability matrices, paper
lists, parameter ranges.

A table is the wrong tool for an argument. If a cell needs a clause, a caveat, or
a "but", that content belongs in prose below the table.

## When density is correct

These pages are reference material and are allowed to be dense: the `api/`
pages, `data-schema.md`, `formal.md`, `raman-cascade.md`, `stability.md`,
`DATABASE.md`.

They still follow every rule above. "Dense" means the content is a
specification, not that the sentences are unreadable.

## Checklist before pushing

```bash
python scripts/check_docs.py        # links, nav, quadrants, reachability, style
python scripts/check_docs_code.py   # executes tutorial and how-to snippets
python scripts/build_gallery.py     # regenerates docs/gallery.md and docs/examples/
zensical build                      # the site still builds
```

The checker covers `docs/`, every root Markdown file, `examples/README.md` and
`notes/`. It skips `.github/copilot-instructions.md`, which is generated.

`docs/gallery.md` and `docs/examples/*.md` are generated from `examples/` and
committed, so the navigation and the link checks work in a fresh clone. Do not
edit them by hand; edit the script and regenerate. `build_gallery.py --check`
fails when they are out of date.

The link check matters most. A renamed page with a stale inbound link is the
failure this repository actually hits.
