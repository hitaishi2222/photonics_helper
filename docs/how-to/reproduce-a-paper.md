---
quadrant: how-to
---

# Reproduce a result from a published paper

## Before you start

You want to know whether this library reproduces a specific paper before you
build on it. Every folder under `reproductions/` holds a `parameters.json`, a
`reproduce.py` that asserts its answer against a closed-form or published
reference, and a `README.md` recording the outcome.

Pick the folder whose physics matches yours. Each one takes seconds to about a
minute.

## Run one

```bash
python reproductions/stolen_lin_1978_spm/reproduce.py
```

The script prints its cases and exits non-zero if an assertion fails. Success
looks like this:

```
SPM validation passed:
  phi=1.0pi: peaks=2 (expected 2), max|Δ|=1.44e-14
  phi=2.0pi: peaks=3 (expected 3), max|Δ|=1.38e-14
  phi=4.0pi: peaks=5 (expected 5), max|Δ|=5.94e-14
  phi=8.0pi: peaks=9 (expected 9), max|Δ|=2.66e-13
```

Each line is one self-phase-modulation case: the number of spectral peaks the
numerical spectrum shows, the number the analytic solution predicts, and the
largest pointwise difference between them. `1e-14` on a normalised spectrum is
round-off.

## Read the assertion, not the prose

The pass criterion is in `reproduce.py`, and it is two lines:

- the maximum spectral difference must stay under `params["spectrum_tolerance"]`
- every case's peak count must equal the analytic prediction

Everything else on the page is commentary. If you want to know what a
reproduction actually guarantees, read those two lines in its `reproduce.py` and
the tolerance in its `parameters.json`.

You can call the same function without the shell:

```python
import contextlib
import io
import sys

sys.path.insert(0, "reproductions")
from stolen_lin_1978_spm.reproduce import validate

# The function prints the figure path, which differs per machine. Capture the
# call and report only the numbers.
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    result = validate()

print(f"cases      = {len(result['cases'])}")
print(f"max |delta| = {result['max_abs_spectrum_diff']:.2e}")
```

```
cases      = 4
max |delta| = 2.66e-13
```

## Interpret a recorded deviation

Some folders carry a caveat rather than a clean pass, and the `README.md` in that
folder says which arm of the paper was out of scope. Two rules keep this honest:

- A deviation with a stated cause is a result. A deviation without one is a bug.
- A reproduction that had to tune its own parameters does not count. The harness
  raises `ReproductionError` when a parameter is inside the tolerance it is meant
  to be outside, and you should treat a raised error as the answer, not as an
  obstacle to work around.

## You know it worked when

The script exits 0 and you can state the tolerance and the reference it is
measured against in one sentence. If you cannot, you have not reproduced the
paper yet; you have run a script.

## Where the background is

[Reproductions](../reproductions.md) lists all of them with the stack each one
exercises and its headline outcome.
[Data provenance](../data-provenance.md) covers the material coefficients those
reproductions depend on.
