# AI usage disclosure

This repository was developed with the assistance of AI tools, and the author
discloses that use here as required by the Journal of Open Source Software's
AI usage policy.

## Tools used

Conversational and command-line AI coding assistants (LLM-based agents,
including the `pi` coding-agent harness) were used at various points during
development, principally:

- **Code generation and refactoring** of solver modules, examples, and the
  test suite;
- **Debugging and code review** — auditing numerical kernels, hunting
  convention/sign bugs, and drafting reproduction scripts;
- **Documentation** — drafting docstrings, markdown pages, and README text;
- **Paper drafting** — assistance with drafting and copy-editing
  `paper/paper.md` and repository markdown files.

## Nature and scope of assistance

AI assistance was used as a productivity tool under the author's direction.
The tools did not set the research agenda, choose the project's goals, or
make architectural decisions autonomously.

## Human responsibility

The human author, **Dr. Hitaishi V**, reviewed, edited, and validated all
AI-assisted outputs. Specifically:

- The core design decisions — the typed-units foundation, the documented FFT
  and dispersion-sign conventions, the stability contract, the material-data
  provenance/licence model, and the scope of the formal Lean 4 specification —
  are the author's.
- Every physics-facing claim is validated against the published literature or
  closed-form ground truth by the code in `reproductions/` and the test suite;
  the author verified these checks and their outcomes.
- The author is responsible for the accuracy, originality, licensing, and
  integrity of all contents of this repository and of the associated JOSS
  submission.
