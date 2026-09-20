# Manuscript status

`manuscript.tex` is the editable A4, two-column, Times-style source. It now
documents the deterministic path, fixed-route scheduler, guarded Gymnasium
environment, GNN--PPO trainer, frozen benchmark v1 split, ablations, and the
measured limitations. The generated Word and PDF copies are kept separately
under `docx/` and `pdf/`. Team author order and venue format remain to be
agreed. Bibliography entries were checked against the linked primary papers
and IBM Quantum documentation on 2026-09-16.

Build the TeX source when a TeX distribution is available from this folder with
`pdflatex manuscript.tex`, `bibtex manuscript`, and two further `pdflatex`
passes. The repository PDF is exported from the matching Word artifact because
no TeX engine was available in the local environment; both generated artifacts
are visually inspected after each meaningful update.

Before external submission, agree author order and contributions, select the
venue template, review citations against full texts, and run a larger
preregistered benchmark with stronger training and independent reruns.
