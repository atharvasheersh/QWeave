# Manuscript status

`manuscript.tex` is the editable A4, two-column, Times-style source. It now
documents the deterministic path, fixed-route scheduler, wider semantic
checks, directed-coupler lowering, the guarded Gymnasium environment, and the
full benchmark-v2 training and ablation study. The reported result is bounded:
PPO training improves on the untrained policy, while the tested message passing
and learned paths do not improve paired median outcomes over the deterministic
references. The generated Word and PDF copies are kept separately under
`docx/` and `pdf/`. The author order is Atharva Sheersh Pandey, Diptesh Das,
Shrivardhini N, Haridasu Sreedhar, and faculty guide Prof. Bhuvaneswari M.
The venue format remains to be agreed.
Bibliography entries and their supporting claims were checked against the
linked primary records and IBM Quantum documentation on 2026-09-21. See the
[academic audit](../docs/ACADEMIC_AUDIT_2026-09-21.md) and the
[submission checklist](SUBMISSION_CHECKLIST.md).

Build the TeX source when a TeX distribution is available from this folder with
`pdflatex manuscript.tex`, `bibtex manuscript`, and two further `pdflatex`
passes. The repository PDF is exported from the matching Word artifact because
no TeX engine was available in the local environment; both generated artifacts
are visually inspected after each meaningful update.

Before external submission, agree author order and contributions, select the
venue template, review citations against full texts, and obtain an independent
rerun of benchmark v2 on a second clean machine. Device-calibration claims
remain out of scope until a calibration-derived study is frozen and run.
