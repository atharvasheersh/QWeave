# Manuscript status

`manuscript.tex` is an A4, two-column, Times-style internal first draft.
The problem, related work, implemented deterministic method, correctness
scope, and planned evaluation are written. Team author order and venue
format remain to be agreed. The manuscript deliberately reports no main
benchmark findings or PPO/GNN contribution, because those experiments and
modules are not present. Bibliography entries were checked against the
linked primary papers and IBM Quantum documentation on 2026-09-16.

Build when a TeX distribution is available from this folder with
`pdflatex manuscript.tex`, `bibtex manuscript`, and two further `pdflatex`
passes. No TeX engine was available in the local environment at drafting
time, so the source has not been compiled or visually inspected as a PDF.

Before external submission, freeze the protocol and dataset, add validated
results and limitations, review citations against full texts, agree author
order and contributions, and render-inspect every final page.
