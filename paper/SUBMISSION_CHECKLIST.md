# QWeave submission checklist

The venue-neutral manuscript, DOCX, PDF, code, tests, and reviewed benchmark
package are ready for the author and venue decisions below. Complete these
items before an external submission or public preprint upload.

## Required metadata

- [ ] Confirm the paper title.
- [ ] Confirm author order independently of code ownership.
- [ ] Add each author's full affiliation and institutional address.
- [ ] Select the corresponding author and add the approved contact email.
- [ ] Add ORCID identifiers where available.
- [ ] Approve a CRediT contribution statement.
- [ ] Add funding, acknowledgements, and conflict-of-interest statements.
- [ ] State data/code availability and the permanent release URL/DOI.

## Venue preparation

- [ ] Select the target venue or preprint service.
- [ ] Apply its current official template, page limit, bibliography style, and
      anonymity rules.
- [ ] Confirm whether supplementary raw data and checkpoints may be uploaded.
- [ ] Run the venue's PDF compliance, font-embedding, accessibility, and
      metadata checks.
- [ ] Re-render and visually inspect every page after template conversion.

## Final scientific checks

- [x] Claims are bounded to the frozen benchmark and declared validators.
- [x] Mapping, routing, scheduling, and directed-cost effects are separated.
- [x] CP-SAT is described only as an initial-mapping oracle.
- [x] The pre-correction hardware-cost record is retained and disclosed.
- [x] Primary citation records and Qiskit references were checked.
- [x] GitHub CI passed on the pushed benchmark-v2 manuscript commit.
- [x] A clean clone installed from locked dependencies and passed 69 tests.
- [x] Published benchmark-v2 artifact hashes match in the clean clone.
- [x] Review the completed full benchmark-v2 rerun comparison: zero
      non-runtime record mismatches and zero training-summary mismatches.
- [ ] Every author reviews and approves the final manuscript and submission.
