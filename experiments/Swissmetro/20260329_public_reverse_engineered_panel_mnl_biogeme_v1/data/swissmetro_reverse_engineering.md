# Swissmetro Public Reverse-Engineering Note

## Classification

- `design_type = other`
- `specific_classification = respondent-specific pivoted stated-preference design with recurring 9-task template families`
- This public reverse engineering does not treat Swissmetro as a single global orthogonal table.
- It also does not treat Swissmetro as purely random task generation.

## Evidence from the public data

- cleaned respondents: `752`
- cleaned observations: `6768`
- tasks per respondent: `[9]`
- `TRAIN_HE` levels: `[30, 60, 120]`
- `SM_HE` levels: `[10, 20, 30]`
- `SM_SEATS` levels: `[0, 1]`

## Reconstruction rule used for AI panels

- Deduplicate the cleaned sample to respondent profiles.
- Compute respondent-level panel baselines for each time and cost attribute.
- Convert each respondent panel to a normalized nine-task blueprint relative to those baselines.
- Keep headway and seat-configuration values as exact public levels.
- Deduplicate the normalized blueprints into a template catalog and sample template families by empirical frequency within survey stratum.
- Sample fresh numeric baselines independently from the empirical baseline distributions and reconstruct new AI panels from those two ingredients.

## Template-family counts by survey stratum

- survey `0`: `187` distinct template families
- survey `1`: `143` distinct template families

## Caveat

This reverse engineering is grounded in the public `swissmetro.dat` structure and the public benchmark notebook. It is suitable for reproducible AI simulation work, but it is not claimed to recover the original historical DOE file exactly.
