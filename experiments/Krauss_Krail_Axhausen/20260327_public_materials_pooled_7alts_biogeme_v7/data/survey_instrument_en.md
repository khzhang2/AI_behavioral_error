# Survey Instrument (English Public-Materials Briefing)

This file paraphrases the public English survey description from the paper into a machine-readable briefing for the AI respondent workflow. It is not claimed to be the original questionnaire script.

## Public survey frame

- The survey is a stated-preference experiment about urban mode choice.
- Each respondent sees six choice situations.
- Each choice situation concerns an intra-urban leisure trip in the respondent's city of residence.
- The scenario assumes no luggage.
- Respondents are assigned to exactly one sub-sample:
  - `SD`: e-scooter sharing, bike sharing, walking, private car
  - `MD`: carsharing, ridepooling, public transport, private car
- Respondents are told to ignore whether they are currently a member of a shared service.
- The original survey forced a choice among the modes presented and did not include an opt-out option.

## Implementation briefing for AI respondents

- Stay in the same persona for the entire survey.
- Use only the provided persona dossier and the choice-card attributes shown in the current task.
- Do not invent new attitudes, histories, or background facts beyond the dossier.
- Choose exactly one option in every task.
- For the grounding turn, return one strict JSON object restating the persona fields.
- For each choice task, return one strict JSON object in the form `{"choice":"A"}`.

## Mode notes used in prompt rendering

- `E-Scooter Sharing`: a shared micromobility service with scheme, availability, and battery range attributes.
- `Bike Sharing`: a shared bicycle service that may use a regular bike or a pedelec.
- `Walking`: direct walking for the same trip.
- `Private Car`: privately available household car access with access, egress, parking, and cost attributes.
- `Carsharing`: shared car access with station-based, free-floating, or hybrid operation.
- `Ridepooling`: pooled on-demand ride with access, waiting, detour, egress, cost, and crowding attributes.
- `Public Transport`: public transport with access, waiting, egress, cost, crowding, and transfers.
