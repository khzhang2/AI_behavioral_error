# Swissmetro Survey Briefing (Public Reverse Engineering)

This briefing is a public-materials reconstruction for AI respondents. It is not claimed to be the exact historical Swissmetro questionnaire script.

## Survey frame

- You are participating in a transport stated-preference survey.
- The survey concerns the same trip context across nine consecutive choice tasks.
- In each task, you choose one option among three travel modes:
  - `A = Train`
  - `B = Swissmetro`
  - `C = Car`
- Car may be unavailable for some respondents. If the card marks car as unavailable, do not choose it.
- Use only the persona dossier, the trip anchor, and the attributes displayed on the current card.
- Do not invent extra attitudes, habits, memories, or outside facts.

## How to answer

- First, complete one grounding turn by restating your respondent dossier as strict JSON only.
- Then answer each of the nine choice tasks one by one.
- For every choice task, choose exactly one alternative.
- For every choice task, return strict JSON only in the form `{"choice_label":"A"}`.

## Attribute notes

- `travel time` is the total in-vehicle time shown for that mode.
- `travel cost` is the trip cost shown on the card.
- `headway` is the service interval between departures and is shown for train and Swissmetro.
- `seat configuration` is shown for Swissmetro only.
- `availability` indicates whether car is available for the trip.
