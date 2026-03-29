from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import median

import pandas as pd

from common import DATA_DIR, ensure_dir, safe_ratio, write_json


WIDE_PATH = DATA_DIR / "human_cleaned_wide.csv"

RATIO_COLUMNS = [
    ("TRAIN_TT", "train_time_multiplier"),
    ("TRAIN_CO", "train_cost_multiplier"),
    ("SM_TT", "sm_time_multiplier"),
    ("SM_CO", "sm_cost_multiplier"),
    ("CAR_TT", "car_time_multiplier"),
    ("CAR_CO", "car_cost_multiplier"),
]


def panel_baseline(values: pd.Series) -> float:
    unique_nonzero = sorted({int(value) for value in values.tolist() if int(value) != 0})
    if not unique_nonzero:
        return 0.0
    return float(median(unique_nonzero))


def build_codebook() -> dict:
    return {
        "survey_design_classification": {
            "design_type": "other",
            "specific_classification": "respondent-specific pivoted stated-preference design with recurring 9-task template families",
            "notes": [
                "The public data do not support labeling the survey as one single global orthogonal table.",
                "The public data also do not look like purely random independent draws.",
                "The reverse engineering therefore treats Swissmetro as a pivoted respondent-specific SP design with repeating template families."
            ],
        },
        "variables": {
            "GROUP": {"description": "survey group code as stored in the public data"},
            "SURVEY": {"description": "survey stratum code in the public data"},
            "SP": {"description": "stated-preference indicator"},
            "ID": {"description": "respondent identifier"},
            "PURPOSE": {
                "description": "trip purpose code",
                "codes": {
                    "1": "purpose code 1",
                    "3": "purpose code 3"
                }
            },
            "FIRST": {
                "description": "travel class indicator",
                "codes": {
                    "0": "second class",
                    "1": "first class"
                }
            },
            "TICKET": {"description": "ticket category code"},
            "WHO": {"description": "payer code"},
            "LUGGAGE": {"description": "luggage code"},
            "AGE": {"description": "age code"},
            "MALE": {
                "description": "sex indicator",
                "codes": {
                    "0": "female",
                    "1": "male"
                }
            },
            "INCOME": {"description": "income code"},
            "GA": {
                "description": "GA travelcard indicator",
                "codes": {
                    "0": "no GA travelcard",
                    "1": "has GA travelcard"
                }
            },
            "ORIGIN": {"description": "origin code"},
            "DEST": {"description": "destination code"},
            "TRAIN_AV": {"description": "train availability"},
            "SM_AV": {"description": "Swissmetro availability"},
            "CAR_AV": {"description": "car availability"},
            "TRAIN_TT": {"description": "train travel time"},
            "TRAIN_CO": {"description": "train travel cost"},
            "TRAIN_HE": {"description": "train headway"},
            "SM_TT": {"description": "Swissmetro travel time"},
            "SM_CO": {"description": "Swissmetro travel cost"},
            "SM_HE": {"description": "Swissmetro headway"},
            "SM_SEATS": {"description": "Swissmetro seat-configuration indicator"},
            "CAR_TT": {"description": "car travel time"},
            "CAR_CO": {"description": "car travel cost"},
            "CHOICE": {
                "description": "chosen alternative",
                "codes": {
                    "0": "missing choice",
                    "1": "TRAIN",
                    "2": "SWISSMETRO",
                    "3": "CAR"
                }
            }
        }
    }


def main() -> None:
    ensure_dir(DATA_DIR)
    if not WIDE_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned wide file: {WIDE_PATH}")

    wide = pd.read_csv(WIDE_PATH)
    template_counts: Counter[tuple[int, str]] = Counter()
    template_payloads: dict[tuple[int, str], list[dict]] = {}
    baseline_rows: list[dict] = []
    family_counts_by_survey: defaultdict[int, Counter[str]] = defaultdict(Counter)

    for respondent_id, panel in wide.groupby("ID", sort=True):
        ordered = panel.sort_values("task_position").copy()
        survey = int(ordered["SURVEY"].iloc[0])
        car_av = int(ordered["CAR_AV"].iloc[0])
        baselines = {
            "source_human_id": int(respondent_id),
            "survey_stratum": survey,
            "car_av": car_av,
            "ga": int(ordered["GA"].iloc[0]),
            "baseline_train_time": panel_baseline(ordered["TRAIN_TT"]),
            "baseline_train_cost": panel_baseline(ordered["TRAIN_CO"]),
            "baseline_sm_time": panel_baseline(ordered["SM_TT"]),
            "baseline_sm_cost": panel_baseline(ordered["SM_CO"]),
            "baseline_car_time": panel_baseline(ordered["CAR_TT"]),
            "baseline_car_cost": panel_baseline(ordered["CAR_CO"]),
        }
        baseline_rows.append(baselines)

        blueprint_rows: list[dict] = []
        template_key_parts: list[str] = []
        for row in ordered.itertuples(index=False):
            blueprint_row = {
                "survey_stratum": survey,
                "task_position": int(row.task_position),
                "train_time_multiplier": safe_ratio(row.TRAIN_TT, baselines["baseline_train_time"], digits=1),
                "train_cost_multiplier": safe_ratio(row.TRAIN_CO, baselines["baseline_train_cost"], digits=1),
                "train_headway": int(row.TRAIN_HE),
                "sm_time_multiplier": safe_ratio(row.SM_TT, baselines["baseline_sm_time"], digits=1),
                "sm_cost_multiplier": safe_ratio(row.SM_CO, baselines["baseline_sm_cost"], digits=1),
                "sm_headway": int(row.SM_HE),
                "sm_seats": int(row.SM_SEATS),
                "car_time_multiplier": safe_ratio(row.CAR_TT, baselines["baseline_car_time"], digits=1),
                "car_cost_multiplier": safe_ratio(row.CAR_CO, baselines["baseline_car_cost"], digits=1),
            }
            blueprint_rows.append(blueprint_row)
            template_key_parts.append(json.dumps(blueprint_row, sort_keys=True))

        template_key = (survey, "|".join(template_key_parts))
        template_counts[template_key] += 1
        template_payloads[template_key] = blueprint_rows
        family_counts_by_survey[survey][template_key[1]] += 1

    baseline_frame = pd.DataFrame(baseline_rows).sort_values(["survey_stratum", "source_human_id"])
    baseline_frame.to_csv(DATA_DIR / "reconstructed_panel_baselines.csv", index=False)

    catalog_rows: list[dict] = []
    survey_template_counters: defaultdict[int, int] = defaultdict(int)
    for (survey, template_blob), frequency in template_counts.most_common():
        survey_template_counters[survey] += 1
        template_id = f"S{survey}_TPL{survey_template_counters[survey]:03d}"
        payload = template_payloads[(survey, template_blob)]
        weight = frequency / baseline_frame.loc[baseline_frame["survey_stratum"] == survey].shape[0]
        for task_row in payload:
            catalog_rows.append(
                {
                    "survey_stratum": survey,
                    "template_id": template_id,
                    "template_frequency": int(frequency),
                    "template_weight": float(weight),
                    "provenance": "inferred_from_public_data",
                    **task_row,
                }
            )

    catalog_frame = pd.DataFrame(catalog_rows).sort_values(["survey_stratum", "template_id", "task_position"])
    catalog_frame.to_csv(DATA_DIR / "reconstructed_panel_catalog.csv", index=False)

    top_families = []
    for survey, counter in family_counts_by_survey.items():
        for rank, (_, frequency) in enumerate(counter.most_common(5), start=1):
            top_families.append({"survey_stratum": survey, "rank": rank, "frequency": int(frequency)})

    design_spec = {
        "design_type": "other",
        "specific_classification": "respondent-specific pivoted stated-preference design with recurring 9-task template families",
        "n_cleaned_respondents": int(wide["ID"].nunique()),
        "n_cleaned_observations": int(len(wide)),
        "tasks_per_respondent": sorted(wide.groupby("ID").size().unique().tolist()),
        "alternatives": ["TRAIN", "SWISSMETRO", "CAR"],
        "attributes_by_mode": {
            "TRAIN": ["travel_time", "travel_cost", "headway"],
            "SWISSMETRO": ["travel_time", "travel_cost", "headway", "seat_configuration"],
            "CAR": ["travel_time", "travel_cost", "availability"]
        },
        "exact_discrete_levels": {
            "TRAIN_HE": sorted(wide["TRAIN_HE"].unique().tolist()),
            "SM_HE": sorted(wide["SM_HE"].unique().tolist()),
            "SM_SEATS": sorted(wide["SM_SEATS"].unique().tolist())
        },
        "template_family_counts_by_survey": {
            str(survey): int(len(counter))
            for survey, counter in family_counts_by_survey.items()
        },
        "top_template_family_counts": top_families,
        "reconstruction_method": {
            "baseline_definition": "median of the non-zero unique within-respondent values for each time/cost attribute",
            "time_cost_template_representation": "ratios to respondent-specific panel baseline, rounded to one decimal to stabilize recurring template families",
            "discrete_attribute_representation": "exact public levels for headway and seat configuration",
            "template_sampling_rule": "sample template families by empirical frequency within survey stratum",
            "baseline_sampling_rule": "sample numeric baselines independently from empirical respondent baseline distributions within survey stratum"
        },
        "notes": [
            "The public data support a recurring-family interpretation rather than a single shared orthogonal table.",
            "Swissmetro and train headways take exact repeated public levels, while time and cost vary respondent by respondent around panel-specific baselines.",
            "This line therefore reconstructs fresh AI panels from inferred template families instead of reusing raw human task bundles."
        ]
    }
    write_json(DATA_DIR / "swissmetro_design_spec.json", design_spec)
    write_json(DATA_DIR / "swissmetro_codebook.json", build_codebook())

    md_lines = [
        "# Swissmetro Public Reverse-Engineering Note",
        "",
        "## Classification",
        "",
        "- `design_type = other`",
        "- `specific_classification = respondent-specific pivoted stated-preference design with recurring 9-task template families`",
        "- This public reverse engineering does not treat Swissmetro as a single global orthogonal table.",
        "- It also does not treat Swissmetro as purely random task generation.",
        "",
        "## Evidence from the public data",
        "",
        f"- cleaned respondents: `{wide['ID'].nunique()}`",
        f"- cleaned observations: `{len(wide)}`",
        f"- tasks per respondent: `{sorted(wide.groupby('ID').size().unique().tolist())}`",
        f"- `TRAIN_HE` levels: `{sorted(wide['TRAIN_HE'].unique().tolist())}`",
        f"- `SM_HE` levels: `{sorted(wide['SM_HE'].unique().tolist())}`",
        f"- `SM_SEATS` levels: `{sorted(wide['SM_SEATS'].unique().tolist())}`",
        "",
        "## Reconstruction rule used for AI panels",
        "",
        "- Deduplicate the cleaned sample to respondent profiles.",
        "- Compute respondent-level panel baselines for each time and cost attribute.",
        "- Convert each respondent panel to a normalized nine-task blueprint relative to those baselines.",
        "- Keep headway and seat-configuration values as exact public levels.",
        "- Deduplicate the normalized blueprints into a template catalog and sample template families by empirical frequency within survey stratum.",
        "- Sample fresh numeric baselines independently from the empirical baseline distributions and reconstruct new AI panels from those two ingredients.",
        "",
        "## Template-family counts by survey stratum",
        "",
    ]

    for survey, counter in family_counts_by_survey.items():
        md_lines.append(f"- survey `{survey}`: `{len(counter)}` distinct template families")

    md_lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This reverse engineering is grounded in the public `swissmetro.dat` structure and the public benchmark notebook. It is suitable for reproducible AI simulation work, but it is not claimed to recover the original historical DOE file exactly.",
            "",
        ]
    )
    (DATA_DIR / "swissmetro_reverse_engineering.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[reverse] wrote {len(catalog_frame)} template-task rows")
    print(f"[reverse] wrote {len(baseline_frame)} respondent baselines")


if __name__ == "__main__":
    main()
