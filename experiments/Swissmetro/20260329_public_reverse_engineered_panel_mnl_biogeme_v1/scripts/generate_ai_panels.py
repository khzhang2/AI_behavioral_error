from __future__ import annotations

import argparse
import random

import pandas as pd

from common import DATA_DIR, OUTPUT_DIR, car_availability_text, ensure_dir, read_json, reconstruct_from_ratio, utc_timestamp, write_json


def choose_weighted(rng: random.Random, frame: pd.DataFrame, weight_col: str) -> pd.Series:
    total_weight = float(frame[weight_col].sum())
    draw = rng.random() * total_weight
    cumulative = 0.0
    for _, row in frame.iterrows():
        cumulative += float(row[weight_col])
        if draw <= cumulative:
            return row
    return frame.iloc[-1]


def load_catalog(catalog: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    grouped: dict[tuple[int, str], pd.DataFrame] = {}
    for (survey_stratum, template_id), frame in catalog.groupby(["survey_stratum", "template_id"], sort=False):
        grouped[(int(survey_stratum), str(template_id))] = frame.sort_values("task_position").copy()
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    config = read_json(DATA_DIR / "experiment_config.json")
    rng = random.Random(int(config["master_seed"]) + int(args.run_id))

    profiles = pd.read_csv(DATA_DIR / "human_respondent_profiles.csv")
    catalog = pd.read_csv(DATA_DIR / "reconstructed_panel_catalog.csv")
    baselines = pd.read_csv(DATA_DIR / "reconstructed_panel_baselines.csv")

    run_label = f"ai_run_{args.run_id:02d}"
    run_output_dir = OUTPUT_DIR / run_label
    ensure_dir(run_output_dir)

    n_respondents = int(config["n_synthetic_respondents_per_run"])
    tasks_per_respondent = int(config["tasks_per_respondent"])

    template_lookup = load_catalog(catalog)
    template_weights = catalog[["survey_stratum", "template_id", "template_frequency"]].drop_duplicates().copy()
    baseline_groups = {
        (int(survey), int(car_av)): frame.copy()
        for (survey, car_av), frame in baselines.groupby(["survey_stratum", "car_av"], sort=False)
    }
    baseline_groups_fallback = {
        int(survey): frame.copy()
        for survey, frame in baselines.groupby("survey_stratum", sort=False)
    }

    persona_rows: list[dict] = []
    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    for synthetic_id in range(1, n_respondents + 1):
        sampled_profile = profiles.iloc[rng.randrange(len(profiles))].to_dict()
        survey_stratum = int(sampled_profile["SURVEY"])
        car_av = int(sampled_profile["CAR_AV"])

        template_pool = template_weights.loc[template_weights["survey_stratum"] == survey_stratum].copy()
        template_row = choose_weighted(rng, template_pool, "template_frequency")
        template_id = str(template_row["template_id"])
        template_frame = template_lookup[(survey_stratum, template_id)]

        baseline_pool = baseline_groups.get((survey_stratum, car_av))
        if baseline_pool is None or baseline_pool.empty:
            baseline_pool = baseline_groups_fallback[survey_stratum]
        baseline_row = baseline_pool.iloc[rng.randrange(len(baseline_pool))].to_dict()

        ga = int(sampled_profile["GA"])
        anchor_train_cost = 0 if ga == 1 else int(round(float(baseline_row["baseline_train_cost"])))
        anchor_sm_cost = 0 if ga == 1 else int(round(float(baseline_row["baseline_sm_cost"])))
        anchor_car_cost = 0 if car_av == 0 else int(round(float(baseline_row["baseline_car_cost"])))
        anchor_payload = {
            "anchor_train_time": int(round(float(baseline_row["baseline_train_time"]))),
            "anchor_train_cost": anchor_train_cost,
            "anchor_sm_time": int(round(float(baseline_row["baseline_sm_time"]))),
            "anchor_sm_cost": anchor_sm_cost,
            "anchor_car_time": 0 if car_av == 0 else int(round(float(baseline_row["baseline_car_time"]))),
            "anchor_car_cost": anchor_car_cost,
        }

        persona_row = {
            "run_label": run_label,
            "run_id": int(args.run_id),
            "synthetic_respondent_id": synthetic_id,
            "persona_id": f"SMAI{args.run_id:02d}_{synthetic_id:04d}",
            "source_profile_id": int(sampled_profile["ID"]),
            "source_baseline_id": int(baseline_row["source_human_id"]),
            "survey_stratum": survey_stratum,
            "survey_text": str(sampled_profile["survey_text"]),
            "group_code": int(sampled_profile["GROUP"]),
            "purpose_code": int(sampled_profile["PURPOSE"]),
            "purpose_text": str(sampled_profile["purpose_text"]),
            "first_class": int(sampled_profile["FIRST"]),
            "first_class_text": str(sampled_profile["first_class_text"]),
            "ticket_code": int(sampled_profile["TICKET"]),
            "ticket_text": str(sampled_profile["ticket_text"]),
            "who_code": int(sampled_profile["WHO"]),
            "who_text": str(sampled_profile["who_text"]),
            "luggage_code": int(sampled_profile["LUGGAGE"]),
            "luggage_text": str(sampled_profile["luggage_text"]),
            "age_code": int(sampled_profile["AGE"]),
            "age_text": str(sampled_profile["age_text"]),
            "male": int(sampled_profile["MALE"]),
            "sex_text": str(sampled_profile["sex_text"]),
            "income_code": int(sampled_profile["INCOME"]),
            "income_text": str(sampled_profile["income_text"]),
            "ga": ga,
            "ga_text": str(sampled_profile["ga_text"]),
            "origin_code": int(sampled_profile["ORIGIN"]),
            "origin_text": str(sampled_profile["origin_text"]),
            "dest_code": int(sampled_profile["DEST"]),
            "dest_text": str(sampled_profile["dest_text"]),
            "car_av": car_av,
            "car_av_text": car_availability_text(car_av),
            "template_id": template_id,
            "generated_at_utc": utc_timestamp(),
            **anchor_payload,
        }
        persona_rows.append(persona_row)

        for task in template_frame.itertuples(index=False):
            train_tt = reconstruct_from_ratio(baseline_row["baseline_train_time"], task.train_time_multiplier)
            train_co = 0 if ga == 1 else reconstruct_from_ratio(baseline_row["baseline_train_cost"], task.train_cost_multiplier)
            sm_tt = reconstruct_from_ratio(baseline_row["baseline_sm_time"], task.sm_time_multiplier)
            sm_co = 0 if ga == 1 else reconstruct_from_ratio(baseline_row["baseline_sm_cost"], task.sm_cost_multiplier)
            car_tt = 0 if car_av == 0 else reconstruct_from_ratio(baseline_row["baseline_car_time"], task.car_time_multiplier)
            car_co = 0 if car_av == 0 else reconstruct_from_ratio(baseline_row["baseline_car_cost"], task.car_cost_multiplier)

            task_id = f"{persona_row['persona_id']}_T{int(task.task_position):02d}"
            wide_row = {
                "run_label": run_label,
                "run_id": int(args.run_id),
                "synthetic_respondent_id": synthetic_id,
                "persona_id": persona_row["persona_id"],
                "task_position": int(task.task_position),
                "task_id": task_id,
                "template_id": template_id,
                "survey_stratum": survey_stratum,
                "GROUP": persona_row["group_code"],
                "PURPOSE": persona_row["purpose_code"],
                "FIRST": persona_row["first_class"],
                "TICKET": persona_row["ticket_code"],
                "WHO": persona_row["who_code"],
                "LUGGAGE": persona_row["luggage_code"],
                "AGE": persona_row["age_code"],
                "MALE": persona_row["male"],
                "INCOME": persona_row["income_code"],
                "GA": ga,
                "ORIGIN": persona_row["origin_code"],
                "DEST": persona_row["dest_code"],
                "TRAIN_AV": 1,
                "SM_AV": 1,
                "CAR_AV": car_av,
                "TRAIN_TT": train_tt,
                "TRAIN_CO": train_co,
                "TRAIN_HE": int(task.train_headway),
                "SM_TT": sm_tt,
                "SM_CO": sm_co,
                "SM_HE": int(task.sm_headway),
                "SM_SEATS": int(task.sm_seats),
                "CAR_TT": car_tt,
                "CAR_CO": car_co,
            }
            wide_rows.append(wide_row)

            alt_rows = [
                (1, "TRAIN", "A", 1, train_tt, train_co, int(task.train_headway), 0),
                (2, "SWISSMETRO", "B", 1, sm_tt, sm_co, int(task.sm_headway), int(task.sm_seats)),
                (3, "CAR", "C", car_av, car_tt, car_co, 0, 0),
            ]
            for mode_id, alt_name, display_label, is_available, travel_time, travel_cost, headway, seat_configuration in alt_rows:
                long_rows.append(
                    {
                        "run_label": run_label,
                        "run_id": int(args.run_id),
                        "synthetic_respondent_id": synthetic_id,
                        "persona_id": persona_row["persona_id"],
                        "task_position": int(task.task_position),
                        "task_id": task_id,
                        "template_id": template_id,
                        "survey_stratum": survey_stratum,
                        "alternative_id": mode_id,
                        "alternative_name": alt_name,
                        "display_label": display_label,
                        "is_available": int(is_available),
                        "travel_time": int(travel_time),
                        "travel_cost": int(travel_cost),
                        "headway": int(headway),
                        "seat_configuration": int(seat_configuration),
                    }
                )

    persona_frame = pd.DataFrame(persona_rows).sort_values("synthetic_respondent_id")
    wide_frame = pd.DataFrame(wide_rows).sort_values(["synthetic_respondent_id", "task_position"])
    wide_frame["custom_id"] = range(1, len(wide_frame) + 1)
    long_frame = pd.DataFrame(long_rows).sort_values(["synthetic_respondent_id", "task_position", "alternative_id"])

    persona_frame.to_csv(run_output_dir / "persona_samples.csv", index=False)
    wide_frame.to_csv(run_output_dir / "reconstructed_panels_wide.csv", index=False)
    long_frame.to_csv(run_output_dir / "reconstructed_panels_long.csv", index=False)
    write_json(
        run_output_dir / "panel_generation_manifest.json",
        {
            "run_label": run_label,
            "run_id": int(args.run_id),
            "n_personas": int(len(persona_frame)),
            "n_tasks": int(len(wide_frame)),
            "n_long_rows": int(len(long_frame)),
            "tasks_per_respondent": tasks_per_respondent,
            "generated_at_utc": utc_timestamp(),
            "sampling_policy": {
                "profiles": "sampled with replacement from empirical cleaned respondent profiles",
                "templates": "sampled by empirical frequency within survey stratum",
                "baselines": "sampled independently from empirical baseline distributions within survey stratum and car availability"
            }
        },
    )

    print(f"[generate] {run_label}: personas={len(persona_frame)} tasks={len(wide_frame)}")


if __name__ == "__main__":
    main()
