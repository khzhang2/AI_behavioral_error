from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from biogeme import database as db
from biogeme import models
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Variable


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def maybe_float(value):
    return None if value is None else float(value)


def main() -> None:
    os.chdir(EXPERIMENT_DIR)
    data = pd.read_csv(OUTPUT_DIR / "ai_choices_wide.csv")
    data = data.loc[data["choice"] > 0].copy()
    data = data.drop(columns=["persona_id"])

    database = db.Database("kka_ai_sd", data)

    CHOICE = Variable("choice")
    AV_ES = Variable("avail_es")
    AV_BS = Variable("avail_bs")
    AV_WALK = Variable("avail_walk")
    AV_CAR = Variable("avail_car")

    ASC_ES = Beta("ASC_ES", 0, None, None, 0)
    ASC_BS = Beta("ASC_BS", 0, None, None, 0)
    ASC_WALK = Beta("ASC_WALK", 0, None, None, 0)

    B_TIME_ES = Beta("B_TIME_ES", 0, None, None, 0)
    B_TIME_BS = Beta("B_TIME_BS", 0, None, None, 0)
    B_TIME_WALK = Beta("B_TIME_WALK", 0, None, None, 0)
    B_TIME_CAR = Beta("B_TIME_CAR", 0, None, None, 0)
    B_ACCESS_SHARED = Beta("B_ACCESS_SHARED", 0, None, None, 0)
    B_ACCESS_CAR = Beta("B_ACCESS_CAR", 0, None, None, 0)
    B_EGRESS_SHARED = Beta("B_EGRESS_SHARED", 0, None, None, 0)
    B_EGRESS_CAR = Beta("B_EGRESS_CAR", 0, None, None, 0)
    B_PARKING = Beta("B_PARKING", 0, None, None, 0)
    B_COST = Beta("B_COST", 0, None, None, 0)
    B_AVAILABILITY = Beta("B_AVAILABILITY", 0, None, None, 0)
    B_FREE_FLOATING = Beta("B_FREE_FLOATING", 0, None, None, 0)
    B_PEDELEC = Beta("B_PEDELEC", 0, None, None, 0)

    V = {
        1: ASC_ES
        + B_TIME_ES * Variable("time_es")
        + B_ACCESS_SHARED * Variable("access_es")
        + B_EGRESS_SHARED * Variable("egress_es")
        + B_PARKING * Variable("parking_es")
        + B_COST * Variable("cost_es")
        + B_AVAILABILITY * Variable("availability_es")
        + B_FREE_FLOATING * Variable("freefloat_es"),
        2: ASC_BS
        + B_TIME_BS * Variable("time_bs")
        + B_ACCESS_SHARED * Variable("access_bs")
        + B_EGRESS_SHARED * Variable("egress_bs")
        + B_PARKING * Variable("parking_bs")
        + B_COST * Variable("cost_bs")
        + B_AVAILABILITY * Variable("availability_bs")
        + B_FREE_FLOATING * Variable("freefloat_bs")
        + B_PEDELEC * Variable("pedelec_bs"),
        3: ASC_WALK + B_TIME_WALK * Variable("time_walk"),
        4: B_TIME_CAR * Variable("time_car")
        + B_ACCESS_CAR * Variable("access_car")
        + B_EGRESS_CAR * Variable("egress_car")
        + B_PARKING * Variable("parking_car")
        + B_COST * Variable("cost_car"),
    }

    av = {1: AV_ES, 2: AV_BS, 3: AV_WALK, 4: AV_CAR}
    logprob = models.loglogit(V, av, CHOICE)

    biogeme = BIOGEME(database, logprob)
    biogeme.modelName = "kka_ai_sd_mnl"
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    results = biogeme.estimate()

    if hasattr(results, "get_estimated_parameters"):
        estimates = results.get_estimated_parameters()
    else:
        estimates = results.getEstimatedParameters()
    estimates.to_csv(OUTPUT_DIR / "biogeme_ai_estimates.csv", index=True)

    summary = {
        "n_rows_after_filter": int(len(data)),
        "null_loglikelihood": maybe_float(getattr(results.data, "nullLogLike", None)),
        "final_loglikelihood": maybe_float(getattr(results.data, "logLike", None)),
        "rho_square": maybe_float(getattr(results.data, "rhoSquare", None)),
        "rho_bar_square": maybe_float(getattr(results.data, "rhoBarSquare", None)),
        "number_of_parameters": int(results.data.nparam),
    }
    (OUTPUT_DIR / "biogeme_model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
