from __future__ import annotations

from optima_common import INDICATOR_NAMES


PARAMETER_ORDER = [
    "LV_CAR_INTERCEPT",
    "LV_CAR_HIGH_EDU",
    "LV_CAR_TOP_MANAGER",
    "LV_CAR_EMPLOYEES",
    "LV_CAR_AGE_30_LESS",
    "LV_CAR_SCALED_INCOME",
    "LV_CAR_PARENTS",
    "SIGMA_CAR",
    "LV_ENV_INTERCEPT",
    "LV_ENV_CHILD_SUBURB",
    "LV_ENV_SCALED_INCOME",
    "LV_ENV_CITY_CENTER_KID",
    "LV_ENV_ARTISANS",
    "LV_ENV_HIGH_EDU",
    "LV_ENV_LOW_EDU",
    "SIGMA_ENV",
    "ASC_PT",
    "ASC_CAR",
    "B_COST",
    "B_TIME_PT",
    "B_TIME_CAR",
    "B_WAIT_WORK",
    "B_WAIT_OTHER",
    "B_DIST_WORK",
    "B_DIST_OTHER",
    "B_LV_CAR_TO_PT",
    "B_LV_ENV_TO_PT",
    "B_LV_CAR_TO_CAR",
    "B_LV_ENV_TO_CAR",
    "INTERCEPT_ENVIR01",
    "INTERCEPT_MOBIL05",
    "INTERCEPT_LIFSTY07",
    "INTERCEPT_ENVIR05",
    "INTERCEPT_MOBIL12",
    "INTERCEPT_LIFSTY01",
    "LOADING_MOBIL05_CAR",
    "LOADING_LIFSTY07_CAR",
    "LOADING_MOBIL12_ENV",
    "LOADING_LIFSTY01_ENV",
    "DELTA_1",
    "DELTA_2",
]


POSITIVE_PARAMETERS = {"SIGMA_CAR", "SIGMA_ENV", "DELTA_1", "DELTA_2"}


STRUCTURAL_CAR_COLUMNS = [
    "high_education",
    "top_manager",
    "employees",
    "age_30_less",
    "ScaledIncome",
    "car_oriented_parents",
]


STRUCTURAL_ENV_COLUMNS = [
    "childSuburb",
    "ScaledIncome",
    "city_center_as_kid",
    "artisans",
    "high_education",
    "low_education",
]


INDICATOR_SPECS = {
    "Envir01": {"latent": "car", "loading": -1.0, "normalized": True, "intercept": "INTERCEPT_ENVIR01"},
    "Mobil05": {"latent": "car", "loading": "LOADING_MOBIL05_CAR", "normalized": False, "intercept": "INTERCEPT_MOBIL05"},
    "LifSty07": {"latent": "car", "loading": "LOADING_LIFSTY07_CAR", "normalized": False, "intercept": "INTERCEPT_LIFSTY07"},
    "Envir05": {"latent": "env", "loading": 1.0, "normalized": True, "intercept": "INTERCEPT_ENVIR05"},
    "Mobil12": {"latent": "env", "loading": "LOADING_MOBIL12_ENV", "normalized": False, "intercept": "INTERCEPT_MOBIL12"},
    "LifSty01": {"latent": "env", "loading": "LOADING_LIFSTY01_ENV", "normalized": False, "intercept": "INTERCEPT_LIFSTY01"},
}


assert list(INDICATOR_SPECS.keys()) == INDICATOR_NAMES
