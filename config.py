"""Central configuration for the SInERT Phase 1 data pipeline.

Project: "Évaluation approfondie, Benchmark multi-critères et Modélisation IA
pour la prédiction énergétique et l'estimation de la facturation de SUP'COM".

The raw data are "Daily Report" CSV exports of the VMU-C EM concentrator
(SInERT project), semicolon-separated, sampled every 15 minutes.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent   # .../sg_project (code only)
DEFAULT_DATA_DIR = PROJECT_DIR.parent           # dataset root (month folders)
OUTPUT_DIR = PROJECT_DIR / "output"

# --------------------------------------------------------------------------- #
# Meter blocks present in every export file
# --------------------------------------------------------------------------- #
# EM_VIRTUAL is the aggregated "Main AC meter" (verified identical to TGBT_TR2).
MAIN_BLOCK = "EM_VIRTUAL"

BLOCKS = [
    "EM_VIRTUAL",
    "TGBT_TR1",
    "TGBT_TR2",
    "Tour_B_(TGBT_D5)",
    "CLIM_SALLE_IT_(TGBT_D1)",
    "CLIM_TOUR_A_(TGBT_D6)",
    "TOUR_C_(TGBT_D10)",
    "TOUR_A_(TGBT_D14)",
    "SALLE_B201_(A_1_ETAGE_TOUR_B)",
    "SALLE_B112_(A._MEZAN_TOUR_B)",
    "SALLE_B101_(A._MEZAN_TOUR_B)",
    "ONDULEUR_1_(TGBT_DXX)",
    "ONDULEUR_2_(TGBT_DXX)",
    "PV_(CHAUFFERIE_PROV.)",
]

# --------------------------------------------------------------------------- #
# Variables extracted per block
# --------------------------------------------------------------------------- #
# logical name -> exact suffix in the raw header (after whitespace stripping).
# The full raw column is f"{block} {suffix}".
VARIABLES = {
    "kWh_abs":  "kWh (kWh) [ABS]",      # cumulative active energy counter
    "kvarh_abs": "kvarh (kvarh) [ABS]", # cumulative reactive energy counter
    "P_avg":    "kW sys (kW) [AVG]",    # mean active power over the step
    "Q_avg":    "kvar sys (kvar) [AVG]",# mean reactive power (negative = capacitive)
    "P_max":    "kW dmd (kW) [MAX]",    # max active demand over the step
    "S_max":    "kVA dmd (kVA) [MAX]",  # max apparent demand over the step
}

# Counters are cumulative -> handled by first-difference (energy per step).
COUNTER_VARS = ("kWh_abs", "kvarh_abs")
# Direct 15-min measurements -> handled by gap-length imputation rules.
MEASUREMENT_VARS = ("P_avg", "Q_avg", "P_max", "S_max")

# Output column naming convention: "{block}__{logical_var}"
COL = "{block}__{var}"

# --------------------------------------------------------------------------- #
# Time grid & imputation rules (Phase 1 - step 2 of the cahier de charges)
# --------------------------------------------------------------------------- #
STEP_MIN = 15                    # sampling period [minutes]
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96               # 24 h x 4
STEPS_PER_WEEK = 7 * STEPS_PER_DAY  # 672 -> sliding window of the cahier de charges

SHORT_GAP_STEPS = 4              # <= 1 h        -> linear interpolation
MEDIUM_GAP_STEPS = 192           # <= 2 days     -> seasonal-naive fill / energy spread
#                                  >  2 days     -> left as NaN, flagged in the report

MISSING_TOKEN = "---"            # raw missing-value marker

# Effective start of the analysis: the EM_VIRTUAL counter produced nothing
# before 2022-06-28 (June 2022 is almost entirely NaN), so everything before
# 1 July 2022 is dropped from the cleaned data. Set to None to keep everything.
ANALYSIS_START = "2022-07-01"

# --------------------------------------------------------------------------- #
# Calendars used by the behaviour analysis (Phase 1.4)
# --------------------------------------------------------------------------- #
# Tariff seasons follow the STEG MT contract (contrat_mt.pdf, art. 9):
#   "hiver" = 1 Sept -> 31 May, "été" = 1 June -> 31 August.
TARIFF_WINTER_MONTHS = (9, 10, 11, 12, 1, 2, 3, 4, 5)   # Sep..May
TARIFF_SUMMER_MONTHS = (6, 7, 8)                        # Jun..Aug

# Academic calendar (documented assumption, adjustable): study periods are all
# weeks outside the summer break (July-August) and the winter break
# (20 Dec -> 6 Jan). Exams/Ramadan are not modelled at this stage.
SUMMER_BREAK = ((7, 1), (8, 31))     # (start month, day), (end month, day)
WINTER_BREAK = ((12, 20), (1, 6))

# Behaviour thresholds (documented definitions, see behavior.py docstrings)
PEAK_DURATION_FRAC = 0.90    # a step belongs to the peak if P >= 90% of daily Pmax
ACTIVATION_MARGIN = 0.15     # site "active" if P >= daily_min + 15% of daily range

# --------------------------------------------------------------------------- #
# Long-gap imputation (analysis stage)
# --------------------------------------------------------------------------- #
# NaN runs longer than MEDIUM_GAP_STEPS are filled by a seasonal-pattern
# method (see cleaning.impute_long_gaps_by_pattern): shapes from previous
# years' same calendar month, energy totals conserved EXACTLY from the
# counter span. Every imputed step is flagged; benchmark samples overlapping
# the gap are marked so metrics can be reported with and without them.
IMPUTE_LONG_GAPS = True
