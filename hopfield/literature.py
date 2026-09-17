"""
hopfield.literature -- published values with their source, plus derived numbers.

Every number here comes from a cited paper. The pipeline uses them to test the
two frameworks. None of them is fitted.
"""
from __future__ import annotations

import numpy as np

from .epistasis import cv_from_fold

# --------------------------------------------------------------------------
# 1. MUTATION RATES BY GENOTYPE (mutation accumulation experiments). There is no
#    viability censoring here, because the genotypes are built and then measured.
# --------------------------------------------------------------------------
MUTATION_RATES = {
    "morrison1993_yeast": dict(
        source="Morrison, Johnson, Johnston & Sugino, EMBO J 12:1467 (1993)",
        system="S. cerevisiae, URA3 reporter, rates relative to wild type",
        wt=1.0, pf_lost=130.0, mmr_lost=41.0, both_lost=2.0e4,
        note="The double mutant is a homozygous diploid; the single mutants were "
             "measured separately. The authors called the result "
             "'multiplicative' because it is within one order of magnitude.",
    ),
}

# --------------------------------------------------------------------------
# 2. TUMOUR MUTATIONAL BURDEN (same cohort and platform where possible).
#    WARNING: TMB is an ACCUMULATED BURDEN, not a rate, and the double mutant is
#    censored from above (error-induced extinction). It cannot test the model.
# --------------------------------------------------------------------------
TMB = dict(
    source="JCO Precision Oncology, 499 colon tumours, single platform",
    mss_pole_wt=6.0, msi_h=54.0, pole_mut_mss=158.0,
    double_reported=[400.0, 203.8],
    double_source="ColoSeq (>400 mut/Mb); endometrial case with MSH6 loss (203.8)",
    caveat="Viability censoring: pol2-P301R msh6D is NOT VIABLE in yeast "
           "(error-induced extinction). Tumours above the multiplicative "
           "prediction simply do not exist to be sequenced.",
)

# --------------------------------------------------------------------------
# 3. REPLICATION TIMING GRADIENTS (cell lines with defined genotypes).
#    CV(alpha) and CV(beta) come from here, WITHOUT fitting.
# --------------------------------------------------------------------------
RT_GRADIENTS = dict(
    source="Cell Genomics (2023), defined-genotype cell lines, early->late",
    mmrd_only=[1.60, 1.09],        # HCT116, LS180
    mmrd_plus_pole=2.24,           # HT115
    mmr_attributable=2.5,          # gradient due to MMR in repair-proficient cells
    note="The authors infer that POLE mutations are enriched in late replicating "
         "regions. So both escapes vary along the SAME axis and in the SAME "
         "direction (rho > 0).",
)

# --------------------------------------------------------------------------
# 4. THE FINITE-WINDOW MECHANISM, MEASURED DIRECTLY.
# --------------------------------------------------------------------------
WINDOW = dict(
    source="Nature Communications (2024), more than 20,000 replication errors "
           "followed in single E. coli cells",
    finding="Many mutations come from errors that MMR DETECTS but repairs "
            "inefficiently. The limited efficiency is caused by a TIME "
            "constraint: the strand discrimination signal is transient. Repair "
            "capacity also varies from cell to cell.",
    relevance="This is the CV(gamma) of the law, measured.",
)

SATURATION_PRIOR = dict(
    source="Schaaper & Radman, EMBO J 8:3511 (1989)",
    finding="The very strong mutator effect of mutD5 comes from SATURATION of "
            "MMR by too many replication errors. The defect is transient: it "
            "goes away in late log phase, when replication is stopped, and "
            "when mutH or mutL are overexpressed.",
    relevance="Saturation as a MECHANISM. Our framework gives it as a BOUND.",
)

FORCE_BOUND = dict(
    source="Owen, Gingrich & Horowitz, PRX 10:011066 (2020)",
    bound="|nu - 1| <= (m-1) tanh(F_EES / 4 kT)",
    relevance="This bounds discrimination by the thermodynamic FORCE: with enough "
              "driving you recover all m stages. Our result is the other axis: "
              "with a finite activity budget the bound is NOT reachable.",
)


# ------------------------------------------------------- derived quantities ---
def observed_epistasis(key="morrison1993_yeast") -> dict:
    d = MUTATION_RATES[key]
    E = (d["both_lost"] * d["wt"]) / (d["pf_lost"] * d["mmr_lost"])
    return dict(E=float(E), multiplicative_prediction=d["pf_lost"] * d["mmr_lost"],
                observed=d["both_lost"], source=d["source"],
                sign="super-multiplicative" if E > 1 else "sub-multiplicative")


def tmb_epistasis() -> list:
    out = []
    for obs in TMB["double_reported"]:
        E = (obs * TMB["mss_pole_wt"]) / (TMB["pole_mut_mss"] * TMB["msi_h"])
        out.append(dict(double=obs, E=float(E),
                        additive_prediction=TMB["mss_pole_wt"]
                        + (TMB["msi_h"] - TMB["mss_pole_wt"])
                        + (TMB["pole_mut_mss"] - TMB["mss_pole_wt"]),
                        multiplicative_prediction=TMB["msi_h"]
                        * TMB["pole_mut_mss"] / TMB["mss_pole_wt"]))
    return out


def predicted_epistasis_from_rt() -> dict:
    """E - 1 = rho CV(alpha) CV(beta), with NO free parameter: both CVs come from
    measured replication timing gradients, and rho <= 1 bounds it from above."""
    g = RT_GRADIENTS
    cv_beta = cv_from_fold(g["mmr_attributable"])
    excess = [g["mmrd_plus_pole"] / x for x in g["mmrd_only"]]
    cv_alpha = [cv_from_fold(x) for x in excess]
    lo = min(cv_alpha) * cv_beta
    hi = max(cv_alpha) * cv_beta
    return dict(cv_beta=float(cv_beta),
                cv_alpha_range=(float(min(cv_alpha)), float(max(cv_alpha))),
                E_minus_1_range=(float(0.5 * lo), float(1.0 * hi)),
                rho_range=(0.5, 1.0), source=g["source"])


def power_required(effect: float, n_bins: int = 10, sigmas: float = 3.0) -> dict:
    """How many mutations you need to see `effect` at `sigmas` with `n_bins` bins."""
    per_bin = (sigmas / effect) ** 2
    return dict(per_bin=float(per_bin), total=float(per_bin * n_bins),
                ultramutated_genome=(1e5, 1e6),
                genomes_needed=float(per_bin * n_bins / 3e5))
