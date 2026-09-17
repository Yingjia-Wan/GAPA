"""Canonical filesystem locations for the GAPA repository.

Every path in this repo is derived from :data:`REPO_ROOT`, which is resolved from this
file's own location. Import from here instead of reconstructing paths with
``Path(__file__).parent`` or bare relative strings, so that scripts keep working
regardless of where they are invoked from or how deeply they are nested.

    from gapa.paths import DATA_DIR, PROMPTS_DIR, resolve
"""

from __future__ import annotations

from pathlib import Path

# gapa/paths.py -> gapa/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- shared inputs -------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIG_JSON = REPO_ROOT / "config.json"
EXPERIMENTS_JSON = REPO_ROOT / "experiments.json"

# --- paper sections ------------------------------------------------------------
HUMAN_ANALYSIS_DIR = REPO_ROOT / "human_analysis"        # §4
LLM_ANALYSIS_DIR = REPO_ROOT / "llm_analysis"            # §5
TRAINING_DIR = REPO_ROOT / "training"                    # §6.1
HP_SEARCH_DIR = TRAINING_DIR / "hp_search"
PREDICTOR_DIR = TRAINING_DIR / "predictor"
NOVEL_EXTRACTOR_DIR = REPO_ROOT / "novel_extractor"      # §3 (novel-extracted attributes)
LITBANK_DIR = REPO_ROOT / "litbank_analysis"             # §6.2

# --- derived subpaths ----------------------------------------------------------
NOISE_CEILING_DIR = HUMAN_ANALYSIS_DIR / "noise_ceiling"
EXTRACTED_DIR = NOVEL_EXTRACTOR_DIR / "extracted"
SOURCE_DIR = NOVEL_EXTRACTOR_DIR / "source"
LLM_RESULTS_DIR = LLM_ANALYSIS_DIR / "results"
LLM_PLOTS_DIR = LLM_ANALYSIS_DIR / "plots"

# --- run outputs ---------------------------------------------------------------
# Not shipped with the repo (gitignored, and absent from the lightweight clone).
# Defined here so the training/sweep scripts stay runnable against a full results tree.
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_FURTHER = REPO_ROOT / "results_further"
RESULTS_BESTRUNS = REPO_ROOT / "results_bestruns"
RESULTS_HPSEARCH = REPO_ROOT / "results_hpsearch"
LOG_DIR = REPO_ROOT / "log"
WANDB_DIR = REPO_ROOT / "wandb_runs"


def resolve(path, base: Path = REPO_ROOT) -> Path:
    """Resolve *path*, anchoring relative paths to *base* (the repo root by default).

    Absolute paths are returned unchanged. This is what makes a relative argument
    like ``--results-dir results_further`` mean the same thing no matter which
    directory the script was launched from.
    """
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


__all__ = [name for name in dir() if not name.startswith("_")]
