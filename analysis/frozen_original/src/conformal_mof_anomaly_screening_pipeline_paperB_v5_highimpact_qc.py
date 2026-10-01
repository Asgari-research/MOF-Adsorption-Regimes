#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conformal MOF Anomaly Screening — Paper-B v5 High-Impact QC/Modes Pipeline
====================================================

Single-file, restart-safe, paper-production Python pipeline for the manuscript:

    "From Rankings to Certificates: Conformal Anomaly Screening of Exceptional
    MOF Adsorbents" / "Conformal MOF Anomaly Screening: Risk-Controlled
    Shortlists of Exceptional Adsorbents Beyond RMSE"

This script is intentionally written as a complete Visual Studio / VS Code
friendly analysis program rather than as a compact notebook. It is designed for
ARC-MOF-style CSV files placed in the same folder as this .py file, although it
also supports an optional data_raw/arc_mof/ folder if you prefer a cleaner layout.

WHAT THIS SCRIPT DOES
---------------------
1. Scans all CSV files rapidly and writes a data audit:
   - row/column counts
   - numeric/categorical/date-like columns
   - missingness by file and by column
   - likely identifier columns
   - candidate target columns

2. Builds a restart-safe merged modelling table from clean_data.csv if present,
   or from raw adsorption/descriptor/topology/cluster CSVs when clean_data.csv is absent:
   - geometric_properties.csv
   - all_topology_lists.csv
   - geo-clusters.csv
   - mc-clusters.csv
   - func-clusters.csv
   - flig-clusters.csv
   - optional process/post-combustion files when joinable

3. Detects the four adsorption targets, with fuzzy matching:
   - CO2 uptake at 0.015 bar
   - CO2 uptake at 0.15 bar
   - CH4 uptake at 5.8 bar
   - CH4 uptake at 65 bar

4. Creates feature families:
   - geometry only
   - enriched geometry
   - topology only
   - geometry + topology
   - geometry + topology + cluster-family proxies
   - optional process-enhanced family, when safe columns exist

5. Runs CPU-friendly models:
   - Ridge regression
   - HistGradientBoostingRegressor
   - RandomForestRegressor

6. Uses repeated random and grouped splits:
   - random split
   - topology-grouped split, if topology labels exist
   - geometry-cluster split, if geo-clusters.csv or equivalent exists
   - chemistry/metal/function/linker grouped split, if cluster columns exist

7. Constructs conformal decision outputs:
   - split-conformal prediction intervals
   - lower confidence bound ranking
   - confident elite / possible elite / uncertain candidate classes
   - positive-surprise residual anomaly certificates
   - negative-surprise residual anomaly certificates
   - two-sided anomaly certificates
   - approximate conformal p-values for residual surprises

8. Saves all outputs in reusable forms:
   - CSV tables
   - pickle/joblib objects
   - JSON manifests
   - figure-data CSVs for every panel
   - publication-style composite figures as PNG, PDF, and SVG
   - LaTeX tables for manuscript and SI
   - logs that make the run auditable and resumable

HOW TO RUN IN VISUAL STUDIO / VS CODE
-------------------------------------
1. Put this file in the same folder as your CSV files, or place CSV files under:
       data_raw/arc_mof/
2. Open the folder in VS Code.
3. Create/activate a Python environment.
4. Install required packages:
       pip install pandas numpy scipy scikit-learn matplotlib joblib openpyxl tqdm jinja2
5. Run:
       python conformal_mof_anomaly_screening_pipeline.py
   Optional: choose processor usage and model size from the command line, for example:
       python conformal_mof_anomaly_screening_pipeline.py --n_jobs 2 --model_profile balanced
       python conformal_mof_anomaly_screening_pipeline.py --n_jobs 4 --model_profile comprehensive
       python conformal_mof_anomaly_screening_pipeline.py --n_jobs 2 --model_profile fast --fast_test_rows 10000
   or in PowerShell:
       $env:MOF_N_JOBS="4"; python conformal_mof_anomaly_screening_pipeline.py
6. To change the default run size, edit the USER CONFIGURATION section below.

RESTART-SAFE DESIGN
-------------------
Every major stage writes a manifest file under:
       results/manifests/
If a stage is complete and its expected outputs exist, the script skips it unless
FORCE_RERUN = True. If the script is interrupted, run it again and it will resume
from the latest completed stage.

IMPORTANT SCIENTIFIC NOTE
-------------------------
Conformal guarantees are split-specific and distributional. Random-split coverage
is easier than topology/chemistry-grouped coverage. The grouped split results are
therefore essential for the publishability of the paper, because they show how
risk control behaves under more realistic extrapolation.

Version: 5.1 Paper-B v5 high-impact QC/casebook pipeline with zero diagnostics, feature-family redundancy checks, sensitivity analysis, casebook and fixed zero-count figures
Author: generated for the ARC-MOF conformal anomaly screening project.
"""

# =============================================================================
# Standard library imports
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import errno
import shutil
import multiprocessing
import pickle
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# =============================================================================
# Third-party imports
# =============================================================================

import joblib
import numpy as np
import pandas as pd

# Matplotlib Agg backend makes the script safe in headless terminals and VS Code.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import patheffects as pe

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# USER CONFIGURATION
# =============================================================================



def default_project_root() -> Path:
    r"""Return the project/code root used for outputs and helper files.

    In this patched version, the default project/output root is the folder that
    contains this Python file.  The data archive can still live elsewhere and is
    selected independently with --data_root.  This avoids the confusing behaviour
    where results were written under E:\\projects\\our_group\\ai\\project_data
    even when the script was launched from the code folder.
    """
    env = os.environ.get("MOF_PROJECT_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent


def normalize_data_root(path: Path) -> Path:
    r"""Resolve either a project root or a raw-data root to the actual data folder.

    Accepted inputs:
    - E:\projects\our_group\ai\project_data
    - E:\projects\our_group\ai\project_data\data_raw
    - a folder that directly contains arc_mof/core_mof_2024/mosaec_db
    - the old ...\\data or ...\\data\raw style used by earlier versions
    """
    p = Path(path)
    # If the user passed the project root, descend to data_raw when present.
    if (p / "data_raw" / "arc_mof").exists() or (p / "data_raw" / "core_mof_2024").exists():
        return p / "data_raw"
    # If the user passed the new raw-data root, keep it.
    if (p / "arc_mof").exists() or (p / "core_mof_2024").exists() or (p / "mosaec_db").exists():
        return p
    # If the user passed old project root, descend to data or data/raw when useful.
    if (p / "data" / "raw" / "arc_mof").exists():
        return p / "data" / "raw"
    if (p / "data" / "arc_mof").exists() or (p / "data" / "raw").exists():
        return p / "data"
    if (p / "raw" / "arc_mof").exists():
        return p / "raw"
    return p


def infer_project_root_from_data_root(data_root: Path) -> Path:
    """Infer project root from a data_raw/data/raw folder for results placement."""
    p = Path(data_root)
    if p.name.lower() in {"data_raw", "data", "raw"}:
        return p.parent
    return p


def default_data_root() -> Path:
    """Return the folder that contains arc_mof/core_mof_2024/mosaec_db."""
    env = os.environ.get("MOF_DATA_ROOT", "").strip()
    if env:
        return normalize_data_root(Path(env))
    project_root = default_project_root()
    candidates = [
        project_root,
        project_root / "data_raw",
        project_root / "data",
        project_root / "data" / "raw",
        Path(r"E:\projects\our_group\ai\project_data\data_raw"),
        Path(r"C:\Projects\Project_Data\arc_core_mos_ver1\data"),
        Path(__file__).resolve().parent / "data_raw",
        Path(__file__).resolve().parent / "data",
    ]
    for cand in candidates:
        resolved = normalize_data_root(cand)
        if resolved.exists() and ((resolved / "arc_mof").exists() or (resolved / "raw" / "arc_mof").exists()):
            return resolved
    return normalize_data_root(project_root)

def default_results_root() -> Path:
    """Return the parent folder for result directories.

    Default: the folder containing this .py file.  Override with --results_root
    if you intentionally want results beside the data archive or on another disk.
    """
    env = os.environ.get("MOF_RESULTS_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent

@dataclass
class Config:
    """All user-facing controls for the pipeline.

    Edit these values rather than changing analysis functions below. The defaults
    are chosen to be CPU-friendly. For a final paper run, increase N_SEEDS and
    possibly N_BOOTSTRAP if runtime permits.
    """

    # Main project paths.  The defaults match the user's local data layout:
    #     C:\Projects\Project_Data\arc_core_mos_ver1\data
    # The script can still be moved anywhere and controlled with either command
    # line flags (--data_root, --results_root) or environment variables
    # (MOF_DATA_ROOT, MOF_PROJECT_ROOT, MOF_RESULTS_ROOT).
    PROJECT_ROOT: Path = field(default_factory=default_project_root)
    DATA_ROOT: Path = field(default_factory=default_data_root)
    RESULTS_ROOT: Path = field(default_factory=default_results_root)

    # Relative to DATA_ROOT.  The pipeline searches recursively inside ARC-MOF,
    # so versioned folders such as raw/arc_mof/v7_zenodo_16802743 are supported.
    RAW_ARC_MOF_SUBDIR: str = "arc_mof"
    RAW_ARCMOF_LEGACY_SUBDIR: str = "arcmof"

    # Optional Paper-B enhancement folders relative to DATA_ROOT. These are not
    # mixed into the ARC-MOF modelling table automatically. They are used after
    # conformal screening to annotate candidate case studies with experimental
    # provenance, structural plausibility, activation/stability metadata and
    # process-relevant proxies.
    CORE_MOF_SUBDIR: str = "core_mof_2024"
    CORE_MOF_2025_SUBDIR: str = "core_mof_2025"
    MOSAEC_SUBDIR: str = "mosaec_db"
    MOFCHECKER_SUBDIR: str = "mofchecker"
    STRUCTURES_SUBDIR: str = "arc_mof/structures"

    RESULTS_DIRNAME: str = "results_paperB_v4_qc_modes"

    # User-facing run-control modes requested for the final Paper-B code.
    # SAVE_MODE controls hard-disk usage, RAM_MODE controls memory pressure, and
    # COMPREHENSIVENESS controls scientific breadth/runtime.
    SAVE_MODE: str = "compact"
    RAM_MODE: str = "balanced"
    COMPREHENSIVENESS: str = "medium"

    # Convenience run modes.  --audit_only verifies file discovery and target
    # mapping without running ML.  --figures_only regenerates figures from an
    # existing results directory.
    AUDIT_ONLY: bool = False
    FIGURES_ONLY: bool = False

    # Disk/output safety. Full ARC-MOF runs can easily create tens of GB of
    # prediction CSVs and Random Forest joblib files. The defaults below keep
    # the run publication-useful but much safer for a normal workstation.
    MIN_FREE_DISK_GB: float = 5.0
    STOP_ON_LOW_DISK: bool = True
    DISK_CHECK_EVERY_N_JOBS: int = 1
    LOG_MAX_BYTES: int = 2_000_000
    LOG_BACKUP_COUNT: int = 3

    # Prediction-output policy.
    # full    = save every prediction/context column; largest and usually unnecessary.
    # compact = save all columns needed for aggregation, figures and case-study tables.
    # minimal = save only essential ranking/calibration columns; use when disk is tight.
    PREDICTION_OUTPUT_MODE: str = "compact"
    COMPRESS_PREDICTION_CSVS: bool = True
    SAVE_MODEL_OBJECTS: bool = False
    SAVE_ADAPTIVE_BIN_TABLES: bool = False
    SAVE_MERGED_ENGINEERED_CSV: bool = False
    MAX_CONTEXT_COLUMNS_IN_PREDICTIONS: int = 12
    METRICS_FLUSH_EVERY_N_JOBS: int = 5

    # Optional job-space reduction if disk/runtime are constrained. Keep all main
    # paper families by default, but exclude process_enhanced_optional unless
    # explicitly requested because it multiplies output size and is not always
    # central to the story.
    RUN_PROCESS_ENHANCED_OPTIONAL: bool = False
    DROP_REDUNDANT_FEATURE_FAMILIES: bool = True
    WRITE_ZERO_QC_REPORTS: bool = True
    WRITE_FILTER_SENSITIVITY_REPORTS: bool = True
    WRITE_SPLIT_DIAGNOSTIC_REPORTS: bool = True
    MAKE_FIXED_ZERO_COUNT_FIGURE: bool = True

    # Paper-B impact additions from the attached recommendations:
    # structural plausibility overlays, candidate-level casebooks, process/stability
    # triage after certification, and mechanistic positive-surprise labels.
    ENABLE_PAPER_B_IMPACT_ADDITIONS: bool = True
    ENABLE_EXTERNAL_PLAUSIBILITY_OVERLAY: bool = True
    ENABLE_PROCESS_STABILITY_TRIAGE: bool = True
    PAPER_B_CASES_PER_CLASS: int = 4
    PAPER_B_MAX_CASEBOOK_ROWS: int = 60

    # Conservative process/stability proxy thresholds. These are intentionally
    # simple and transparent; they are post-certification filters, not extra
    # training labels. Edit them if the target application requires different
    # pore or density windows.
    PROCESS_PROXY_MIN_DENSITY: float = 0.20
    PROCESS_PROXY_MAX_DENSITY: float = 1.60
    PROCESS_PROXY_MIN_VOID_FRACTION: float = 0.05
    PROCESS_PROXY_MAX_VOID_FRACTION: float = 0.90
    PROCESS_PROXY_MIN_PLD_A: float = 2.4
    PROCESS_PROXY_MAX_PLD_A: float = 14.0

    # Restart control. Set FORCE_RERUN=True only when you want to recompute all
    # outputs. Otherwise completed stages are skipped.
    FORCE_RERUN: bool = False

    # CPU and reproducibility controls.
    # N_JOBS controls the number of processor cores used by scikit-learn models
    # that support parallel execution, especially RandomForestRegressor.
    # Recommended values:
    #   1  = safest/default; fully reproducible and laptop-friendly
    #   2  = good modest speed-up on a normal laptop
    #   4  = useful on a workstation, but more memory intensive
    #  -1  = let scikit-learn use all available cores for supported models
    #   0  = auto-detect all physical/logical cores and convert to that number
    # You can override this without editing the file using either:
    #   python conformal_mof_anomaly_screening_pipeline.py --n_jobs 4
    # or the environment variable MOF_N_JOBS=4.
    N_JOBS: int = 1
    RANDOM_SEED: int = 42

    # Runtime/accuracy profile for the ML models. The default "balanced" profile
    # is intended for a serious manuscript run without making Random Forest and
    # Gradient Boosting unnecessarily expensive. Use --model_profile comprehensive for a
    # heavier final robustness run, or --model_profile fast for a quick pilot.
    MODEL_PROFILE: str = "balanced"

    # Default number of repeated seeds. Three seeds are a reasonable balanced
    # default because each seed is crossed with targets, feature families, models,
    # and split types. For the final submitted paper, run with --model_profile comprehensive
    # or --n_seeds 5 if your machine can handle it.
    N_SEEDS: int = 3
    SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

    # For testing on a laptop, set MAX_ROWS_FOR_FAST_TEST to e.g. 10000. For the
    # final paper, keep it as None.
    MAX_ROWS_FOR_FAST_TEST: Optional[int] = None

    # Split proportions: train / calibration / test = 60 / 20 / 20 by default.
    TEST_SIZE: float = 0.20
    CALIBRATION_SIZE_OF_REMAINING: float = 0.25

    # Conformal levels. 0.90 is the main level; 0.80 and 0.95 give useful curves.
    COVERAGE_LEVELS: Tuple[float, ...] = (0.80, 0.90, 0.95)
    MAIN_COVERAGE: float = 0.90

    # Elite definitions. 0.99 = top 1%, 0.95 = top 5%, 0.90 = top 10%.
    ELITE_QUANTILES: Tuple[float, ...] = (0.90, 0.95, 0.99)
    MAIN_ELITE_Q: float = 0.95

    # Models and feature families. You can reduce these for rapid debugging.
    MODELS_TO_RUN: Tuple[str, ...] = ("ridge", "hgb", "rf")
    FEATURE_FAMILIES_TO_RUN: Tuple[str, ...] = (
        "geometry_only",
        "enriched_geometry",
        "topology_only",
        "geometry_topology",
        "geometry_topology_clusters",
    )
    SPLIT_TYPES_TO_RUN: Tuple[str, ...] = (
        "random",
        "topology_grouped",
        "geo_cluster_grouped",
        "chemistry_cluster_grouped",
    )

    # Main target used for main figures. The detector will map this semantic name
    # to the real column in your CSV.
    MAIN_TARGET_KEY: str = "co2_0p15"

    # Target-detection quality control. This is important for the present project:
    # an earlier run mapped CO2 0.15 bar to the CO2 0.015 bar column because
    # fuzzy regex matching treated "0.015" as containing "0.15". The revised
    # detector parses numeric pressure values and prevents duplicate target-column
    # assignment.
    STRICT_TARGET_QC: bool = True
    TARGET_COLUMN_OVERRIDES: Dict[str, str] = field(default_factory=dict)

    # Optional adaptive/Mondrian-style conformal intervals binned by predicted
    # uptake. Standard split-conformal intervals are constant-width within one
    # job, so ordinary LCB rankings are identical to predicted rankings. Adaptive
    # bins produce locally variable intervals and are more suitable for high-IF
    # framing around auditable screening decisions.
    ADD_ADAPTIVE_CONFORMAL: bool = True
    ADAPTIVE_CONFORMAL_N_BINS: int = 10
    ADAPTIVE_CONFORMAL_MIN_CAL_PER_BIN: int = 200

    # Structure-image support. If you render candidate CIFs manually, place PNG/JPG
    # files in one of these folders using either the display_id or a slugified
    # version of it as the filename. If images are absent, the script creates
    # clear placeholders and a structural_case_study_manifest.csv telling you
    # which structures should be rendered.
    STRUCTURE_IMAGE_SUBDIRS: Tuple[str, ...] = (
        "structure_images",
        "structures_rendered",
        "figures/structures",
        "data/structures_rendered",
    )
    N_STRUCTURAL_CASES_PER_CLASS: int = 4

    # Lightweight-but-publishable model sizes.
    # These defaults are deliberately not toy values: they are strong enough to
    # support manuscript-quality comparisons, but regularized enough to avoid very
    # long Random Forest runs on full ARC-MOF-scale tables.
    #
    # Balanced profile defaults:
    #   RF: 160 trees, finite depth, subsampled bootstrap trees
    #   HGB: 220 boosting iterations with early stopping and mild L2 regularization
    #
    # The --model_profile flag can change these settings automatically.
    RF_N_ESTIMATORS: int = 160
    RF_MAX_DEPTH: Optional[int] = 18
    RF_MIN_SAMPLES_LEAF: int = 3
    RF_MIN_SAMPLES_SPLIT: int = 6
    RF_MAX_FEATURES: Any = "sqrt"
    RF_BOOTSTRAP: bool = True
    RF_MAX_SAMPLES: Optional[float] = 0.75

    HGB_MAX_ITER: int = 220
    HGB_LEARNING_RATE: float = 0.06
    HGB_MAX_LEAF_NODES: int = 31
    HGB_MIN_SAMPLES_LEAF: int = 25
    HGB_L2_REGULARIZATION: float = 0.05
    HGB_EARLY_STOPPING: Any = "auto"
    HGB_VALIDATION_FRACTION: float = 0.10
    HGB_N_ITER_NO_CHANGE: int = 20
    HGB_TOL: float = 1e-7

    # One-hot category cap. Very high-cardinality categorical columns are either
    # treated as group labels or ignored to avoid huge sparse matrices.
    MAX_CATEGORICAL_CARDINALITY: int = 128

    # Memory controls for aggregation. The model-fitting stage writes many large
    # prediction CSVs. Aggregation is therefore deliberately file-by-file rather
    # than concatenating all predictions into one giant dataframe. Keep this True
    # for full ARC-MOF-scale runs.
    MEMORY_SAFE_AGGREGATION: bool = True
    ENRICHMENT_USE_REPRESENTATIVE_JOB: bool = True

    # Bootstrap controls for confidence intervals in tables.
    N_BOOTSTRAP: int = 300
    BOOTSTRAP_ALPHA: float = 0.05

    # Figure controls.
    FIG_DPI: int = 150
    SAVEFIG_DPI: int = 300
    FIG_FORMATS: Tuple[str, ...] = ("png", "pdf", "svg")

    # File names expected from ARC-MOF-style inputs. The script tolerates missing
    # optional files and logs what was available.
    EXPECTED_FILES: Tuple[str, ...] = (
        "clean_data.csv",
        "geometric_properties.csv",
        "all_topology_lists.csv",
        "geo-clusters.csv",
        "mc-clusters.csv",
        "func-clusters.csv",
        "flig-clusters.csv",
        "overall_process.csv",
        "post_comb_vsa-CO2.csv",
        "post_comb_vsa-N2.csv",
        "methane.csv",
    )

    # Exact optional downloads used to cover the Paper-B suggestions. The script
    # tolerates missing optional files and records what was found.
    EXPECTED_CORE_MOF_FILES: Tuple[str, ...] = (
        "ASR_data_SI_20250204.csv",
        "FSR_data_SI_20250204.csv",
        "ION_data_SI_20250204.csv",
        "12089-recommended-screening-list.csv",
        "ASR_FSR_check.csv",
        "water.zip",
        "TSA.zip",
        "mofid-v2.zip",
        "CoREMOF2024DB_SI_20250204.zip",
    )
    EXPECTED_MOSAEC_FILES: Tuple[str, ...] = (
        "mosaec-db.csv",
        "mosaec-db.xlsx",
        "GEOM_mosaec-db.csv",
        "RAC_mosaec-db.csv",
        "APRDF_mosaec-db.csv",
        "PHOM_mosaec-db.csv",
    )
    EXPECTED_MOFCHECKER_FILES: Tuple[str, ...] = (
        "mofchecker_results.csv",
    )


CFG = Config()

# =============================================================================
# Save/RAM/comprehensiveness modes
# =============================================================================

SAVE_MODES: Dict[str, Dict[str, Any]] = {
    # Smallest audit/smoke outputs. Keeps metrics/tables and compressed minimal predictions.
    "ultra_minimal": dict(prediction_output_mode="minimal", compress_predictions=True, save_models=False, save_adaptive_bins=False, save_merged_csv=False, fig_formats=("png",), max_context_cols=4, min_free_disk_gb=2.0, casebook_rows=25),
    # Good for weak disks: enough to reproduce figures/tables, no model objects.
    "minimal": dict(prediction_output_mode="minimal", compress_predictions=True, save_models=False, save_adaptive_bins=False, save_merged_csv=False, fig_formats=("png",), max_context_cols=8, min_free_disk_gb=3.0, casebook_rows=40),
    # Recommended default: compact predictions plus figure source data and PNG/PDF figures.
    "compact": dict(prediction_output_mode="compact", compress_predictions=True, save_models=False, save_adaptive_bins=False, save_merged_csv=False, fig_formats=("png", "pdf"), max_context_cols=14, min_free_disk_gb=5.0, casebook_rows=80),
    # Submission-friendly: more context, adaptive bins, SVG/PDF/PNG figures.
    "standard": dict(prediction_output_mode="compact", compress_predictions=True, save_models=False, save_adaptive_bins=True, save_merged_csv=False, fig_formats=("png", "pdf", "svg"), max_context_cols=24, min_free_disk_gb=8.0, casebook_rows=120),
    # Archival/full reproducibility: full predictions and model objects. Can be very large.
    "complete": dict(prediction_output_mode="full", compress_predictions=True, save_models=True, save_adaptive_bins=True, save_merged_csv=True, fig_formats=("png", "pdf", "svg"), max_context_cols=64, min_free_disk_gb=15.0, casebook_rows=250),
}

RAM_MODES: Dict[str, Dict[str, Any]] = {
    # Ultra-light is intended for laptops/shared machines. It keeps models shallower.
    "ultra_light": dict(max_context_cols=4, rf_estimators=60, rf_depth=12, rf_max_samples=0.55, hgb_iter=100, hgb_leaf=45, bootstrap=80, adaptive_min_cal=400),
    "light": dict(max_context_cols=8, rf_estimators=100, rf_depth=15, rf_max_samples=0.65, hgb_iter=160, hgb_leaf=35, bootstrap=150, adaptive_min_cal=300),
    "balanced": dict(max_context_cols=14, rf_estimators=160, rf_depth=18, rf_max_samples=0.75, hgb_iter=220, hgb_leaf=25, bootstrap=300, adaptive_min_cal=200),
    "normal": dict(max_context_cols=24, rf_estimators=240, rf_depth=24, rf_max_samples=0.85, hgb_iter=320, hgb_leaf=20, bootstrap=500, adaptive_min_cal=150),
}

COMPREHENSIVENESS_MODES: Dict[str, Dict[str, Any]] = {
    # very_low = correctness/debug smoke; low = useful screening; medium = main paper; high = robustness.
    "very_low": dict(model_profile="fast", n_seeds=1, models=("ridge",), families=("geometry_only", "geometry_topology_clusters"), splits=("random",), coverage=(0.90,), elite=(0.95,), run_process=False, max_rows_default=10000),
    "low": dict(model_profile="fast", n_seeds=2, models=("ridge", "hgb", "rf"), families=("geometry_only", "enriched_geometry", "geometry_topology_clusters"), splits=("random", "topology_grouped"), coverage=(0.80, 0.90), elite=(0.90, 0.95), run_process=False, max_rows_default=None),
    "medium": dict(model_profile="balanced", n_seeds=3, models=("ridge", "hgb", "rf"), families=("geometry_only", "enriched_geometry", "topology_only", "geometry_topology", "geometry_topology_clusters"), splits=("random", "topology_grouped", "geo_cluster_grouped", "chemistry_cluster_grouped"), coverage=(0.80, 0.90, 0.95), elite=(0.90, 0.95, 0.99), run_process=False, max_rows_default=None),
    "high": dict(model_profile="comprehensive", n_seeds=5, models=("ridge", "hgb", "rf"), families=("geometry_only", "enriched_geometry", "topology_only", "geometry_topology", "geometry_topology_clusters", "process_enhanced_optional"), splits=("random", "topology_grouped", "geo_cluster_grouped", "chemistry_cluster_grouped"), coverage=(0.80, 0.90, 0.95), elite=(0.90, 0.95, 0.99), run_process=True, max_rows_default=None),
}

COMPREHENSIVENESS_ALIASES: Dict[str, str] = {
    "debug": "very_low", "smoke": "very_low", "verylow": "very_low",
    "screening": "low", "quick": "low",
    "paper": "medium", "balanced": "medium", "main": "medium",
    "exhaustive": "high", "very_high": "high", "veryhigh": "high", "complete": "high",
}


def apply_save_mode(cfg: Config) -> Config:
    mode = str(getattr(cfg, "SAVE_MODE", "compact") or "compact").lower().strip()
    if mode not in SAVE_MODES:
        raise ValueError(f"Unknown --save_mode {mode!r}. Use one of: {', '.join(SAVE_MODES)}")
    settings = SAVE_MODES[mode]
    cfg.SAVE_MODE = mode
    cfg.PREDICTION_OUTPUT_MODE = settings["prediction_output_mode"]
    cfg.COMPRESS_PREDICTION_CSVS = bool(settings["compress_predictions"])
    cfg.SAVE_MODEL_OBJECTS = bool(settings["save_models"])
    cfg.SAVE_ADAPTIVE_BIN_TABLES = bool(settings["save_adaptive_bins"])
    cfg.SAVE_MERGED_ENGINEERED_CSV = bool(settings["save_merged_csv"])
    cfg.FIG_FORMATS = tuple(settings["fig_formats"])
    cfg.MAX_CONTEXT_COLUMNS_IN_PREDICTIONS = int(settings["max_context_cols"])
    cfg.MIN_FREE_DISK_GB = float(settings["min_free_disk_gb"])
    cfg.PAPER_B_MAX_CASEBOOK_ROWS = int(settings["casebook_rows"])
    return cfg


def apply_ram_mode(cfg: Config) -> Config:
    mode = str(getattr(cfg, "RAM_MODE", "balanced") or "balanced").lower().strip().replace("-", "_")
    if mode not in RAM_MODES:
        raise ValueError(f"Unknown --ram_mode {mode!r}. Use one of: {', '.join(RAM_MODES)}")
    s = RAM_MODES[mode]
    cfg.RAM_MODE = mode
    cfg.MAX_CONTEXT_COLUMNS_IN_PREDICTIONS = min(int(getattr(cfg, "MAX_CONTEXT_COLUMNS_IN_PREDICTIONS", 12)), int(s["max_context_cols"])) if cfg.SAVE_MODE in {"ultra_minimal", "minimal"} else max(int(getattr(cfg, "MAX_CONTEXT_COLUMNS_IN_PREDICTIONS", 12)), int(s["max_context_cols"]))
    cfg.RF_N_ESTIMATORS = int(s["rf_estimators"])
    cfg.RF_MAX_DEPTH = int(s["rf_depth"])
    cfg.RF_MAX_SAMPLES = float(s["rf_max_samples"])
    cfg.HGB_MAX_ITER = int(s["hgb_iter"])
    cfg.HGB_MIN_SAMPLES_LEAF = int(s["hgb_leaf"])
    cfg.N_BOOTSTRAP = int(s["bootstrap"])
    cfg.ADAPTIVE_CONFORMAL_MIN_CAL_PER_BIN = int(s["adaptive_min_cal"])
    return cfg


def apply_comprehensiveness_mode(cfg: Config, cli_fast_test_rows_was_set: bool = False) -> Config:
    raw = str(getattr(cfg, "COMPREHENSIVENESS", "medium") or "medium").lower().strip().replace("-", "_")
    mode = COMPREHENSIVENESS_ALIASES.get(raw, raw)
    if mode not in COMPREHENSIVENESS_MODES:
        raise ValueError(f"Unknown --comprehensiveness {raw!r}. Use one of: {', '.join(COMPREHENSIVENESS_MODES)}")
    s = COMPREHENSIVENESS_MODES[mode]
    cfg.COMPREHENSIVENESS = mode
    cfg.MODEL_PROFILE = s["model_profile"]
    cfg.N_SEEDS = int(s["n_seeds"])
    cfg.MODELS_TO_RUN = tuple(s["models"])
    cfg.FEATURE_FAMILIES_TO_RUN = tuple(s["families"])
    cfg.SPLIT_TYPES_TO_RUN = tuple(s["splits"])
    cfg.COVERAGE_LEVELS = tuple(s["coverage"])
    cfg.MAIN_COVERAGE = 0.90 if 0.90 in cfg.COVERAGE_LEVELS else cfg.COVERAGE_LEVELS[-1]
    cfg.ELITE_QUANTILES = tuple(s["elite"])
    cfg.MAIN_ELITE_Q = 0.95 if 0.95 in cfg.ELITE_QUANTILES else cfg.ELITE_QUANTILES[-1]
    cfg.RUN_PROCESS_ENHANCED_OPTIONAL = bool(s["run_process"])
    if (not cli_fast_test_rows_was_set) and s.get("max_rows_default") is not None:
        cfg.MAX_ROWS_FOR_FAST_TEST = int(s["max_rows_default"])
    cfg = apply_model_profile(cfg)
    # apply_model_profile may modify N_SEEDS; enforce the comprehensiveness level after it.
    cfg.N_SEEDS = min(max(1, int(s["n_seeds"])), len(cfg.SEEDS))
    return cfg


def _resolve_n_jobs(value: Optional[int]) -> int:
    """Resolve user-facing n_jobs values into a safe scikit-learn value.

    Interpretation:
    - None: keep the value already stored in Config.
    - 0: use all detected CPU cores as a positive integer.
    - -1: pass through to scikit-learn, meaning all cores for estimators that
      support n_jobs.
    - positive integer: use exactly that many parallel workers where supported.
    """
    if value is None:
        return CFG.N_JOBS
    try:
        n = int(value)
    except Exception:
        return CFG.N_JOBS
    if n == 0:
        return max(1, multiprocessing.cpu_count() or 1)
    if n < -1:
        # Avoid surprising joblib semantics such as -2 = all but one. Keep this
        # script simple and explicit for students.
        return -1
    return n


def apply_model_profile(cfg: Config) -> Config:
    """Apply coherent lightweight/paper model settings.

    The aim is to make model size a deliberate scientific choice rather than an
    accidental runtime burden. All profiles keep Ridge, HistGradientBoosting and
    RandomForest in the comparison so that the paper retains transparent,
    fast-nonlinear and ensemble baselines.

    Profiles
    --------
    fast:
        Pilot/debugging profile. Useful for checking that files, target detection,
        splitting, conformal intervals and figures all work. Not recommended as
        the final manuscript result.

    balanced:
        Default. Recommended first full run. It is regularized and substantially
        lighter than an unrestricted RF, but still strong enough for a quality
        paper because it uses repeated seeds, multiple split types, nonlinear HGB
        and RF baselines, and conformal evaluation metrics.

    comprehensive:
        Heavier final robustness profile. Use this if the balanced run looks
        good and you want a more conservative final manuscript/SI rerun. The old
        value "paper" is accepted as a deprecated alias for backwards
        compatibility.
    """
    profile = str(getattr(cfg, "MODEL_PROFILE", "balanced") or "balanced").lower().strip()
    if profile == "paper":
        # Backwards compatibility for older commands. The public-facing name is
        # now "comprehensive" because it more clearly describes a heavier, more
        # complete robustness run rather than a different scientific standard.
        profile = "comprehensive"
    if profile not in {"fast", "balanced", "comprehensive"}:
        profile = "balanced"
    cfg.MODEL_PROFILE = profile

    if profile == "fast":
        cfg.N_SEEDS = min(cfg.N_SEEDS, 2)
        cfg.RF_N_ESTIMATORS = 80
        cfg.RF_MAX_DEPTH = 14
        cfg.RF_MIN_SAMPLES_LEAF = 5
        cfg.RF_MIN_SAMPLES_SPLIT = 10
        cfg.RF_MAX_FEATURES = "sqrt"
        cfg.RF_BOOTSTRAP = True
        cfg.RF_MAX_SAMPLES = 0.60
        cfg.HGB_MAX_ITER = 120
        cfg.HGB_LEARNING_RATE = 0.08
        cfg.HGB_MAX_LEAF_NODES = 25
        cfg.HGB_MIN_SAMPLES_LEAF = 35
        cfg.HGB_L2_REGULARIZATION = 0.10
        cfg.HGB_N_ITER_NO_CHANGE = 15
        return cfg

    if profile == "comprehensive":
        cfg.N_SEEDS = max(cfg.N_SEEDS, 5)
        cfg.RF_N_ESTIMATORS = 240
        cfg.RF_MAX_DEPTH = 24
        cfg.RF_MIN_SAMPLES_LEAF = 2
        cfg.RF_MIN_SAMPLES_SPLIT = 4
        cfg.RF_MAX_FEATURES = "sqrt"
        cfg.RF_BOOTSTRAP = True
        cfg.RF_MAX_SAMPLES = 0.85
        cfg.HGB_MAX_ITER = 320
        cfg.HGB_LEARNING_RATE = 0.045
        cfg.HGB_MAX_LEAF_NODES = 31
        cfg.HGB_MIN_SAMPLES_LEAF = 20
        cfg.HGB_L2_REGULARIZATION = 0.03
        cfg.HGB_N_ITER_NO_CHANGE = 25
        return cfg

    # balanced default
    cfg.N_SEEDS = min(max(cfg.N_SEEDS, 3), 5)
    cfg.RF_N_ESTIMATORS = 160
    cfg.RF_MAX_DEPTH = 18
    cfg.RF_MIN_SAMPLES_LEAF = 3
    cfg.RF_MIN_SAMPLES_SPLIT = 6
    cfg.RF_MAX_FEATURES = "sqrt"
    cfg.RF_BOOTSTRAP = True
    cfg.RF_MAX_SAMPLES = 0.75
    cfg.HGB_MAX_ITER = 220
    cfg.HGB_LEARNING_RATE = 0.06
    cfg.HGB_MAX_LEAF_NODES = 31
    cfg.HGB_MIN_SAMPLES_LEAF = 25
    cfg.HGB_L2_REGULARIZATION = 0.05
    cfg.HGB_N_ITER_NO_CHANGE = 20
    return cfg


def apply_runtime_overrides(cfg: Config) -> Config:
    """Allow common run controls to be changed without editing the script.

    Examples:
        python conformal_mof_anomaly_screening_pipeline.py --n_jobs 4
        python conformal_mof_anomaly_screening_pipeline.py --n_jobs -1
        python conformal_mof_anomaly_screening_pipeline.py --force_rerun
        python conformal_mof_anomaly_screening_pipeline.py --fast_test_rows 10000

    Environment-variable alternative:
        MOF_N_JOBS=4

    Notes:
    - N_JOBS currently affects scikit-learn estimators that expose n_jobs, most
      importantly RandomForestRegressor. Ridge and HistGradientBoostingRegressor
      do not expose n_jobs in the same way.
    - The outer manuscript job loop remains serial by design. That avoids race
      conditions in restart manifests and keeps memory usage predictable.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--save_mode", choices=tuple(SAVE_MODES.keys()), default=None, help="Hard-disk output mode: ultra_minimal, minimal, compact, standard, complete.")
    parser.add_argument("--ram_mode", choices=tuple(RAM_MODES.keys()), default=None, help="RAM mode: ultra_light, light, balanced, normal.")
    parser.add_argument("--comprehensiveness", choices=tuple(COMPREHENSIVENESS_MODES.keys()) + tuple(COMPREHENSIVENESS_ALIASES.keys()), default=None, help="Scientific breadth/runtime: very_low, low, medium, high. Old aliases smoke/screening/paper/exhaustive also work.")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help=(
            "Number of processor cores for parallel scikit-learn estimators. "
            "Use 1 for safest default, 2-4 for laptops/workstations, -1 for all "
            "cores, or 0 to auto-detect all cores as a positive integer."
        ),
    )
    parser.add_argument(
        "--model_profile",
        choices=("fast", "balanced", "comprehensive", "paper"),
        default=None,
        help=(
            "Model-size profile. 'fast' is for debugging, 'balanced' is the "
            "default paper-friendly runtime profile, and 'comprehensive' is a heavier "
            "final robustness profile. The old value 'paper' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore restart manifests and recompute all stages/jobs.",
    )
    parser.add_argument(
        "--fast_test_rows",
        type=int,
        default=None,
        help="Read only the first N rows of each CSV for debugging; omit for full run.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Number of seeds from Config.SEEDS to use.",
    )
    parser.add_argument(
        "--results_dirname",
        type=str,
        default=None,
        help="Name of the results folder. Use this to send outputs to a larger drive or a fresh folder.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Folder containing raw/processed/release_safe data. Default is "
            "C:\\Projects\\Project_Data\\arc_core_mos_ver1\\data on Windows, "
            "or MOF_DATA_ROOT if set."
        ),
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Parent folder for the results directory. Default is "
            "C:\\Projects\\Project_Data\\arc_core_mos_ver1 on Windows, "
            "or MOF_RESULTS_ROOT if set."
        ),
    )
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Stop after input discovery, CSV audit, merge, target detection and feature-family tables.",
    )
    parser.add_argument(
        "--figures_only",
        action="store_true",
        help="Regenerate aggregate tables, Paper-B casebook and figures from an existing completed metrics folder.",
    )
    parser.add_argument(
        "--min_free_disk_gb",
        type=float,
        default=None,
        help="Stop before/while writing outputs if free disk drops below this many GB. Default comes from Config.",
    )
    parser.add_argument(
        "--prediction_output_mode",
        choices=("minimal", "compact", "full"),
        default=None,
        help="How many prediction columns to save. Use compact/minimal to avoid filling the drive.",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Save trained model objects. Off by default because RF joblib files can fill the disk quickly.",
    )
    parser.add_argument(
        "--save_adaptive_bins",
        action="store_true",
        help="Save per-job adaptive conformal bin tables. Off by default to reduce file count.",
    )
    parser.add_argument(
        "--save_merged_engineered_csv",
        action="store_true",
        help="Also save the large merged engineered dataframe as CSV. Pickle is still saved for restart.",
    )
    parser.add_argument(
        "--run_process_enhanced_optional",
        action="store_true",
        help="Also run the process_enhanced_optional feature family. Off by default to reduce disk use.",
    )
    parser.add_argument(
        "--skip_paper_b_additions",
        action="store_true",
        help="Skip external plausibility overlays, process/stability triage and Paper-B candidate casebooks.",
    )
    parser.add_argument(
        "--paper_b_cases_per_class",
        type=int,
        default=None,
        help="Number of candidates per decision/anomaly class for the Paper-B casebook.",
    )
    parser.add_argument(
        "--no_compress_predictions",
        action="store_true",
        help="Save prediction files as .csv instead of .csv.gz. This uses much more disk space.",
    )

    # parse_known_args prevents failures if VS Code, notebooks or external launch
    # tools inject extra arguments.
    args, _unknown = parser.parse_known_args()

    if getattr(args, "save_mode", None) is not None:
        cfg.SAVE_MODE = args.save_mode
    if getattr(args, "ram_mode", None) is not None:
        cfg.RAM_MODE = args.ram_mode
    if getattr(args, "comprehensiveness", None) is not None:
        cfg.COMPREHENSIVENESS = args.comprehensiveness

    if args.model_profile is not None:
        cfg.MODEL_PROFILE = args.model_profile
    # Apply user-facing modes.  Comprehensiveness determines the default model
    # profile, then RAM and save modes tune resources and output size.  An
    # explicit --model_profile can still override the comprehensiveness default.
    cli_fast_test_rows_was_set = args.fast_test_rows is not None
    cfg = apply_comprehensiveness_mode(cfg, cli_fast_test_rows_was_set=cli_fast_test_rows_was_set)
    if args.model_profile is not None:
        cfg.MODEL_PROFILE = args.model_profile
        cfg = apply_model_profile(cfg)
    cfg = apply_ram_mode(cfg)
    cfg = apply_save_mode(cfg)

    env_n_jobs = os.environ.get("MOF_N_JOBS", None)
    if args.n_jobs is not None:
        cfg.N_JOBS = _resolve_n_jobs(args.n_jobs)
    elif env_n_jobs not in (None, ""):
        cfg.N_JOBS = _resolve_n_jobs(int(env_n_jobs))

    if args.force_rerun:
        cfg.FORCE_RERUN = True
    if args.fast_test_rows is not None:
        cfg.MAX_ROWS_FOR_FAST_TEST = max(1, int(args.fast_test_rows))
    if args.n_seeds is not None:
        cfg.N_SEEDS = max(1, min(int(args.n_seeds), len(cfg.SEEDS)))
    if args.results_dirname:
        cfg.RESULTS_DIRNAME = str(args.results_dirname)
    if getattr(args, "data_root", None):
        cfg.DATA_ROOT = normalize_data_root(Path(args.data_root))
        # Keep outputs beside the code by default.  Passing --data_root should not
        # silently move the results directory into the data archive.  Use
        # --results_root explicitly if you want a different output location.
    if getattr(args, "results_root", None):
        cfg.RESULTS_ROOT = Path(args.results_root)
        cfg.PROJECT_ROOT = Path(args.results_root)
    if getattr(args, "audit_only", False):
        cfg.AUDIT_ONLY = True
    if getattr(args, "figures_only", False):
        cfg.FIGURES_ONLY = True
    if args.min_free_disk_gb is not None:
        cfg.MIN_FREE_DISK_GB = max(0.1, float(args.min_free_disk_gb))
    if args.prediction_output_mode is not None:
        cfg.PREDICTION_OUTPUT_MODE = args.prediction_output_mode
    if args.save_models:
        cfg.SAVE_MODEL_OBJECTS = True
    if args.save_adaptive_bins:
        cfg.SAVE_ADAPTIVE_BIN_TABLES = True
    if args.save_merged_engineered_csv:
        cfg.SAVE_MERGED_ENGINEERED_CSV = True
    if args.run_process_enhanced_optional:
        cfg.RUN_PROCESS_ENHANCED_OPTIONAL = True
    if getattr(args, "skip_paper_b_additions", False):
        cfg.ENABLE_PAPER_B_IMPACT_ADDITIONS = False
    if getattr(args, "paper_b_cases_per_class", None) is not None:
        cfg.PAPER_B_CASES_PER_CLASS = max(1, int(args.paper_b_cases_per_class))
    if args.no_compress_predictions:
        cfg.COMPRESS_PREDICTION_CSVS = False

    # Make thread-heavy numerical libraries less likely to oversubscribe when RF
    # also runs in parallel. Users can still override these before launching.
    if cfg.N_JOBS not in (None, 1):
        thread_value = str(cfg.N_JOBS if cfg.N_JOBS > 0 else multiprocessing.cpu_count())
        os.environ.setdefault("OMP_NUM_THREADS", thread_value)
        os.environ.setdefault("MKL_NUM_THREADS", thread_value)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_value)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", thread_value)

    return cfg

CFG = apply_runtime_overrides(CFG)

# =============================================================================
# Global constants and target definitions
# =============================================================================

TARGET_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "co2_0p015": {
        "pretty": "CO$_2$ uptake at 0.015 bar",
        "short": "CO2 0.015 bar",
        "patterns": [
            r"co\s*2.*0[\._]?015", r"0[\._]?015.*co\s*2", r"co2.*15\s*mbar",
            r"uptake.*co\s*2.*0[\._]?015",
        ],
    },
    "co2_0p15": {
        "pretty": "CO$_2$ uptake at 0.15 bar",
        "short": "CO2 0.15 bar",
        "patterns": [
            r"co\s*2.*0[\._]?15", r"0[\._]?15.*co\s*2", r"co2.*150\s*mbar",
            r"uptake.*co\s*2.*0[\._]?15",
        ],
    },
    "ch4_5p8": {
        "pretty": "CH$_4$ uptake at 5.8 bar",
        "short": "CH4 5.8 bar",
        "patterns": [
            r"ch\s*4.*5[\._]?8", r"methane.*5[\._]?8", r"5[\._]?8.*methane", r"5[\._]?8.*ch\s*4",
        ],
    },
    "ch4_65": {
        "pretty": "CH$_4$ uptake at 65 bar",
        "short": "CH4 65 bar",
        "patterns": [
            r"ch\s*4.*65", r"methane.*65", r"65.*methane", r"65.*ch\s*4",
        ],
    },
}

CANONICAL_GEOMETRY_HINTS = [
    "density", "asa", "gasa", "vasa", "nasa", "gnasa", "vnasa",
    "ava", "avaf", "avag", "poava", "poavaf", "poavag",
    "nava", "navaf", "navag", "npoava", "npoavaf", "npoavag",
    "di", "df", "dif", "pld", "lcd", "uc_volume", "unitcell", "void",
    "pore", "surface", "volume", "diameter", "largest", "limiting",
]

TOPOLOGY_HINTS = ["topology", "topo", "net", "rcsr", "tbo", "sbu", "symbol"]
CLUSTER_HINTS = ["cluster", "geo_cluster", "mc_cluster", "func_cluster", "flig_cluster", "metal", "functional", "linker"]
ID_HINTS = ["filename", "name", "mof", "mofid", "id", "refcode", "cif", "structure"]

# =============================================================================
# Publication aesthetics
# =============================================================================
# A restrained, colorblind-safe palette inspired by Okabe-Ito plus MOF-friendly
# blues/teals. These colors are used only for figures; all numeric source data
# remain saved separately as CSV files. The goal is a Chemical Science / J. Mater.
# Chem. A style: clean white background, high contrast, minimal clutter, and
# chemically meaningful highlight colors for certified/anomalous materials.
JOURNAL_COLORS: Dict[str, str] = {
    "ink": "#222831",
    "muted_ink": "#5B6470",
    "grid": "#D6DEE6",
    "light_grid": "#EEF2F5",
    "blue": "#2F6B9A",
    "sky": "#78B7D8",
    "teal": "#2A9D8F",
    "green": "#4C956C",
    "gold": "#E9A53A",
    "orange": "#E76F51",
    "red": "#B23A48",
    "purple": "#6D5A9C",
    "grey": "#A6AFB8",
    "pale_blue": "#EAF4FA",
    "pale_teal": "#E8F5F2",
    "pale_orange": "#FFF1E8",
}
COLOR_CYCLE: Tuple[str, ...] = (
    JOURNAL_COLORS["blue"],
    JOURNAL_COLORS["orange"],
    JOURNAL_COLORS["teal"],
    JOURNAL_COLORS["purple"],
    JOURNAL_COLORS["gold"],
    JOURNAL_COLORS["green"],
    JOURNAL_COLORS["red"],
    JOURNAL_COLORS["grey"],
)
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "coverage_deviation",
    [JOURNAL_COLORS["blue"], "#FFFFFF", JOURNAL_COLORS["orange"]],
)
EFFICIENCY_CMAP = LinearSegmentedColormap.from_list(
    "efficiency_blues",
    ["#F7FBFF", JOURNAL_COLORS["sky"], JOURNAL_COLORS["blue"], JOURNAL_COLORS["ink"]],
)
ANOMALY_CMAP = LinearSegmentedColormap.from_list(
    "anomaly_residuals",
    [JOURNAL_COLORS["blue"], "#FFFFFF", JOURNAL_COLORS["red"]],
)


TARGET_PRESSURES_BAR: Dict[str, float] = {
    "co2_0p015": 0.015,
    "co2_0p15": 0.15,
    "ch4_5p8": 5.8,
    "ch4_65": 65.0,
}
TARGET_GASES: Dict[str, Tuple[str, ...]] = {
    "co2_0p015": ("co2", "co_2", "carbon_dioxide"),
    "co2_0p15": ("co2", "co_2", "carbon_dioxide"),
    "ch4_5p8": ("ch4", "ch_4", "methane"),
    "ch4_65": ("ch4", "ch_4", "methane"),
}


def _normalise_text_for_pressure(text: str) -> str:
    """Prepare column text for pressure parsing without confusing CO2/CH4 formulae.

    The old fuzzy detector could accidentally match the 0.15-bar target to a
    0.015-bar column because the substring "015" contains "15". For this paper,
    target identity is scientifically critical, so pressure values are parsed as
    numbers before matching.
    """
    s = str(text).lower()
    # Convert common decimal encodings only when they occur between digits.
    s = re.sub(r"(?<=\d)[_p](?=\d)", ".", s)
    return s


def _extract_pressure_values_bar(column_name: str) -> List[float]:
    """Extract plausible pressure values from a column name, returned in bar.

    Handles examples such as "0.015 bar", "0_15_bar", "5p8bar", and
    "150 mbar". Formula digits in CO2/CH4 are ignored because they are adjacent
    to letters and therefore do not satisfy the number boundary rules.
    """
    s = _normalise_text_for_pressure(column_name)
    values: List[float] = []

    # Prefer values explicitly followed by bar/mbar.
    for match in re.finditer(r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)\s*(mbar|bar)(?![a-z])", s):
        val = float(match.group(1))
        unit = match.group(2)
        values.append(val / 1000.0 if unit == "mbar" else val)

    # Also keep standalone decimal-looking numbers if no explicit unit was found.
    if not values:
        for token in re.findall(r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)(?![a-z0-9])", s):
            try:
                val = float(token)
            except Exception:
                continue
            # Avoid trivial formula values and table counters unless they are
            # known adsorption pressures used by this project.
            if any(abs(val - p) <= max(1e-6, 1e-3 * p) for p in TARGET_PRESSURES_BAR.values()):
                values.append(val)
    return values


def _column_has_gas(column_name: str, target_key: str) -> bool:
    slug = slugify(column_name)
    return any(g in slug for g in TARGET_GASES.get(target_key, ()))


def _target_candidate_score(column_name: str, series: pd.Series, target_key: str) -> Tuple[float, str]:
    """Score how well a column matches one semantic adsorption target.

    Returns (score, reason). The scoring is deliberately conservative: correct
    gas + correct numeric pressure dominates regex substring matches. This avoids
    CO2 0.015/0.15 confusion and makes the target table citable in the SI.
    """
    target_p = TARGET_PRESSURES_BAR[target_key]
    slug = slugify(column_name)
    score = 0.0
    reasons: List[str] = []

    if _column_has_gas(column_name, target_key):
        score += 20
        reasons.append("gas")
    else:
        return -1e9, "wrong_gas"

    pressures = _extract_pressure_values_bar(column_name)
    if pressures:
        best_diff = min(abs(pv - target_p) for pv in pressures)
        if best_diff <= max(1e-6, target_p * 1e-3):
            score += 100
            reasons.append(f"pressure={target_p:g}bar")
        else:
            # Strongly penalise columns for the same gas at a different pressure.
            score -= 50 + 10 * min(best_diff / max(target_p, 1e-9), 10)
            reasons.append("different_pressure")
    else:
        # Regex fallback only after numeric pressure parsing. It is intentionally
        # weak and cannot override a correct numeric-pressure match elsewhere.
        info = TARGET_DEFINITIONS[target_key]
        if any(re.search(pat, slug, flags=re.IGNORECASE) for pat in info["patterns"]):
            score += 5
            reasons.append("regex_fallback")

    if pd.api.types.is_numeric_dtype(series):
        score += 5
        reasons.append("numeric")
    nonmissing = int(series.notna().sum())
    score += min(nonmissing / max(len(series), 1), 1.0)
    return score, ";".join(reasons)


# =============================================================================
# Path helpers, logging, and restart manifests
# =============================================================================

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(text: str, max_len: int = 120) -> str:
    """Return a filesystem-safe string."""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_paths(cfg: Config) -> Dict[str, Path]:
    r"""Create all output/input path aliases.

    This version is archive-aware.  CFG.DATA_ROOT is normalised so that both
    `--data_root E:\projects\our_group\ai\project_data` and
    `--data_root E:\projects\our_group\ai\project_data\data_raw` work.
    """
    data_root = normalize_data_root(Path(cfg.DATA_ROOT))
    cfg.DATA_ROOT = data_root
    root = Path(getattr(cfg, "PROJECT_ROOT", infer_project_root_from_data_root(data_root)))
    if not root.exists() and data_root.exists():
        root = infer_project_root_from_data_root(data_root)
    cfg.PROJECT_ROOT = root
    results_root = Path(cfg.RESULTS_ROOT) if getattr(cfg, "RESULTS_ROOT", None) else root
    results = ensure_dir(results_root / cfg.RESULTS_DIRNAME)
    paths = {
        "root": root,
        "data_root": data_root,
        "results_root": results_root,
        "raw_same_folder": root,
        "raw_subdir": data_root / cfg.RAW_ARC_MOF_SUBDIR,
        "raw_arc_mof": data_root / cfg.RAW_ARC_MOF_SUBDIR,
        "raw_arcmof_legacy": data_root / cfg.RAW_ARCMOF_LEGACY_SUBDIR,
        "raw_core_mof": data_root / cfg.CORE_MOF_SUBDIR,
        "raw_core_mof_2025": data_root / getattr(cfg, "CORE_MOF_2025_SUBDIR", "core_mof_2025"),
        "raw_mosaec": data_root / cfg.MOSAEC_SUBDIR,
        "raw_mofchecker": data_root / cfg.MOFCHECKER_SUBDIR,
        "raw_structures": data_root / cfg.STRUCTURES_SUBDIR,
        "results": results,
        "logs": ensure_dir(results / "logs"),
        "audit": ensure_dir(results / "data_audit"),
        "processed": ensure_dir(results / "processed_data"),
        "models": ensure_dir(results / "models"),
        "predictions": ensure_dir(results / "predictions"),
        "metrics": ensure_dir(results / "metrics"),
        "tables": ensure_dir(results / "tables"),
        "tables_main": ensure_dir(results / "tables" / "main"),
        "tables_si": ensure_dir(results / "tables" / "si"),
        "figures": ensure_dir(results / "figures"),
        "figures_main": ensure_dir(results / "figures" / "main"),
        "figures_si": ensure_dir(results / "figures" / "si"),
        "figure_data": ensure_dir(results / "figure_data"),
        "figure_data_main": ensure_dir(results / "figure_data" / "main"),
        "figure_data_si": ensure_dir(results / "figure_data" / "si"),
        "qc": ensure_dir(results / "qc_reports"),
        "structural_cases": ensure_dir(results / "structural_case_studies"),
        "selected_cifs": ensure_dir(results / "structural_case_studies" / "selected_cifs"),
        "paper_b": ensure_dir(results / "paper_b_impact_additions"),
        "paper_b_tables": ensure_dir(results / "paper_b_impact_additions" / "tables"),
        "paper_b_figures": ensure_dir(results / "paper_b_impact_additions" / "figures"),
        "paper_b_figure_data": ensure_dir(results / "paper_b_impact_additions" / "figure_data"),
        "download_guides": ensure_dir(results / "download_guides"),
        "manifests": ensure_dir(results / "manifests"),
    }
    return paths

PATHS = make_paths(CFG)


class _DeduplicateLogFilter(logging.Filter):
    """Suppress identical noisy messages after a small number of repeats."""

    def __init__(self, max_repeats: int = 3) -> None:
        super().__init__()
        self.max_repeats = max_repeats
        self.counts: Dict[Tuple[str, str], int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelname, record.getMessage())
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        return n <= self.max_repeats


def setup_logging(paths: Dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger("conformal_mof")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logging.raiseExceptions = False

    log_file = paths["logs"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    dedup_filter = _DeduplicateLogFilter(max_repeats=3)

    try:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=int(getattr(CFG, "LOG_MAX_BYTES", 2_000_000)),
            backupCount=int(getattr(CFG, "LOG_BACKUP_COUNT", 3)),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        fh.addFilter(dedup_filter)
        logger.addHandler(fh)
    except OSError:
        # If the drive is already full, continue with console logging only.
        pass

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    sh.addFilter(dedup_filter)
    logger.addHandler(sh)

    logger.info("STAGE >>> JOB_START")
    logger.info("Project root: %s", CFG.PROJECT_ROOT)
    logger.info("Data root: %s", CFG.DATA_ROOT)
    logger.info("Results root: %s", CFG.RESULTS_ROOT)
    logger.info("ARC-MOF search root: %s", paths.get("raw_arc_mof"))
    logger.info("CoRE MOF overlay root: %s", paths.get("raw_core_mof"))
    logger.info("MOSAEC overlay root: %s", paths.get("raw_mosaec"))
    logger.info("Results directory: %s", paths["results"])
    logger.info("Log file: %s", log_file)
    logger.info("Configuration: %s", json.dumps({k: str(v) for k, v in asdict(CFG).items()}, indent=2))
    return logger


LOGGER = setup_logging(PATHS)


def manifest_path(stage_name: str) -> Path:
    return PATHS["manifests"] / f"{slugify(stage_name)}.done.json"


def is_stage_done(stage_name: str, expected_outputs: Optional[Sequence[Path]] = None) -> bool:
    """Return True when a stage manifest and all expected outputs exist."""
    if CFG.FORCE_RERUN:
        return False
    mp = manifest_path(stage_name)
    if not mp.exists():
        return False
    if expected_outputs:
        for p in expected_outputs:
            if not Path(p).exists():
                return False
    return True


def mark_stage_done(stage_name: str, outputs: Optional[Sequence[Path]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "stage": stage_name,
        "completed_at": now(),
        "outputs": [str(p) for p in (outputs or [])],
        "extra": extra or {},
    }
    with open(manifest_path(stage_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("STAGE_DONE | %s", stage_name)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def free_disk_gb(path: Path) -> float:
    """Return free disk space in GB for the drive containing path."""
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(str(probe))
    return usage.free / (1024 ** 3)


def check_disk_space(path: Path, context: str = "write", min_free_gb: Optional[float] = None) -> None:
    """Stop early with an actionable message before the drive fills completely."""
    threshold = CFG.MIN_FREE_DISK_GB if min_free_gb is None else float(min_free_gb)
    free = free_disk_gb(path)
    if getattr(CFG, "STOP_ON_LOW_DISK", True) and free < threshold:
        raise RuntimeError(
            f"LOW_DISK_SPACE before {context}: only {free:.2f} GB free at {path}. "
            f"Free disk space, move the results folder, or rerun with compact outputs. "
            f"Suggested command: python {Path(__file__).name} --prediction_output_mode minimal "
            f"--n_jobs 1 --results_dirname results_paperB_disk_safe --min_free_disk_gb {max(1.0, threshold):.1f}"
        )


def prediction_extension() -> str:
    return ".csv.gz" if getattr(CFG, "COMPRESS_PREDICTION_CSVS", True) else ".csv"


def prediction_path_for_job(job_id: str) -> Path:
    return PATHS["predictions"] / f"predictions__{job_id}{prediction_extension()}"


def find_existing_prediction_path(job_id: str) -> Optional[Path]:
    for ext in [prediction_extension(), ".csv.gz", ".csv"]:
        p = PATHS["predictions"] / f"predictions__{job_id}{ext}"
        if p.exists():
            return p
    return None


def compact_prediction_columns(pred: pd.DataFrame) -> List[str]:
    """Columns to keep in compact/minimal prediction files.

    This is the main disk-saving step. It keeps all columns needed by later
    aggregation and figure functions while dropping dozens of duplicated context
    columns that are not needed for manuscript-level analysis.
    """
    mode = str(getattr(CFG, "PREDICTION_OUTPUT_MODE", "compact")).lower()
    if mode == "full":
        return list(pred.columns)
    essential_prefixes = (
        "L_", "U_", "q_abs_", "q_pos_", "q_neg_",
        "positive_anomaly_", "negative_anomaly_", "two_sided_anomaly_",
        "true_elite_", "predicted_elite_", "confident_elite_", "possible_elite_",
    )
    essential = {
        "job_id", "target_key", "target_col", "feature_family", "model_name",
        "split_type", "seed", "display_id", "row_index", "y_true", "y_pred",
        "residual", "p_positive_surprise", "p_negative_surprise",
        "p_two_sided_surprise", "decision_class_main",
    }
    cols = [c for c in pred.columns if c in essential or c.startswith(essential_prefixes)]
    if mode == "minimal":
        return cols
    context_candidates = []
    for c in pred.columns:
        cslug = slugify(c)
        if c in cols:
            continue
        if any(h in cslug for h in TOPOLOGY_HINTS + CLUSTER_HINTS + ["density", "asa", "ava", "avaf", "pld", "lcd", "di", "df", "dif"]):
            context_candidates.append(c)
    return cols + context_candidates[: int(getattr(CFG, "MAX_CONTEXT_COLUMNS_IN_PREDICTIONS", 12))]


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    check_disk_space(path, context=f"writing {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        compression = "gzip" if str(path).lower().endswith(".gz") else None
        df.to_csv(tmp, index=index, compression=compression)
        os.replace(tmp, path)
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        if getattr(e, "errno", None) == errno.ENOSPC:
            raise RuntimeError(
                f"No space left while writing {path}. Delete old results/predictions/models, "
                f"or rerun with --prediction_output_mode minimal --results_dirname results_paperB_disk_safe."
            ) from e
        raise


def safe_to_pickle_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    check_disk_space(path, context=f"writing {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_pickle(tmp)
        os.replace(tmp, path)
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        if getattr(e, "errno", None) == errno.ENOSPC:
            raise RuntimeError(f"No space left while writing {path}.") from e
        raise


def _latex_escape_text(value: Any) -> str:
    """Escape a Python value for a simple LaTeX tabular cell.

    This fallback is intentionally dependency-free.  It is used only when
    pandas.DataFrame.to_latex is unavailable because the optional pandas
    dependency jinja2 is not installed.  It keeps the pipeline restart-safe even
    on minimal environments.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value)
    # Keep strings readable in narrow SI tables.
    if len(s) > 240:
        s = s[:237] + "..."
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _simple_latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Create a small LaTeX table without pandas Styler/Jinja2.

    Pandas >=2 can route DataFrame.to_latex through Styler, which requires the
    optional package jinja2.  This fallback deliberately avoids Styler so that
    audit, SI and preview-table stages do not stop an otherwise valid ML run.
    """
    data = df.copy()
    # Avoid very wide rows causing enormous .tex files.
    max_rows_for_tex = 200
    if len(data) > max_rows_for_tex:
        data = data.head(max_rows_for_tex).copy()
    ncols = max(1, data.shape[1])
    colspec = "l" * ncols
    lines: List[str] = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    if caption:
        lines.append(r"\caption{" + _latex_escape_text(caption) + r"}")
    if label:
        lines.append(r"\label{" + _latex_escape_text(label) + r"}")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    if ncols:
        lines.append(" & ".join(_latex_escape_text(c) for c in data.columns) + r" \\")
        lines.append(r"\hline")
        for _, row in data.iterrows():
            lines.append(" & ".join(_latex_escape_text(row[c]) for c in data.columns) + r" \\")
    else:
        lines.append(r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def save_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    """Save a LaTeX table, with a dependency-free fallback if Jinja2 is missing.

    The original version used pandas.DataFrame.to_latex directly.  In recent
    pandas versions that can require the optional jinja2 package.  The pipeline
    should not fail during data audit just because a preview .tex table cannot be
    rendered through pandas Styler, so this function falls back to a simple
    manually written tabular environment.
    """
    ensure_dir(path.parent)
    try:
        latex = df.to_latex(index=False, escape=False, longtable=False, caption=caption, label=label)
    except Exception as e:
        try:
            LOGGER.warning("PANDAS_TO_LATEX_FAILED_FALLING_BACK | %s | %s", path.name, e)
        except Exception:
            pass
        latex = _simple_latex_table(df, caption=caption, label=label)
    path.write_text(latex, encoding="utf-8")

# =============================================================================
# CSV discovery and data audit
# =============================================================================

def discover_csv_files(cfg: Config) -> List[Path]:
    """Find ARC-MOF CSV files in the current archive layout.

    The new archive has:
        DATA_ROOT/arc_mof/adsorption/*.csv
        DATA_ROOT/arc_mof/clusters/*.csv
        DATA_ROOT/arc_mof/descriptors/*.csv
        DATA_ROOT/arc_mof/process/*.csv
        DATA_ROOT/arc_mof/topology/*.csv

    The discovery deliberately avoids mixing CoRE/MOSAEC/QMOF into the modelling
    table.  Those databases are reserved for external plausibility overlays.
    """
    candidates: List[Path] = []
    arc_root = PATHS.get("raw_arc_mof")
    legacy_root = PATHS.get("raw_arcmof_legacy")
    possible_roots = [arc_root, legacy_root, Path(cfg.DATA_ROOT) / "raw" / "arc_mof"]
    seen_roots = set()
    for folder in possible_roots:
        if folder is None:
            continue
        folder = Path(folder)
        key = str(folder.resolve()) if folder.exists() else str(folder)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))
            candidates.extend(sorted(folder.rglob("*.csv")))
    # Backwards compatibility: if the script folder itself contains ARC-MOF CSVs.
    same_folder = PATHS.get("raw_same_folder")
    if same_folder and Path(same_folder).exists():
        for p in Path(same_folder).glob("*.csv"):
            candidates.append(p)
    unique = []
    seen = set()
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        # Avoid accidentally including external overlay databases in the model table.
        lower = rp.lower().replace("\\", "/")
        if any(x in lower for x in ["core_mof_2024", "core_mof_2025", "mosaec_db", "qmof", "csd_mof_collection"]):
            continue
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    save_json({"n_csv": len(unique), "csv_files": [str(p) for p in unique]}, PATHS["audit"] / "arc_mof_csv_discovery_manifest.json")
    return unique

def read_csv_robust(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, nrows=nrows, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read {path}: {last_err}")


def infer_column_roles(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    missing_frac = df.isna().mean().sort_values(ascending=False)
    likely_ids = [c for c in cols if any(h in slugify(c) for h in ID_HINTS)]
    likely_targets = []
    for c in cols:
        cslug = slugify(c)
        for key, info in TARGET_DEFINITIONS.items():
            if any(re.search(pat, cslug, flags=re.IGNORECASE) for pat in info["patterns"]):
                likely_targets.append(c)
                break
    likely_topology = [c for c in cols if any(h in slugify(c) for h in TOPOLOGY_HINTS)]
    likely_clusters = [c for c in cols if any(h in slugify(c) for h in CLUSTER_HINTS)]
    likely_geometry = [c for c in cols if any(h in slugify(c) for h in CANONICAL_GEOMETRY_HINTS)]
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_numeric_cols": int(len(numeric_cols)),
        "n_categorical_cols": int(len(categorical_cols)),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "likely_id_cols": likely_ids,
        "likely_target_cols": likely_targets,
        "likely_topology_cols": likely_topology,
        "likely_cluster_cols": likely_clusters,
        "likely_geometry_cols": likely_geometry,
        "total_missing_cells": int(df.isna().sum().sum()),
        "missing_fraction_overall": float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1)),
        "top_missing_columns": missing_frac.head(30).to_dict(),
    }


def stage_data_audit(csv_files: List[Path]) -> Dict[str, Any]:
    stage = "data_audit"
    out_json = PATHS["audit"] / "csv_data_audit_summary.json"
    out_csv = PATHS["audit"] / "csv_data_audit_summary.csv"
    out_cols = PATHS["audit"] / "csv_column_audit_long.csv"

    if is_stage_done(stage, [out_json, out_csv, out_cols]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f)

    LOGGER.info("STAGE >>> DATA_AUDIT | n_csv=%d", len(csv_files))
    summary_rows = []
    column_rows = []
    audit: Dict[str, Any] = {"files": {}, "discovered_csv_files": [str(p) for p in csv_files]}

    for path in csv_files:
        try:
            df = read_csv_robust(path, nrows=CFG.MAX_ROWS_FOR_FAST_TEST)
            roles = infer_column_roles(df)
            audit["files"][path.name] = roles
            summary_rows.append({
                "file": path.name,
                "path": str(path),
                "n_rows_read": roles["n_rows"],
                "n_cols": roles["n_cols"],
                "n_numeric_cols": roles["n_numeric_cols"],
                "n_categorical_cols": roles["n_categorical_cols"],
                "missing_fraction_overall": roles["missing_fraction_overall"],
                "likely_id_cols": "; ".join(roles["likely_id_cols"][:10]),
                "likely_target_cols": "; ".join(roles["likely_target_cols"][:10]),
                "likely_topology_cols": "; ".join(roles["likely_topology_cols"][:10]),
                "likely_cluster_cols": "; ".join(roles["likely_cluster_cols"][:10]),
            })
            for c in df.columns:
                column_rows.append({
                    "file": path.name,
                    "column": c,
                    "dtype": str(df[c].dtype),
                    "n_missing": int(df[c].isna().sum()),
                    "missing_fraction": float(df[c].isna().mean()),
                    "n_unique": int(df[c].nunique(dropna=True)),
                    "example_values": "; ".join(map(str, df[c].dropna().astype(str).head(3).tolist())),
                })
            LOGGER.info("AUDIT_FILE_DONE | %s | rows=%d cols=%d missing=%.3f", path.name, roles["n_rows"], roles["n_cols"], roles["missing_fraction_overall"])
        except Exception as e:
            LOGGER.exception("AUDIT_FILE_FAILED | %s | %s", path, e)
            summary_rows.append({"file": path.name, "path": str(path), "error": str(e)})

    summary_df = pd.DataFrame(summary_rows)
    cols_df = pd.DataFrame(column_rows)
    safe_to_csv(summary_df, out_csv)
    safe_to_csv(cols_df, out_cols)
    save_json(audit, out_json)
    save_latex_table(summary_df.head(30), PATHS["tables_si"] / "table_si_data_audit_files.tex", caption="Rapid audit of input CSV files.", label="tab:si_data_audit")
    mark_stage_done(stage, [out_json, out_csv, out_cols])
    return audit

# =============================================================================
# Data loading, ID normalization, merging, and target detection
# =============================================================================

def normalize_identifier_series(s: pd.Series) -> pd.Series:
    """Normalize MOF identifiers to make joins more robust.

    Examples:
    - strips whitespace
    - removes trailing .cif
    - lowercases only for join key while preserving original columns elsewhere
    """
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.cif$", "", regex=True, case=False)
    out = out.str.replace(r"\.json$", "", regex=True, case=False)
    out = out.str.replace(r"^./", "", regex=True)
    return out.str.lower()


def choose_id_column(df: pd.DataFrame) -> Optional[str]:
    """Choose the most plausible identifier column in a dataframe."""
    if df.empty:
        return None
    cols = list(df.columns)
    priority_names = ["filename", "name", "Name", "MOF", "mof", "mofid", "MOFid", "id", "ID", "refcode", "cif"]
    for p in priority_names:
        if p in cols:
            return p
    scored = []
    for c in cols:
        cslug = slugify(c)
        score = sum(1 for h in ID_HINTS if h in cslug)
        # Identifiers are often high-cardinality strings.
        nunique = df[c].nunique(dropna=True)
        if score > 0:
            scored.append((score, nunique, c))
    if scored:
        scored.sort(reverse=True)
        return scored[0][2]
    return None


def add_join_id(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    id_col = choose_id_column(df)
    if id_col is None:
        df["__join_id__"] = [f"{source_name}_{i}" for i in range(len(df))]
        df["__source_id_col__"] = "row_number_generated"
    else:
        df["__join_id__"] = normalize_identifier_series(df[id_col])
        df["__source_id_col__"] = id_col
    return df


def prefix_non_id_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix columns from support files to avoid collisions during merge."""
    rename = {}
    for c in df.columns:
        if c in ["__join_id__", "__source_id_col__"]:
            continue
        if c.startswith(prefix + "__"):
            continue
        rename[c] = f"{prefix}__{c}"
    return df.rename(columns=rename)


def find_file(csv_files: List[Path], target_name: str) -> Optional[Path]:
    target_slug = slugify(target_name).replace("_csv", "")
    for p in csv_files:
        if slugify(p.name).replace("_csv", "") == target_slug:
            return p
    # Tolerate minor naming variants.
    for p in csv_files:
        if target_slug in slugify(p.name):
            return p
    return None



def _target_key_for_gas_pressure(gas_text: str, pressure_bar: float, tol_rel: float = 1e-3) -> Optional[str]:
    """Return the semantic target key for a gas/pressure pair, if recognised."""
    gas_slug = slugify(gas_text)
    for key, p in TARGET_PRESSURES_BAR.items():
        if not _column_has_gas(gas_slug, key):
            continue
        if abs(float(pressure_bar) - float(p)) <= max(1e-6, abs(float(p)) * tol_rel):
            return key
    return None


def _infer_gas_text_from_row_or_context(row: Optional[pd.Series], gas_cols: Sequence[str], context: str) -> str:
    """Infer gas text from a row gas column, falling back to source/table context."""
    pieces: List[str] = [str(context)]
    if row is not None:
        for c in gas_cols:
            try:
                val = row.get(c, "")
            except Exception:
                val = ""
            if pd.notna(val):
                pieces.append(str(val))
    return " ".join(pieces)


def _looks_like_pressure_column(col: str) -> bool:
    s = slugify(col)
    return any(x in s for x in ["pressure", "press", "bar", "mbar", "p_bar", "pbar"]) or s in {"p", "p0"}


def _looks_like_gas_column(col: str) -> bool:
    s = slugify(col)
    return any(x in s for x in ["gas", "component", "adsorbate", "species", "molecule"])


def _looks_like_uptake_column(col: str) -> bool:
    s = slugify(col)
    if any(x in s for x in ["pressure", "temperature", "temp", "bar", "mbar", "index", "id"]):
        return False
    return any(x in s for x in ["uptake", "loading", "adsorption", "amount", "mmol", "mol_kg", "molkg", "q_", "n_"])


def _first_nonnull(s: pd.Series) -> Any:
    """First non-null value helper for duplicate-id aggregation."""
    try:
        non = s.dropna()
        if len(non):
            return non.iloc[0]
    except Exception:
        pass
    return np.nan


def add_derived_adsorption_target_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add canonical target columns derived from raw adsorption tables.

    This is the important clean_data-free fix.  Some ARC-MOF archives do not
    contain a pre-built clean_data.csv; adsorption labels are spread across raw
    CSV files.  This helper detects both wide tables (target encoded in
    filename/column name) and long tables (gas/pressure/loading columns), then
    creates canonical columns named like:

        derived_target__co2_0p15__<source>__<column>

    The original columns are not removed.  The downstream target detector can
    then find these derived columns reliably.
    """
    out = df.copy()
    context = str(source_name)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]

    # Wide-table case: gas/pressure may be in the filename, the column, or both.
    for c in numeric_cols:
        if c in {"__join_id__"}:
            continue
        combined = f"{context} {c}"
        for key in TARGET_DEFINITIONS:
            score, reason = _target_candidate_score(combined, out[c], key)
            if score >= 50 and out[c].notna().sum() > 0:
                new_col = f"derived_target__{key}__{slugify(Path(source_name).stem)}__{slugify(c)}"
                if new_col not in out.columns:
                    out[new_col] = pd.to_numeric(out[c], errors="coerce")

    # Long-table case: separate gas / pressure / loading columns.
    gas_cols = [c for c in out.columns if _looks_like_gas_column(c)]
    pressure_cols = [c for c in out.columns if _looks_like_pressure_column(c) or _extract_pressure_values_bar(c)]
    uptake_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and _looks_like_uptake_column(c)]
    if uptake_cols and (pressure_cols or _extract_pressure_values_bar(context)):
        for pcol in pressure_cols if pressure_cols else [None]:
            if pcol is None:
                pressure_values = pd.Series([_extract_pressure_values_bar(context)[0]] * len(out), index=out.index)
            else:
                raw_pressure = pd.to_numeric(out[pcol], errors="coerce")
                # If the column name says mbar, convert; otherwise assume bar.
                pressure_values = raw_pressure / 1000.0 if "mbar" in slugify(pcol) else raw_pressure
            for ucol in uptake_cols:
                if ucol == pcol:
                    continue
                vals = pd.to_numeric(out[ucol], errors="coerce")
                for key, target_p in TARGET_PRESSURES_BAR.items():
                    gas_ok = pd.Series(True, index=out.index)
                    if gas_cols:
                        gas_ok = pd.Series(False, index=out.index)
                        for gc in gas_cols:
                            gas_ok = gas_ok | out[gc].astype(str).map(lambda x: _column_has_gas(x, key))
                    else:
                        gas_ok = pd.Series([_column_has_gas(context, key)] * len(out), index=out.index)
                    pressure_ok = (pressure_values - target_p).abs() <= max(1e-6, target_p * 1e-3)
                    mask = gas_ok & pressure_ok & vals.notna()
                    if mask.any():
                        new_col = f"derived_target__{key}__{slugify(Path(source_name).stem)}__{slugify(ucol)}"
                        if new_col not in out.columns:
                            out[new_col] = np.nan
                        out.loc[mask, new_col] = vals.loc[mask]
    return out


def prepare_table_for_merge(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add join IDs and clean_data-free derived target columns before merging."""
    tbl = add_join_id(df, source_name)
    tbl = add_derived_adsorption_target_columns(tbl, source_name)
    return tbl


def aggregate_duplicate_join_ids(tbl: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Collapse duplicate IDs without losing derived target columns.

    Long adsorption tables can contain multiple rows per MOF (different gases or
    pressures).  The previous code kept the first row, which can drop the target
    value.  This version takes the first non-null value for derived target columns
    and the first row for ordinary metadata.
    """
    if "__join_id__" not in tbl.columns or not tbl["__join_id__"].duplicated().any():
        return tbl
    before = len(tbl)
    derived_cols = [c for c in tbl.columns if str(c).startswith("derived_target__")]
    other_cols = [c for c in tbl.columns if c not in derived_cols + ["__join_id__"]]
    agg: Dict[str, Any] = {c: "first" for c in other_cols}
    for c in derived_cols:
        agg[c] = _first_nonnull
    try:
        collapsed = tbl.groupby("__join_id__", sort=False, dropna=False).agg(agg).reset_index()
    except TypeError:
        collapsed = tbl.groupby("__join_id__", sort=False).agg(agg).reset_index()
    LOGGER.info("AGGREGATE_DUPLICATE_IDS | %s | %d -> %d | derived_targets=%d", source_name, before, len(collapsed), len(derived_cols))
    return collapsed


def score_table_as_master(name: str, df: pd.DataFrame) -> Tuple[int, int, int, str]:
    """Rank candidate master tables when clean_data.csv is absent."""
    prepared = prepare_table_for_merge(df, name)
    derived_target_count = len([c for c in prepared.columns if str(c).startswith("derived_target__")])
    target_like_count = 0
    for c in prepared.columns:
        for key in TARGET_DEFINITIONS:
            score, _reason = _target_candidate_score(f"{name} {c}", prepared[c], key)
            if score >= 50:
                target_like_count += 1
                break
    # Prefer adsorption-label tables, then descriptor-rich tables, then row count.
    name_score = 0
    s = slugify(name)
    if any(x in s for x in ["adsorption", "uptake", "co2", "methane", "ch4"]):
        name_score += 50
    if any(x in s for x in ["geometric", "descriptor", "topology", "cluster"]):
        name_score += 10
    return (derived_target_count * 100 + target_like_count * 50 + name_score, df.shape[1], df.shape[0], name)

def load_all_available_tables(csv_files: List[Path]) -> Dict[str, pd.DataFrame]:
    tables = {}
    for p in csv_files:
        try:
            df = read_csv_robust(p, nrows=CFG.MAX_ROWS_FOR_FAST_TEST)
            tables[p.name] = df
        except Exception as e:
            LOGGER.warning("LOAD_TABLE_FAILED | %s | %s", p.name, e)
    return tables


def build_merged_table(csv_files: List[Path]) -> pd.DataFrame:
    """Build the modelling table from available ARC-MOF CSVs.

    Clean-data-free behaviour:
    - If clean_data.csv exists, it is still used as the master table.
    - If it does not exist, the code no longer chooses the widest table blindly.
      Instead, it creates a union of MOF identifiers from all ARC-MOF CSV files
      and merges adsorption, descriptor, topology, cluster and process tables
      onto that union.
    - Raw adsorption files are scanned for target information and canonical
      derived target columns are created automatically.

    This fixes the previous failure mode where overall_process.csv or another
    wide non-adsorption file became the master and no target columns were
    detected.
    """
    stage = "build_merged_modelling_table"
    out_csv = PATHS["processed"] / "merged_modelling_table.csv"
    out_pkl = PATHS["processed"] / "merged_modelling_table.pkl"
    out_report = PATHS["processed"] / "merge_report.json"

    if is_stage_done(stage, [out_csv, out_pkl, out_report]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        return pd.read_pickle(out_pkl)

    LOGGER.info("STAGE >>> BUILD_MERGED_TABLE")
    tables = load_all_available_tables(csv_files)
    if not tables:
        raise FileNotFoundError(
            "No CSV files were found. Pass --data_root to the folder that contains data_raw/arc_mof "
            "or place ARC-MOF CSV files beside the script."
        )

    prepared_tables: Dict[str, pd.DataFrame] = {}
    table_scores: List[Dict[str, Any]] = []
    for name, df in tables.items():
        prepared = prepare_table_for_merge(df, name)
        prepared_tables[name] = prepared
        score_tuple = score_table_as_master(name, df)
        table_scores.append({
            "file": name,
            "score": int(score_tuple[0]),
            "n_cols": int(df.shape[1]),
            "n_rows": int(df.shape[0]),
            "derived_target_cols": int(len([c for c in prepared.columns if str(c).startswith("derived_target__")])),
            "source_id_col": str(prepared["__source_id_col__"].iloc[0]) if len(prepared) else None,
        })
    score_df = pd.DataFrame(table_scores).sort_values(["score", "n_cols", "n_rows"], ascending=[False, False, False])
    safe_to_csv(score_df, PATHS["processed"] / "master_table_selection_scores.csv")

    clean_candidates = [name for name in prepared_tables if slugify(name) == "clean_data_csv"]
    use_union_master = len(clean_candidates) == 0

    merge_report = {
        "clean_data_present": bool(clean_candidates),
        "master_strategy": "clean_data_master" if not use_union_master else "union_of_all_join_ids_clean_data_free",
        "master_file": clean_candidates[0] if clean_candidates else None,
        "support_files": [],
        "table_selection_scores": table_scores,
    }

    if not use_union_master:
        master_name = clean_candidates[0]
        master = aggregate_duplicate_join_ids(prepared_tables[master_name], master_name)
        merged = master.copy()
        merge_report["master_file"] = master_name
        merge_report["master_rows"] = int(merged.shape[0])
        merge_report["master_cols"] = int(merged.shape[1])
        merge_items = [(name, tbl) for name, tbl in prepared_tables.items() if name != master_name]
        LOGGER.info("MERGE_MASTER | clean_data.csv found and used as master")
    else:
        # Build a union of IDs from all tables.  This is safer than choosing the
        # widest file because adsorption targets and descriptors may live in
        # different folders and no single table is the full clean_data table.
        all_ids = []
        for name, tbl in prepared_tables.items():
            if "__join_id__" in tbl.columns:
                all_ids.append(tbl["__join_id__"].astype(str))
        if not all_ids:
            raise ValueError("No identifier columns could be inferred from the input CSV files.")
        union_ids = pd.concat(all_ids, ignore_index=True).dropna().drop_duplicates()
        merged = pd.DataFrame({"__join_id__": union_ids.values})
        merged["display_id"] = merged["__join_id__"].astype(str)
        merge_report["master_rows"] = int(merged.shape[0])
        merge_report["master_cols"] = int(merged.shape[1])
        merge_items = list(prepared_tables.items())
        LOGGER.warning("clean_data.csv not found. Building modelling table from the union of all join IDs across %d CSV files.", len(merge_items))

    for name, tbl in merge_items:
        support = aggregate_duplicate_join_ids(tbl, name)
        prefix = slugify(Path(name).stem)
        support_pref = prefix_non_id_columns(support, prefix)
        before_cols = set(merged.columns)
        n_before = len(merged)
        merged = merged.merge(support_pref, on="__join_id__", how="left", suffixes=("", f"_{prefix}"))
        added_cols = [c for c in merged.columns if c not in before_cols]
        match_frac = float(merged[added_cols].notna().any(axis=1).mean()) if added_cols else 0.0
        merge_report["support_files"].append({
            "file": name,
            "rows": int(tbl.shape[0]),
            "cols": int(tbl.shape[1]),
            "added_cols": int(len(added_cols)),
            "derived_target_cols_added": int(len([c for c in added_cols if str(c).startswith(prefix + "__derived_target__") or str(c).startswith("derived_target__")])),
            "master_rows_before": int(n_before),
            "master_rows_after": int(len(merged)),
            "row_match_fraction_any_added_col": match_frac,
            "source_id_col": str(tbl["__source_id_col__"].iloc[0]) if len(tbl) else None,
        })
        LOGGER.info("MERGE_SUPPORT_DONE | %s | added_cols=%d match_frac=%.3f", name, len(added_cols), match_frac)

    # Preserve a clean display ID for tables.  In union mode, prefer any filename/name
    # column that survived the merge; otherwise keep the join id.
    display_candidates = [c for c in merged.columns if slugify(c).endswith("filename") or slugify(c).endswith("name")]
    if display_candidates:
        best = max(display_candidates, key=lambda c: merged[c].notna().sum())
        merged["display_id"] = merged[best].fillna(merged["__join_id__"]).astype(str)
    elif "display_id" not in merged.columns:
        merged["display_id"] = merged["__join_id__"].astype(str)

    # Basic duplicate cleanup.
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    # If union mode creates rows without targets or without descriptors, they are
    # harmless.  The model split stage filters per target and feature availability.
    derived_cols = [c for c in merged.columns if "derived_target__" in str(c)]
    merge_report["merged_rows"] = int(len(merged))
    merge_report["merged_cols"] = int(merged.shape[1])
    merge_report["derived_target_columns_in_merged_table"] = derived_cols
    LOGGER.info("MERGED_TABLE_READY | rows=%d cols=%d derived_target_cols=%d", len(merged), merged.shape[1], len(derived_cols))

    if getattr(CFG, "SAVE_MERGED_ENGINEERED_CSV", False):
        safe_to_csv(merged, out_csv)
    else:
        safe_to_csv(pd.DataFrame([{"note": "large merged CSV skipped; use merged_modelling_table.pkl", "n_rows": len(merged), "n_cols": merged.shape[1]}]), out_csv)
    safe_to_pickle_df(merged, out_pkl)
    save_json(merge_report, out_report)
    mark_stage_done(stage, [out_csv, out_pkl, out_report], {"n_rows": len(merged), "n_cols": merged.shape[1]})
    return merged


def detect_target_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map semantic target keys to real dataframe columns with strict QC.

    This revised detector is pressure-aware. It prevents the common high-impact
    paper failure mode in which CO2 at 0.15 bar is accidentally mapped to CO2 at
    0.015 bar. Manual overrides can be supplied in CFG.TARGET_COLUMN_OVERRIDES,
    for example:

        TARGET_COLUMN_OVERRIDES={"co2_0p15": "uptake(mmol/g) CO2 at 0.15 bar"}

    The detector writes a ranked candidate table to
    results/processed_data/target_detection_candidate_scores.csv.
    """
    mapping: Dict[str, Optional[str]] = {}
    candidate_rows: List[Dict[str, Any]] = []

    # Honour explicit user overrides first.
    for key in TARGET_DEFINITIONS:
        override = CFG.TARGET_COLUMN_OVERRIDES.get(key) if hasattr(CFG, "TARGET_COLUMN_OVERRIDES") else None
        if override:
            if override not in df.columns:
                raise ValueError(f"TARGET_COLUMN_OVERRIDES[{key!r}]={override!r} was not found in the merged dataframe.")
            mapping[key] = override
            candidate_rows.append({
                "target_key": key,
                "column": override,
                "score": 1e6,
                "reason": "manual_override",
                "n_non_missing": int(df[override].notna().sum()),
                "dtype": str(df[override].dtype),
            })

    used_cols = set(mapping.values())
    for key in TARGET_DEFINITIONS:
        if key in mapping:
            continue
        scored: List[Tuple[float, str, str]] = []
        for c in df.columns:
            if c in {"__join_id__", "display_id", "__source_id_col__"}:
                continue
            score, reason = _target_candidate_score(c, df[c], key)
            if score > -1e8:
                scored.append((score, reason, c))
                candidate_rows.append({
                    "target_key": key,
                    "column": c,
                    "score": float(score),
                    "reason": reason,
                    "n_non_missing": int(df[c].notna().sum()),
                    "dtype": str(df[c].dtype),
                    "pressures_bar_detected": ";".join(f"{v:g}" for v in _extract_pressure_values_bar(c)),
                })
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = None
        for score, reason, c in scored:
            # score >= 50 means gas + pressure were detected either directly in
            # the column name or via the derived clean_data-free target columns.
            if score >= 50 and c not in used_cols:
                chosen = c
                break
        # Fallback for derived canonical target columns.  These are generated from
        # raw adsorption tables and are safer than fuzzy matching arbitrary names.
        if chosen is None:
            derived_hits = [
                c for c in df.columns
                if str(c).startswith("derived_target__" + key + "__")
                or ("__derived_target__" + key + "__") in str(c)
            ]
            derived_hits = sorted(
                derived_hits,
                key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum()) if c in df.columns else 0,
                reverse=True,
            )
            for c in derived_hits:
                if c not in used_cols and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 50:
                    chosen = c
                    candidate_rows.append({
                        "target_key": key,
                        "column": c,
                        "score": 1000.0,
                        "reason": "derived_target_fallback",
                        "n_non_missing": int(pd.to_numeric(df[c], errors="coerce").notna().sum()),
                        "dtype": str(df[c].dtype),
                        "pressures_bar_detected": ";".join(f"{v:g}" for v in _extract_pressure_values_bar(c)),
                    })
                    break
        mapping[key] = chosen
        if chosen is not None:
            used_cols.add(chosen)

    cand_df = pd.DataFrame(candidate_rows).sort_values(["target_key", "score"], ascending=[True, False]) if candidate_rows else pd.DataFrame()
    if not cand_df.empty:
        safe_to_csv(cand_df, PATHS["processed"] / "target_detection_candidate_scores.csv")

    # Duplicate mapping is almost always a target-QC problem. Do not silently run
    # two targets on the same label.
    reverse: Dict[str, List[str]] = {}
    for key, col in mapping.items():
        if col is not None:
            reverse.setdefault(col, []).append(key)
    duplicates = {col: keys for col, keys in reverse.items() if len(keys) > 1}
    qc_report = {
        "mapping": mapping,
        "duplicates": duplicates,
        "strict_target_qc": getattr(CFG, "STRICT_TARGET_QC", True),
    }
    save_json(qc_report, PATHS["processed"] / "target_detection_qc_report.json")
    if duplicates:
        msg = "Duplicate target-column mapping detected: " + json.dumps(duplicates, default=str)
        if getattr(CFG, "STRICT_TARGET_QC", True):
            raise ValueError(msg + "\nPlease set CFG.TARGET_COLUMN_OVERRIDES or inspect target_detection_candidate_scores.csv.")
        LOGGER.warning("TARGET_QC_DUPLICATE_MAPPING | %s", msg)
    return mapping

# =============================================================================
# Feature engineering and feature families
# =============================================================================

def is_probably_target(col: str) -> bool:
    c = slugify(col)
    target_words = ["uptake", "loading", "adsorption", "co2", "ch4", "methane", "n2", "selectivity", "working_capacity"]
    pressure_pattern = bool(re.search(r"(^|_)\d+p?\d*(_|$)", c))
    return any(w in c for w in target_words) and pressure_pattern


def add_engineered_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create chemically interpretable pore-shape features when inputs exist."""
    out = df.copy()
    cols_by_slug = {slugify(c): c for c in out.columns}

    def find_first(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            for slug, col in cols_by_slug.items():
                if cand in slug:
                    return col
        return None

    density = find_first(["density"])
    asa = find_first(["asa", "surface_area"])
    ava = find_first(["ava", "pore_volume", "void_volume"])
    avaf = find_first(["avaf", "void_fraction", "porosity"])
    pld = find_first(["pld", "df", "limiting_pore"])
    lcd = find_first(["lcd", "di", "largest_cavity"])

    eps = 1e-9
    if pld and lcd:
        out["eng__lcd_pld_ratio"] = pd.to_numeric(out[lcd], errors="coerce") / (pd.to_numeric(out[pld], errors="coerce") + eps)
        out["eng__cavity_window_gap"] = pd.to_numeric(out[lcd], errors="coerce") - pd.to_numeric(out[pld], errors="coerce")
    if asa and ava:
        out["eng__sa_pv_ratio"] = pd.to_numeric(out[asa], errors="coerce") / (pd.to_numeric(out[ava], errors="coerce") + eps)
    if avaf and density:
        out["eng__vf_density_ratio"] = pd.to_numeric(out[avaf], errors="coerce") / (pd.to_numeric(out[density], errors="coerce") + eps)
    if pld:
        out["eng__log_pld_plus1"] = np.log1p(pd.to_numeric(out[pld], errors="coerce").clip(lower=0))
    if lcd:
        out["eng__log_lcd_plus1"] = np.log1p(pd.to_numeric(out[lcd], errors="coerce").clip(lower=0))
    if avaf and pld:
        out["eng__porosity_window_product"] = pd.to_numeric(out[avaf], errors="coerce") * pd.to_numeric(out[pld], errors="coerce")
    if density and pld:
        out["eng__density_pld_interaction"] = pd.to_numeric(out[density], errors="coerce") * pd.to_numeric(out[pld], errors="coerce")
    return out


def identify_feature_columns(df: pd.DataFrame, target_mapping: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    """Identify candidate geometry, topology, cluster and process columns."""
    target_cols = set(c for c in target_mapping.values() if c)
    forbidden = set(target_cols) | {"__join_id__", "display_id", "__source_id_col__"}

    geometry = []
    topology = []
    clusters = []
    process = []
    engineered = []

    for c in df.columns:
        if c in forbidden:
            continue
        cslug = slugify(c)
        if c.startswith("eng__"):
            engineered.append(c)
            continue
        if is_probably_target(c):
            continue
        if any(h in cslug for h in TOPOLOGY_HINTS):
            topology.append(c)
        if any(h in cslug for h in CLUSTER_HINTS):
            clusters.append(c)
        if any(h in cslug for h in CANONICAL_GEOMETRY_HINTS):
            geometry.append(c)
        if any(h in cslug for h in ["process", "vsa", "productivity", "purity", "recovery", "energy", "working", "qst", "heat"]):
            process.append(c)

    # Remove overlaps where process columns are target-like.
    process = [c for c in process if c not in target_cols and not is_probably_target(c)]
    # Make unique while preserving order.
    def unique(seq: Sequence[str]) -> List[str]:
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return {
        "geometry": unique(geometry),
        "engineered": unique(engineered),
        "topology": unique(topology),
        "clusters": unique(clusters),
        "process": unique(process),
    }


def filter_usable_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Keep columns that have at least some information and manageable cardinality."""
    usable = []
    for c in columns:
        if c not in df.columns:
            continue
        if df[c].notna().sum() < max(10, int(0.005 * len(df))):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) <= 1:
                continue
            usable.append(c)
        else:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= CFG.MAX_CATEGORICAL_CARDINALITY:
                usable.append(c)
    return usable


def build_feature_families(df: pd.DataFrame, target_mapping: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    df = add_engineered_geometry_features(df)
    role_cols = identify_feature_columns(df, target_mapping)
    geometry = filter_usable_columns(df, role_cols["geometry"])
    engineered = filter_usable_columns(df, role_cols["engineered"])
    topology = filter_usable_columns(df, role_cols["topology"])
    clusters = filter_usable_columns(df, role_cols["clusters"])
    process = filter_usable_columns(df, role_cols["process"])

    families = {
        "geometry_only": geometry,
        "enriched_geometry": list(dict.fromkeys(geometry + engineered)),
        "topology_only": topology,
        "geometry_topology": list(dict.fromkeys(geometry + engineered + topology)),
        "geometry_topology_clusters": list(dict.fromkeys(geometry + engineered + topology + clusters)),
        "process_enhanced_optional": list(dict.fromkeys(geometry + engineered + topology + clusters + process)),
    }
    families = {k: v for k, v in families.items() if len(v) > 0}

    report = {
        "roles": {k: v for k, v in role_cols.items()},
        "families": {k: {"n_columns": len(v), "columns": v} for k, v in families.items()},
    }
    save_json(report, PATHS["processed"] / "feature_family_report_unfiltered.json")
    families, redundancy_df = feature_family_redundancy_report(families, drop_duplicates=getattr(CFG, "DROP_REDUNDANT_FEATURE_FAMILIES", True))
    report["families_after_redundancy_filter"] = {k: {"n_columns": len(v), "columns": v} for k, v in families.items()}
    save_json(report, PATHS["processed"] / "feature_family_report.json")
    return families

# =============================================================================
# Split construction
# =============================================================================

def choose_group_column(df: pd.DataFrame, split_type: str, feature_roles: Dict[str, List[str]]) -> Optional[str]:
    """Choose a grouping column for grouped splits."""
    candidates: List[str] = []
    if split_type == "topology_grouped":
        candidates = [c for c in df.columns if any(h in slugify(c) for h in TOPOLOGY_HINTS)]
    elif split_type == "geo_cluster_grouped":
        candidates = [c for c in df.columns if "geo" in slugify(c) and "cluster" in slugify(c)]
    elif split_type == "chemistry_cluster_grouped":
        candidates = [c for c in df.columns if any(h in slugify(c) for h in ["mc_cluster", "func_cluster", "flig_cluster", "metal", "functional", "linker", "cluster"])]
    else:
        return None

    scored = []
    for c in candidates:
        nunique = df[c].nunique(dropna=True)
        if 3 <= nunique <= max(5000, len(df) // 5):
            nonmissing = df[c].notna().sum()
            scored.append((nonmissing, -nunique, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def make_three_way_split(
    df: pd.DataFrame,
    target_col: str,
    split_type: str,
    seed: int,
    feature_roles: Dict[str, List[str]],
) -> Optional[Dict[str, np.ndarray]]:
    """Return train/calibration/test indices for random or grouped split."""
    valid_idx = np.where(pd.to_numeric(df[target_col], errors="coerce").notna().values)[0]
    if len(valid_idx) < 100:
        LOGGER.warning("SPLIT_TOO_SMALL | target=%s n=%d", target_col, len(valid_idx))
        return None

    if split_type == "random":
        train_cal_idx, test_idx = train_test_split(valid_idx, test_size=CFG.TEST_SIZE, random_state=seed)
        train_idx, cal_idx = train_test_split(train_cal_idx, test_size=CFG.CALIBRATION_SIZE_OF_REMAINING, random_state=seed + 1000)
        return {"train": np.array(train_idx), "cal": np.array(cal_idx), "test": np.array(test_idx), "group_col": None}

    group_col = choose_group_column(df.iloc[valid_idx].copy(), split_type, feature_roles)
    if group_col is None or group_col not in df.columns:
        LOGGER.warning("GROUP_SPLIT_SKIPPED_NO_GROUP | split=%s", split_type)
        return None

    groups_all = df.iloc[valid_idx][group_col].astype(str).fillna("missing_group").values
    if len(set(groups_all)) < 3:
        LOGGER.warning("GROUP_SPLIT_SKIPPED_FEW_GROUPS | split=%s group_col=%s", split_type, group_col)
        return None

    try:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=CFG.TEST_SIZE, random_state=seed)
        train_cal_local, test_local = next(gss1.split(valid_idx, groups=groups_all))
        train_cal_idx = valid_idx[train_cal_local]
        test_idx = valid_idx[test_local]
        groups_train_cal = df.iloc[train_cal_idx][group_col].astype(str).fillna("missing_group").values
        gss2 = GroupShuffleSplit(n_splits=1, test_size=CFG.CALIBRATION_SIZE_OF_REMAINING, random_state=seed + 1000)
        train_local, cal_local = next(gss2.split(train_cal_idx, groups=groups_train_cal))
        train_idx = train_cal_idx[train_local]
        cal_idx = train_cal_idx[cal_local]
        return {"train": train_idx, "cal": cal_idx, "test": test_idx, "group_col": group_col}
    except Exception as e:
        LOGGER.warning("GROUP_SPLIT_FAILED | split=%s group_col=%s error=%s", split_type, group_col, e)
        return None

# =============================================================================
# Model building
# =============================================================================

def split_numeric_categorical(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric = []
    categorical = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def make_ohe() -> OneHotEncoder:
    """Version-tolerant OneHotEncoder constructor."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def make_model_pipeline(model_name: str, df: pd.DataFrame, feature_cols: Sequence[str], seed: int) -> Pipeline:
    """Create a scikit-learn pipeline with model-appropriate preprocessing."""
    numeric_cols, categorical_cols = split_numeric_categorical(df, feature_cols)

    if model_name == "ridge":
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ])
        pre = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ], remainder="drop")
        model = Ridge(alpha=1.0, random_state=seed)
        return Pipeline([("pre", pre), ("model", model)])

    if model_name == "rf":
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ])
        pre = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ], remainder="drop")
        model = RandomForestRegressor(
            n_estimators=CFG.RF_N_ESTIMATORS,
            max_depth=CFG.RF_MAX_DEPTH,
            min_samples_leaf=CFG.RF_MIN_SAMPLES_LEAF,
            min_samples_split=CFG.RF_MIN_SAMPLES_SPLIT,
            max_features=CFG.RF_MAX_FEATURES,
            bootstrap=CFG.RF_BOOTSTRAP,
            max_samples=CFG.RF_MAX_SAMPLES if CFG.RF_BOOTSTRAP else None,
            random_state=seed,
            n_jobs=CFG.N_JOBS,
        )
        return Pipeline([("pre", pre), ("model", model)])

    if model_name == "hgb":
        # HistGradientBoostingRegressor accepts dense numeric arrays. We ordinal-
        # encode categoricals, which is fast and CPU-friendly. This is not ideal
        # for nominal categories but works well enough as a nonlinear baseline.
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        pre = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ], remainder="drop")
        model = HistGradientBoostingRegressor(
            max_iter=CFG.HGB_MAX_ITER,
            learning_rate=CFG.HGB_LEARNING_RATE,
            max_leaf_nodes=CFG.HGB_MAX_LEAF_NODES,
            min_samples_leaf=CFG.HGB_MIN_SAMPLES_LEAF,
            l2_regularization=CFG.HGB_L2_REGULARIZATION,
            early_stopping=CFG.HGB_EARLY_STOPPING,
            validation_fraction=CFG.HGB_VALIDATION_FRACTION,
            n_iter_no_change=CFG.HGB_N_ITER_NO_CHANGE,
            tol=CFG.HGB_TOL,
            random_state=seed,
        )
        return Pipeline([("pre", pre), ("model", model)])

    raise ValueError(f"Unknown model_name: {model_name}")

# =============================================================================
# Metrics, conformal intervals, elite/anomaly decisions
# =============================================================================

def finite_sample_conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal quantile.

    For calibration scores s_1,...,s_n and desired coverage 1-alpha, use the
    ceil((n+1)*(1-alpha))/n empirical quantile with higher interpolation. Here
    coverage is 1-alpha, e.g. 0.90.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("nan")
    rank = int(math.ceil((n + 1) * coverage))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def adaptive_conformal_widths_by_prediction_bin(
    yhat_cal: np.ndarray,
    cal_abs_scores: np.ndarray,
    yhat_test: np.ndarray,
    coverage: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return locally adaptive conformal widths using prediction-value bins.

    Standard split-conformal regression gives one constant residual quantile per
    job. That is valid and useful, but it means LCB_i = prediction_i - constant,
    so LCB ranking is identical to predicted ranking. For high-impact screening
    figures, this optional adaptive variant bins the calibration set by predicted
    uptake and computes a finite-sample conformal quantile inside each bin, with
    fallback to the global quantile when a bin is too small.

    The resulting intervals are still simple, CPU-light and auditable. They are
    best described as prediction-bin/Mondrian-style adaptive conformal intervals.
    """
    yhat_cal = np.asarray(yhat_cal, dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)
    cal_abs_scores = np.asarray(cal_abs_scores, dtype=float)
    global_q = finite_sample_conformal_quantile(cal_abs_scores, coverage)
    widths = np.full(len(yhat_test), global_q, dtype=float)
    rows: List[Dict[str, Any]] = []

    mask = np.isfinite(yhat_cal) & np.isfinite(cal_abs_scores)
    if mask.sum() < max(50, CFG.ADAPTIVE_CONFORMAL_MIN_CAL_PER_BIN):
        return widths, pd.DataFrame([{"bin": "global", "n_cal": int(mask.sum()), "q": global_q, "fallback": True}])

    n_bins = int(min(CFG.ADAPTIVE_CONFORMAL_N_BINS, max(2, mask.sum() // max(CFG.ADAPTIVE_CONFORMAL_MIN_CAL_PER_BIN, 1))))
    n_bins = max(2, n_bins)
    cal_pred = pd.Series(yhat_cal[mask])
    cal_scores = pd.Series(cal_abs_scores[mask])
    try:
        bins = pd.qcut(cal_pred, q=n_bins, duplicates="drop")
    except Exception:
        return widths, pd.DataFrame([{"bin": "global", "n_cal": int(mask.sum()), "q": global_q, "fallback": True}])

    intervals = list(bins.cat.categories)
    for interval in intervals:
        idx = bins == interval
        scores = cal_scores.loc[idx].values
        if len(scores) >= CFG.ADAPTIVE_CONFORMAL_MIN_CAL_PER_BIN:
            q = finite_sample_conformal_quantile(scores, coverage)
            fallback = False
        else:
            q = global_q
            fallback = True
        rows.append({
            "bin": str(interval),
            "left": float(interval.left),
            "right": float(interval.right),
            "n_cal": int(len(scores)),
            "q": float(q),
            "fallback": bool(fallback),
        })
        test_idx = (yhat_test > interval.left) & (yhat_test <= interval.right)
        widths[test_idx] = q

    # Include out-of-range test predictions in nearest edge bin.
    if rows:
        leftmost = min(r["left"] for r in rows)
        rightmost = max(r["right"] for r in rows)
        left_q = rows[np.argmin([r["left"] for r in rows])]["q"]
        right_q = rows[np.argmax([r["right"] for r in rows])]["q"]
        widths[yhat_test <= leftmost] = left_q
        widths[yhat_test > rightmost] = right_q
    return widths, pd.DataFrame(rows)


def compute_screening_metrics_from_bounds(
    pred_df: pd.DataFrame,
    elite_thresholds: Dict[float, float],
    coverage: float,
    lower_col: str,
    upper_col: str,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute screening metrics for arbitrary lower/upper interval columns."""
    y = pred_df["y_true"].values.astype(float)
    yhat = pred_df["y_pred"].values.astype(float)
    L = pred_df[lower_col].values.astype(float)
    U = pred_df[upper_col].values.astype(float)
    inside = (y >= L) & (y <= U)
    metrics = regression_metrics(y, yhat)
    p = f"{prefix}_" if prefix else ""
    out: Dict[str, Any] = {
        f"{p}coverage_level": coverage,
        f"{p}empirical_coverage": float(np.nanmean(inside)),
        f"{p}mean_interval_width": float(np.nanmean(U - L)),
        f"{p}median_interval_width": float(np.nanmedian(U - L)),
    }
    out.update({f"{p}{k}": v for k, v in metrics.items()})
    for q, thr in elite_thresholds.items():
        qtag = int(q * 100)
        true_elite = y >= thr
        pred_elite = yhat >= thr
        confident_elite = L >= thr
        possible_elite = U >= thr
        out[f"{p}elite_q{qtag}_confident_count"] = int(confident_elite.sum())
        out[f"{p}elite_q{qtag}_possible_count"] = int(possible_elite.sum())
        out[f"{p}elite_q{qtag}_confident_recall"] = float((confident_elite & true_elite).sum() / max(true_elite.sum(), 1))
        out[f"{p}elite_q{qtag}_possible_recall"] = float((possible_elite & true_elite).sum() / max(true_elite.sum(), 1))
        out[f"{p}elite_q{qtag}_confident_false_elite_rate"] = float((confident_elite & ~true_elite).sum() / max(confident_elite.sum(), 1))
        out[f"{p}elite_q{qtag}_predicted_false_elite_rate"] = float((pred_elite & ~true_elite).sum() / max(pred_elite.sum(), 1))
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(math.ceil(frac * len(pred_df))))
        top_bound_ids = pred_df.sort_values(lower_col, ascending=False).head(k)["display_id"].tolist()
        top_true_ids = pred_df.sort_values("y_true", ascending=False).head(k)["display_id"].tolist()
        out[f"{p}top{int(frac*100)}_jaccard_lcb_vs_true"] = jaccard(top_bound_ids, top_true_ids)
        out[f"{p}top{int(frac*100)}_ndcg_lcb"] = ndcg_at_k(y, L, k)
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": 0}
    rmse = math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    try:
        r2 = r2_score(y_true[mask], y_pred[mask])
    except Exception:
        r2 = np.nan
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "n": int(mask.sum())}


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return np.nan
    return len(sa & sb) / max(len(sa | sb), 1)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute NDCG@k for continuous relevance y_true."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return np.nan
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    ideal = np.argsort(-y_true)[:k]
    gains = np.maximum(y_true, 0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / idcg if idcg > 0 else np.nan


def compute_screening_metrics(
    pred_df: pd.DataFrame,
    elite_thresholds: Dict[float, float],
    coverage: float,
) -> Dict[str, Any]:
    """Compute coverage and screening metrics for a prediction dataframe."""
    y = pred_df["y_true"].values.astype(float)
    yhat = pred_df["y_pred"].values.astype(float)
    L = pred_df[f"L_{int(coverage*100)}"].values.astype(float)
    U = pred_df[f"U_{int(coverage*100)}"].values.astype(float)
    inside = (y >= L) & (y <= U)
    metrics = regression_metrics(y, yhat)
    metrics.update({
        "coverage_level": coverage,
        "empirical_coverage": float(np.nanmean(inside)),
        "mean_interval_width": float(np.nanmean(U - L)),
        "median_interval_width": float(np.nanmedian(U - L)),
    })
    for q, thr in elite_thresholds.items():
        true_elite = y >= thr
        pred_elite = yhat >= thr
        confident_elite = L >= thr
        possible_elite = U >= thr
        metrics[f"elite_q{int(q*100)}_threshold"] = float(thr)
        metrics[f"elite_q{int(q*100)}_true_count"] = int(true_elite.sum())
        metrics[f"elite_q{int(q*100)}_predicted_count"] = int(pred_elite.sum())
        metrics[f"elite_q{int(q*100)}_confident_count"] = int(confident_elite.sum())
        metrics[f"elite_q{int(q*100)}_possible_count"] = int(possible_elite.sum())
        metrics[f"elite_q{int(q*100)}_predicted_recall"] = float((pred_elite & true_elite).sum() / max(true_elite.sum(), 1))
        metrics[f"elite_q{int(q*100)}_confident_recall"] = float((confident_elite & true_elite).sum() / max(true_elite.sum(), 1))
        metrics[f"elite_q{int(q*100)}_possible_recall"] = float((possible_elite & true_elite).sum() / max(true_elite.sum(), 1))
        metrics[f"elite_q{int(q*100)}_predicted_false_elite_rate"] = float((pred_elite & ~true_elite).sum() / max(pred_elite.sum(), 1))
        metrics[f"elite_q{int(q*100)}_confident_false_elite_rate"] = float((confident_elite & ~true_elite).sum() / max(confident_elite.sum(), 1))
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(math.ceil(frac * len(pred_df))))
        top_pred_ids = pred_df.sort_values("y_pred", ascending=False).head(k)["display_id"].tolist()
        top_lcb_ids = pred_df.sort_values(f"L_{int(coverage*100)}", ascending=False).head(k)["display_id"].tolist()
        top_true_ids = pred_df.sort_values("y_true", ascending=False).head(k)["display_id"].tolist()
        metrics[f"top{int(frac*100)}_jaccard_pred_vs_true"] = jaccard(top_pred_ids, top_true_ids)
        metrics[f"top{int(frac*100)}_jaccard_lcb_vs_true"] = jaccard(top_lcb_ids, top_true_ids)
        metrics[f"top{int(frac*100)}_jaccard_pred_vs_lcb"] = jaccard(top_pred_ids, top_lcb_ids)
        metrics[f"top{int(frac*100)}_ndcg_pred"] = ndcg_at_k(y, yhat, k)
        metrics[f"top{int(frac*100)}_ndcg_lcb"] = ndcg_at_k(y, L, k)
    return metrics


def conditional_coverage_by_group(pred_df: pd.DataFrame, group_col: Optional[str], coverage: float, min_group_size: int = 20) -> pd.DataFrame:
    if group_col is None or group_col not in pred_df.columns:
        return pd.DataFrame()
    Lcol, Ucol = f"L_{int(coverage*100)}", f"U_{int(coverage*100)}"
    rows = []
    for g, sub in pred_df.groupby(group_col, dropna=False):
        if len(sub) < min_group_size:
            continue
        inside = (sub["y_true"] >= sub[Lcol]) & (sub["y_true"] <= sub[Ucol])
        rows.append({
            "group_col": group_col,
            "group": str(g),
            "n": int(len(sub)),
            "coverage": float(inside.mean()),
            "mean_width": float((sub[Ucol] - sub[Lcol]).mean()),
            "mean_y": float(sub["y_true"].mean()),
        })
    return pd.DataFrame(rows)

# =============================================================================
# Main model-fitting loop
# =============================================================================

def run_single_job(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    feature_family: str,
    feature_cols: List[str],
    model_name: str,
    split_type: str,
    seed: int,
    feature_roles: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """Fit one target/family/model/split/seed job and save all outputs."""
    # Include target_col in the job id so corrected target mappings cannot
    # accidentally reuse old prediction files from an earlier mis-mapped run.
    target_col_tag = slugify(target_col, max_len=45)
    job_id = slugify(f"{target_key}__{target_col_tag}__{feature_family}__{model_name}__{split_type}__seed{seed}")
    pred_path = prediction_path_for_job(job_id)
    existing_pred_path = find_existing_prediction_path(job_id)
    model_path = PATHS["models"] / f"model__{job_id}.joblib"
    metrics_path = PATHS["metrics"] / f"metrics__{job_id}.json"

    if metrics_path.exists() and existing_pred_path is not None and not CFG.FORCE_RERUN:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return {"job_id": job_id, "metrics": metrics, "pred_path": existing_pred_path}
        except Exception:
            pass

    try:
        split = make_three_way_split(df, target_col, split_type, seed, feature_roles)
        if split is None:
            return None

        train_idx, cal_idx, test_idx = split["train"], split["cal"], split["test"]
        group_col = split.get("group_col")
        if len(train_idx) < 50 or len(cal_idx) < 30 or len(test_idx) < 30:
            LOGGER.warning("JOB_SKIPPED_SMALL_SPLIT | %s", job_id)
            return None

        use_cols = [c for c in feature_cols if c in df.columns]
        if len(use_cols) == 0:
            LOGGER.warning("JOB_SKIPPED_NO_FEATURES | %s", job_id)
            return None

        X_train = df.iloc[train_idx][use_cols]
        y_train = pd.to_numeric(df.iloc[train_idx][target_col], errors="coerce").values.astype(float)
        X_cal = df.iloc[cal_idx][use_cols]
        y_cal = pd.to_numeric(df.iloc[cal_idx][target_col], errors="coerce").values.astype(float)
        X_test = df.iloc[test_idx][use_cols]
        y_test = pd.to_numeric(df.iloc[test_idx][target_col], errors="coerce").values.astype(float)

        # Remove any rows with missing y after split.
        tr_mask = np.isfinite(y_train); ca_mask = np.isfinite(y_cal); te_mask = np.isfinite(y_test)
        X_train, y_train = X_train.loc[tr_mask], y_train[tr_mask]
        X_cal, y_cal = X_cal.loc[ca_mask], y_cal[ca_mask]
        X_test, y_test = X_test.loc[te_mask], y_test[te_mask]
        test_idx_clean = np.array(test_idx)[te_mask]
        cal_idx_clean = np.array(cal_idx)[ca_mask]

        pipe = make_model_pipeline(model_name, df, use_cols, seed)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        yhat_train = pipe.predict(X_train)
        yhat_cal = pipe.predict(X_cal)
        yhat_test = pipe.predict(X_test)
        cal_resid = y_cal - yhat_cal
        cal_abs = np.abs(cal_resid)
        cal_pos = cal_resid
        cal_neg = -cal_resid

        # Elite thresholds are estimated from training+calibration labels, never
        # from the test labels. This matches a screening setting where the elite
        # boundary is learned before evaluating the held-out screening pool.
        y_ref = np.concatenate([y_train, y_cal])
        elite_thresholds = {q: float(np.nanquantile(y_ref, q)) for q in CFG.ELITE_QUANTILES}

        pred = pd.DataFrame({
            "job_id": job_id,
            "target_key": target_key,
            "target_col": target_col,
            "feature_family": feature_family,
            "model_name": model_name,
            "split_type": split_type,
            "seed": seed,
            "display_id": df.iloc[test_idx_clean]["display_id"].astype(str).values,
            "row_index": test_idx_clean,
            "y_true": y_test,
            "y_pred": yhat_test,
            "residual": y_test - yhat_test,
        })
        # Add group/chemistry/context columns useful for interpretation.
        context_cols = []
        for c in df.columns:
            cslug = slugify(c)
            if any(h in cslug for h in TOPOLOGY_HINTS + CLUSTER_HINTS + ["density", "asa", "ava", "avaf", "di", "df", "dif", "pld", "lcd"]):
                if c not in context_cols and len(context_cols) < int(getattr(CFG, "MAX_CONTEXT_COLUMNS_IN_PREDICTIONS", 12)):
                    context_cols.append(c)
        for c in context_cols:
            try:
                pred[c] = df.iloc[test_idx_clean][c].values
            except Exception:
                pass

        # Conformal intervals and anomaly certificates.
        for cov in CFG.COVERAGE_LEVELS:
            tag = int(cov * 100)
            q_abs = finite_sample_conformal_quantile(cal_abs, cov)
            q_pos = finite_sample_conformal_quantile(cal_pos, cov)
            q_neg = finite_sample_conformal_quantile(cal_neg, cov)
            pred[f"q_abs_{tag}"] = q_abs
            pred[f"q_pos_{tag}"] = q_pos
            pred[f"q_neg_{tag}"] = q_neg
            pred[f"L_{tag}"] = pred["y_pred"] - q_abs
            pred[f"U_{tag}"] = pred["y_pred"] + q_abs
            pred[f"positive_anomaly_{tag}"] = pred["residual"] > q_pos
            pred[f"negative_anomaly_{tag}"] = -pred["residual"] > q_neg
            pred[f"two_sided_anomaly_{tag}"] = np.abs(pred["residual"]) > q_abs

            if getattr(CFG, "ADD_ADAPTIVE_CONFORMAL", True):
                adaptive_widths, adaptive_bin_table = adaptive_conformal_widths_by_prediction_bin(
                    yhat_cal=yhat_cal,
                    cal_abs_scores=cal_abs,
                    yhat_test=yhat_test,
                    coverage=cov,
                )
                pred[f"q_abs_adaptive_{tag}"] = adaptive_widths
                pred[f"L_adaptive_{tag}"] = pred["y_pred"] - adaptive_widths
                pred[f"U_adaptive_{tag}"] = pred["y_pred"] + adaptive_widths
                if getattr(CFG, "SAVE_ADAPTIVE_BIN_TABLES", False) and not adaptive_bin_table.empty:
                    adaptive_bin_path = PATHS["metrics"] / f"adaptive_conformal_bins__{job_id}__cov{tag}.csv"
                    safe_to_csv(adaptive_bin_table, adaptive_bin_path)

        # Approximate conformal p-values for ranking anomaly strength.
        # p_pos = fraction of calibration positive residuals >= test residual.
        pred["p_positive_surprise"] = [float((1 + np.sum(cal_pos >= r)) / (len(cal_pos) + 1)) for r in pred["residual"].values]
        pred["p_negative_surprise"] = [float((1 + np.sum(cal_neg >= -r)) / (len(cal_neg) + 1)) for r in pred["residual"].values]
        pred["p_two_sided_surprise"] = [float((1 + np.sum(cal_abs >= abs(r))) / (len(cal_abs) + 1)) for r in pred["residual"].values]

        main_tag = int(CFG.MAIN_COVERAGE * 100)
        for q, thr in elite_thresholds.items():
            qtag = int(q * 100)
            pred[f"true_elite_q{qtag}"] = pred["y_true"] >= thr
            pred[f"predicted_elite_q{qtag}"] = pred["y_pred"] >= thr
            pred[f"confident_elite_q{qtag}"] = pred[f"L_{main_tag}"] >= thr
            pred[f"possible_elite_q{qtag}"] = pred[f"U_{main_tag}"] >= thr
            if f"L_adaptive_{main_tag}" in pred.columns and f"U_adaptive_{main_tag}" in pred.columns:
                pred[f"confident_elite_adaptive_q{qtag}"] = pred[f"L_adaptive_{main_tag}"] >= thr
                pred[f"possible_elite_adaptive_q{qtag}"] = pred[f"U_adaptive_{main_tag}"] >= thr

        pred["decision_class_main"] = "uncertain_or_nonelite"
        thr_main = elite_thresholds[CFG.MAIN_ELITE_Q]
        pred.loc[pred[f"possible_elite_q{int(CFG.MAIN_ELITE_Q*100)}"], "decision_class_main"] = "possible_elite"
        pred.loc[pred[f"confident_elite_q{int(CFG.MAIN_ELITE_Q*100)}"], "decision_class_main"] = "confident_elite"
        pred.loc[pred[f"positive_anomaly_{main_tag}"], "decision_class_main"] = "positive_conformal_anomaly"
        pred.loc[pred[f"negative_anomaly_{main_tag}"], "decision_class_main"] = "negative_conformal_anomaly"
        if f"confident_elite_adaptive_q{int(CFG.MAIN_ELITE_Q*100)}" in pred.columns:
            pred.loc[pred[f"confident_elite_adaptive_q{int(CFG.MAIN_ELITE_Q*100)}"], "decision_class_main"] = "adaptive_confident_elite"

        metrics: Dict[str, Any] = {
            "job_id": job_id,
            "target_key": target_key,
            "target_col": target_col,
            "feature_family": feature_family,
            "model_name": model_name,
            "split_type": split_type,
            "seed": seed,
            "group_col": group_col,
            "n_train": int(len(y_train)),
            "n_cal": int(len(y_cal)),
            "n_test": int(len(y_test)),
            "n_features": int(len(use_cols)),
            "fit_seconds": float(fit_seconds),
            "model_profile": CFG.MODEL_PROFILE,
            "n_jobs_configured": int(CFG.N_JOBS),
            "rf_n_estimators": int(CFG.RF_N_ESTIMATORS),
            "rf_max_depth": CFG.RF_MAX_DEPTH,
            "rf_min_samples_leaf": int(CFG.RF_MIN_SAMPLES_LEAF),
            "rf_max_samples": CFG.RF_MAX_SAMPLES,
            "hgb_max_iter": int(CFG.HGB_MAX_ITER),
            "hgb_min_samples_leaf": int(CFG.HGB_MIN_SAMPLES_LEAF),
            "elite_thresholds": {str(k): v for k, v in elite_thresholds.items()},
            "train_metrics": regression_metrics(y_train, yhat_train),
            "cal_metrics": regression_metrics(y_cal, yhat_cal),
        }
        for cov in CFG.COVERAGE_LEVELS:
            tag = int(cov * 100)
            metrics.update({f"test_{k}": v for k, v in compute_screening_metrics(pred, elite_thresholds, cov).items()})
            if f"L_adaptive_{tag}" in pred.columns and f"U_adaptive_{tag}" in pred.columns:
                adaptive_metrics = compute_screening_metrics_from_bounds(
                    pred,
                    elite_thresholds,
                    cov,
                    lower_col=f"L_adaptive_{tag}",
                    upper_col=f"U_adaptive_{tag}",
                    prefix="adaptive",
                )
                metrics.update({f"test_{k}": v for k, v in adaptive_metrics.items()})

        # Conditional coverage table.
        cond_cov = conditional_coverage_by_group(pred, group_col, CFG.MAIN_COVERAGE)
        if not cond_cov.empty:
            cond_path = PATHS["metrics"] / f"conditional_coverage__{job_id}.csv"
            safe_to_csv(cond_cov, cond_path)
            metrics["conditional_coverage_path"] = str(cond_path)

        pred_to_save = pred[compact_prediction_columns(pred)].copy()
        safe_to_csv(pred_to_save, pred_path)
        if getattr(CFG, "SAVE_MODEL_OBJECTS", False):
            try:
                check_disk_space(model_path, context=f"saving model {model_path.name}")
                joblib.dump(pipe, model_path, compress=3)
            except Exception as e:
                LOGGER.warning("MODEL_SAVE_FAILED | %s | %s", job_id, e)
        save_json(metrics, metrics_path)
        mark_stage_done(f"job_{job_id}", [pred_path, metrics_path])
        LOGGER.info("JOB_DONE | %s | n_train=%d n_cal=%d n_test=%d rmse=%.4g cov90=%.3f", job_id, len(y_train), len(y_cal), len(y_test), metrics["test_rmse"], metrics.get("test_empirical_coverage", np.nan))
        return {"job_id": job_id, "metrics": metrics, "pred_path": pred_path}
    except Exception as e:
        msg = str(e)
        if "LOW_DISK_SPACE" in msg or "No space left" in msg or isinstance(e, RuntimeError):
            # Disk exhaustion should stop the outer loop immediately; otherwise
            # the script would continue launching jobs and repeatedly fail while
            # also trying to write more logs.
            raise
        LOGGER.error("JOB_FAILED | %s | %s", job_id, e)
        LOGGER.error(traceback.format_exc())
        return None


def stage_run_all_models(df: pd.DataFrame, target_mapping: Dict[str, Optional[str]], families: Dict[str, List[str]]) -> pd.DataFrame:
    stage = "run_all_models_v3_targetqc_adaptive"
    out_csv = PATHS["metrics"] / "all_job_metrics.csv"
    out_pkl = PATHS["metrics"] / "all_job_metrics.pkl"

    if is_stage_done(stage, [out_csv, out_pkl]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        return pd.read_pickle(out_pkl)

    LOGGER.info("STAGE >>> RUN_ALL_MODELS")
    # Recompute feature roles on the engineered df.
    engineered_df = add_engineered_geometry_features(df)
    feature_roles = identify_feature_columns(engineered_df, target_mapping)
    all_results = []
    available_targets = {k: v for k, v in target_mapping.items() if v is not None}
    if not available_targets:
        raise ValueError(
            "No target columns detected after building the modelling table. "
            "This usually means the adsorption-label CSV files were not discovered, "
            "their identifiers could not be joined, or the labels are stored in a "
            "long/unknown format not recognised by the automatic detector. Inspect "
            f"{PATHS['processed'] / 'master_table_selection_scores.csv'} and "
            f"{PATHS['processed'] / 'target_detection_candidate_scores.csv'}; also confirm "
            "that --data_root points to the folder containing data_raw\\arc_mof. "
            "If needed, rerun with --force_rerun after correcting --data_root."
        )

    partial_metrics_csv = PATHS["metrics"] / "all_job_metrics_partial.csv"
    job_counter = 0
    check_disk_space(PATHS["results"], context="starting model stage")

    for target_key, target_col in available_targets.items():
        for family_name, feature_cols in families.items():
            if family_name == "process_enhanced_optional" and not getattr(CFG, "RUN_PROCESS_ENHANCED_OPTIONAL", False):
                continue
            if family_name not in CFG.FEATURE_FAMILIES_TO_RUN and family_name != "process_enhanced_optional":
                continue
            for model_name in CFG.MODELS_TO_RUN:
                for split_type in CFG.SPLIT_TYPES_TO_RUN:
                    for seed in CFG.SEEDS[:CFG.N_SEEDS]:
                        job_counter += 1
                        if job_counter % max(1, int(getattr(CFG, "DISK_CHECK_EVERY_N_JOBS", 1))) == 0:
                            check_disk_space(PATHS["results"], context=f"before model job {job_counter}")
                        res = run_single_job(engineered_df, target_key, target_col, family_name, feature_cols, model_name, split_type, seed, feature_roles)
                        if res is not None:
                            all_results.append(res["metrics"])
                            if len(all_results) % max(1, int(getattr(CFG, "METRICS_FLUSH_EVERY_N_JOBS", 5))) == 0:
                                safe_to_csv(pd.DataFrame(all_results), partial_metrics_csv)

    metrics_df = pd.DataFrame(all_results)
    safe_to_csv(metrics_df, out_csv)
    safe_to_pickle_df(metrics_df, out_pkl)
    save_latex_table(
        metrics_df.head(40),
        PATHS["tables_si"] / "table_si_all_job_metrics_preview.tex",
        caption="Preview of all model, feature, split and target jobs.",
        label="tab:si_all_jobs_preview",
    )
    mark_stage_done(stage, [out_csv, out_pkl], {"n_jobs_completed": len(metrics_df)})
    return metrics_df

# =============================================================================
# Aggregation, chemistry enrichment, and final tables
# =============================================================================

def prediction_file_paths(metrics_df: pd.DataFrame, target_key: Optional[str] = None) -> List[Tuple[pd.Series, Path]]:
    """Return available prediction CSV paths paired with their metrics row.

    This helper is used by memory-safe aggregation functions. It avoids creating
    a single enormous dataframe from every prediction CSV, which is unnecessary
    and can exceed RAM for full ARC-MOF-scale runs.
    """
    pairs: List[Tuple[pd.Series, Path]] = []
    if metrics_df.empty or "job_id" not in metrics_df.columns:
        return pairs
    for _, row in metrics_df.iterrows():
        job_id = row.get("job_id")
        if not isinstance(job_id, str):
            continue
        if target_key is not None and row.get("target_key") != target_key:
            continue
        path = find_existing_prediction_path(job_id)
        if path is not None and path.exists():
            pairs.append((row, path))
    return pairs


def read_prediction_csv_light(path: Path, wanted_cols: Sequence[str]) -> pd.DataFrame:
    """Read only selected columns from a prediction CSV.

    pandas.read_csv(usecols=...) fails if some columns are absent, so this helper
    first inspects the header and then reads only the intersection. This keeps
    aggregation and figure generation memory-safe.
    """
    try:
        header = pd.read_csv(path, nrows=0)
        cols = [c for c in wanted_cols if c in header.columns]
        if not cols:
            return pd.DataFrame()
        return pd.read_csv(path, usecols=cols, low_memory=False)
    except Exception as e:
        LOGGER.warning("READ_PRED_LIGHT_FAILED | %s | %s", path, e)
        return pd.DataFrame()


def read_all_predictions(
    metrics_df: pd.DataFrame,
    target_key: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """Read prediction CSVs, preferably with a restricted column list.

    This function remains for backwards compatibility with figure code, but it
    should not be used to concatenate every full prediction file in a large run.
    For full ARC-MOF aggregation, use the streaming helpers below.
    """
    frames = []
    pairs = prediction_file_paths(metrics_df, target_key=target_key)
    if max_files is not None:
        pairs = pairs[:max_files]
    for _, path in pairs:
        try:
            if columns is None:
                frames.append(pd.read_csv(path, low_memory=False))
            else:
                frames.append(read_prediction_csv_light(path, columns))
        except Exception as e:
            LOGGER.warning("READ_PRED_FAILED | %s | %s", path, e)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    group_cols = ["target_key", "feature_family", "model_name", "split_type"]
    value_cols = [c for c in metrics_df.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(metrics_df[c])]
    rows = []
    for keys, sub in metrics_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub else int(len(sub))
        for c in value_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[c + "_mean"] = float(vals.mean())
            row[c + "_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_shortlist_stability_from_files(metrics_df: pd.DataFrame, coverage: float = 0.90) -> pd.DataFrame:
    """Memory-safe Jaccard stability of shortlists across seeds.

    The previous implementation concatenated all prediction CSVs into one very
    large dataframe before grouping. For full ARC-MOF runs this can involve tens
    of millions of rows. This implementation reads one prediction file at a time,
    extracts only the small top-k ID sets needed for stability, and immediately
    discards the dataframe.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    Lcol_standard = f"L_{int(coverage * 100)}"
    Lcol_adaptive = f"L_adaptive_{int(coverage * 100)}"
    fractions = [0.01, 0.05, 0.10]
    # Nested dictionary:
    # (target, family, model, split) -> frac -> seed -> {"pred": set, "lcb": set}
    shortlists: Dict[Tuple[Any, ...], Dict[float, Dict[Any, Dict[str, set]]]] = {}

    wanted = ["display_id", "y_pred", Lcol_standard, Lcol_adaptive]
    for row, path in prediction_file_paths(metrics_df):
        setup_key = (
            row.get("target_key"),
            row.get("feature_family"),
            row.get("model_name"),
            row.get("split_type"),
        )
        seed = row.get("seed")
        df = read_prediction_csv_light(path, wanted)
        if df.empty or "display_id" not in df.columns or "y_pred" not in df.columns:
            continue
        n = len(df)
        if n == 0:
            continue
        df["display_id"] = df["display_id"].astype(str)
        # Numeric coercion is cheap here and protects against occasional string
        # serialization quirks in CSV outputs.
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        lcb_col = Lcol_adaptive if Lcol_adaptive in df.columns else Lcol_standard
        if lcb_col in df.columns:
            df[lcb_col] = pd.to_numeric(df[lcb_col], errors="coerce")
        for frac in fractions:
            k = max(1, int(math.ceil(frac * n)))
            setup_store = shortlists.setdefault(setup_key, {}).setdefault(frac, {})
            pred_ids = set(df.nlargest(k, "y_pred")["display_id"].tolist())
            if lcb_col in df.columns:
                lcb_ids = set(df.nlargest(k, lcb_col)["display_id"].tolist())
            else:
                lcb_ids = set()
            setup_store[seed] = {"pred": pred_ids, "lcb": lcb_ids}

    rows = []
    for setup_key, frac_dict in shortlists.items():
        for frac, seed_dict in frac_dict.items():
            seed_values = sorted(seed_dict.keys())
            pairs = [(a, b) for i, a in enumerate(seed_values) for b in seed_values[i + 1:]]
            pred_j = [jaccard(seed_dict[a].get("pred", set()), seed_dict[b].get("pred", set())) for a, b in pairs]
            lcb_j = [jaccard(seed_dict[a].get("lcb", set()), seed_dict[b].get("lcb", set())) for a, b in pairs]
            row = dict(zip(["target_key", "feature_family", "model_name", "split_type"], setup_key))
            row.update({
                "top_fraction": frac,
                "n_seed_pairs": len(pairs),
                "jaccard_pred_mean": float(np.nanmean(pred_j)) if pred_j else np.nan,
                "jaccard_lcb_mean": float(np.nanmean(lcb_j)) if lcb_j else np.nan,
                "jaccard_pred_std": float(np.nanstd(pred_j)) if pred_j else np.nan,
                "jaccard_lcb_std": float(np.nanstd(lcb_j)) if lcb_j else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def select_representative_prediction_file_light(metrics_df: pd.DataFrame, target_key: Optional[str] = None) -> Optional[Path]:
    """Choose one representative prediction file without loading predictions."""
    if metrics_df.empty:
        return None
    sub = metrics_df.copy()
    if target_key is not None and "target_key" in sub.columns:
        sub = sub[sub["target_key"] == target_key].copy()
    if sub.empty:
        return None
    score = np.zeros(len(sub), dtype=float)
    if "target_key" in sub.columns:
        score += (sub["target_key"] == CFG.MAIN_TARGET_KEY).astype(float).values * 10
    if "feature_family" in sub.columns:
        score += (sub["feature_family"] == "geometry_topology_clusters").astype(float).values * 5
        score += (sub["feature_family"] == "process_enhanced_optional").astype(float).values * 2
    if "model_name" in sub.columns:
        score += (sub["model_name"].isin(["hgb", "rf"])).astype(float).values * 3
    if "split_type" in sub.columns:
        score += (sub["split_type"].isin(["random", "topology_grouped"])).astype(float).values * 2
    if "test_empirical_coverage" in sub.columns:
        score += -np.abs(pd.to_numeric(sub["test_empirical_coverage"], errors="coerce").fillna(0).values - CFG.MAIN_COVERAGE)
    sub["_score"] = score
    for _, row in sub.sort_values("_score", ascending=False).iterrows():
        job_id = row.get("job_id")
        if isinstance(job_id, str):
            path = find_existing_prediction_path(job_id)
            if path is not None and path.exists():
                return path
    return None


def compute_target_anomaly_rates_from_files(metrics_df: pd.DataFrame, coverage: float = 0.90) -> pd.Series:
    """Compute target-wise positive-anomaly rates without concatenating files."""
    label = f"positive_anomaly_{int(coverage * 100)}"
    counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    for row, path in prediction_file_paths(metrics_df):
        target = str(row.get("target_key"))
        df = read_prediction_csv_light(path, [label])
        if df.empty or label not in df.columns:
            continue
        vals = df[label].astype(bool)
        counts[target] = counts.get(target, 0) + int(vals.sum())
        totals[target] = totals.get(target, 0) + int(len(vals))
    rates = {k: counts.get(k, 0) / max(v, 1) for k, v in totals.items()}
    return pd.Series(rates).sort_values(ascending=False)


def enrichment_table(pred_df: pd.DataFrame, label_col: str, group_cols: Sequence[str], min_count: int = 10) -> pd.DataFrame:
    """Compute enrichment odds ratios for anomaly/elite labels over categorical groups."""
    rows = []
    if pred_df.empty or label_col not in pred_df.columns:
        return pd.DataFrame()
    label = pred_df[label_col].astype(bool)
    for gc in group_cols:
        if gc not in pred_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(pred_df[gc]):
            # Bin numeric pore descriptors for pore-regime enrichment.
            try:
                groups = pd.qcut(pd.to_numeric(pred_df[gc], errors="coerce"), q=5, duplicates="drop").astype(str)
            except Exception:
                continue
        else:
            groups = pred_df[gc].astype(str).fillna("missing")
        for g, idx in groups.groupby(groups).groups.items():
            in_group = pred_df.index.isin(idx)
            a = int((label & in_group).sum())
            b = int((~label & in_group).sum())
            c = int((label & ~in_group).sum())
            d = int((~label & ~in_group).sum())
            if a + b < min_count:
                continue
            # Haldane-Anscombe correction for stable odds ratios.
            odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            try:
                _, p = stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                p = np.nan
            rows.append({
                "label_col": label_col,
                "group_col": gc,
                "group": str(g),
                "n_group": int(a + b),
                "n_label_in_group": int(a),
                "label_rate_in_group": float(a / max(a + b, 1)),
                "label_rate_out_group": float(c / max(c + d, 1)),
                "odds_ratio": float(odds_ratio),
                "fisher_p": float(p) if np.isfinite(p) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["odds_ratio", "n_label_in_group"], ascending=[False, False])
    return out


def stage_aggregate_outputs(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    stage = "aggregate_outputs_v3_targetqc_adaptive"
    out_summary = PATHS["metrics"] / "summary_metrics_by_setup.csv"
    out_stability = PATHS["metrics"] / "shortlist_stability.csv"
    out_enrich = PATHS["metrics"] / "chemistry_enrichment_main_target.csv"

    if is_stage_done(stage, [out_summary, out_stability, out_enrich]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        return {
            "summary": pd.read_csv(out_summary),
            "stability": pd.read_csv(out_stability),
            "enrichment": pd.read_csv(out_enrich),
        }

    LOGGER.info("STAGE >>> AGGREGATE_OUTPUTS")
    summary = summarize_metrics(metrics_df)
    stability = compute_shortlist_stability_from_files(metrics_df, CFG.MAIN_COVERAGE)

    # Chemistry enrichment is computed from one representative main-target job by
    # default. That avoids duplicating every MOF across many targets/seeds/models,
    # keeps memory small, and is scientifically clearer for a case-study table.
    rep_path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY)
    main_pred = pd.read_csv(rep_path, low_memory=False) if rep_path is not None else pd.DataFrame()
    group_cols = []
    for c in main_pred.columns if not main_pred.empty else []:
        cslug = slugify(c)
        if any(h in cslug for h in TOPOLOGY_HINTS + CLUSTER_HINTS + ["density", "asa", "ava", "avaf", "pld", "lcd", "di", "df", "dif"]):
            group_cols.append(c)
    label_col = f"positive_anomaly_{int(CFG.MAIN_COVERAGE * 100)}"
    enrichment = enrichment_table(main_pred, label_col, group_cols[:20]) if not main_pred.empty else pd.DataFrame()
    if not enrichment.empty and rep_path is not None:
        enrichment.insert(0, "representative_prediction_file", rep_path.name)

    safe_to_csv(summary, out_summary)
    safe_to_csv(stability, out_stability)
    safe_to_csv(enrichment, out_enrich)
    safe_to_pickle_df(summary, PATHS["metrics"] / "summary_metrics_by_setup.pkl")
    safe_to_pickle_df(stability, PATHS["metrics"] / "shortlist_stability.pkl")
    safe_to_pickle_df(enrichment, PATHS["metrics"] / "chemistry_enrichment_main_target.pkl")

    # Main and SI tables.
    if not summary.empty:
        key_cols = ["target_key", "feature_family", "model_name", "split_type", "n_seeds"]
        metric_cols = [c for c in summary.columns if any(s in c for s in ["rmse_mean", "empirical_coverage_mean", "mean_interval_width_mean", "jaccard_lcb_vs_true_mean", "confident_false_elite_rate_mean"])]
        table = summary[key_cols + metric_cols[:10]].copy()
        safe_to_csv(table, PATHS["tables_main"] / "table_1_model_calibration_screening_summary.csv")
        save_latex_table(table.head(30), PATHS["tables_main"] / "table_1_model_calibration_screening_summary.tex", caption="Model accuracy, conformal coverage and screening reliability across targets, feature families and split types.", label="tab:model_calibration_screening")
    if not enrichment.empty:
        enrich_top = enrichment.head(30)
        safe_to_csv(enrich_top, PATHS["tables_main"] / "table_2_chemistry_enrichment_positive_anomalies.csv")
        save_latex_table(enrich_top, PATHS["tables_main"] / "table_2_chemistry_enrichment_positive_anomalies.tex", caption="Topology, pore-regime and cluster enrichments among conformal positive-surprise anomalies.", label="tab:chemistry_enrichment")
    if not stability.empty:
        save_latex_table(stability.head(50), PATHS["tables_si"] / "table_si_shortlist_stability.tex", caption="Seed-to-seed shortlist stability for uncalibrated and lower-confidence-bound rankings.", label="tab:si_stability")

    mark_stage_done(stage, [out_summary, out_stability, out_enrich])
    return {"summary": summary, "stability": stability, "enrichment": enrichment}

# =============================================================================
# Publication figures
# =============================================================================

def apply_publication_style() -> None:
    """Apply a clean, journal-style Matplotlib theme.

    The theme is deliberately conservative: no decorative backgrounds, no heavy
    chartjunk, colorblind-safe contrast, and consistent typography. It makes the
    output figures suitable for direct manuscript drafting while preserving all
    source data as CSV files for later Illustrator/Inkscape polishing.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": JOURNAL_COLORS["ink"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": JOURNAL_COLORS["muted_ink"],
        "ytick.color": JOURNAL_COLORS["muted_ink"],
        "axes.edgecolor": JOURNAL_COLORS["ink"],
        "axes.linewidth": 0.8,
        "grid.color": JOURNAL_COLORS["grid"],
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.prop_cycle": plt.cycler(color=list(COLOR_CYCLE)),
    })


def palette(n: int, offset: int = 0) -> List[str]:
    """Return n colors from the journal palette."""
    if n <= 0:
        return []
    return [COLOR_CYCLE[(i + offset) % len(COLOR_CYCLE)] for i in range(n)]


def bar_colors(labels: Sequence[Any], default_offset: int = 0) -> List[str]:
    """Assign semantically useful colors to common labels."""
    out = []
    for i, lab in enumerate(labels):
        s = slugify(lab)
        if "confident" in s or "certificate" in s or "adaptive" in s:
            out.append(JOURNAL_COLORS["teal"])
        elif "possible" in s:
            out.append(JOURNAL_COLORS["sky"])
        elif "positive" in s or "surprise" in s:
            out.append(JOURNAL_COLORS["orange"])
        elif "negative" in s or "false" in s or "fragile" in s:
            out.append(JOURNAL_COLORS["red"])
        elif "pred" in s or "uncalibrated" in s:
            out.append(JOURNAL_COLORS["blue"])
        else:
            out.append(COLOR_CYCLE[(i + default_offset) % len(COLOR_CYCLE)])
    return out


def save_current_figure(base_path: Path) -> List[Path]:
    """Save current Matplotlib figure in all requested formats with a white canvas."""
    outputs = []
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        try:
            ax.set_facecolor("white")
        except Exception:
            pass
    for fmt in CFG.FIG_FORMATS:
        path = base_path.with_suffix(f".{fmt}")
        ensure_dir(path.parent)
        plt.savefig(path, dpi=CFG.SAVEFIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="white")
        outputs.append(path)
    plt.close()
    return outputs


def style_axes(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Consistent axes styling for elegant composite figures."""
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=JOURNAL_COLORS["ink"], pad=8)
    ax.set_xlabel(xlabel, fontsize=8.8, color=JOURNAL_COLORS["ink"], labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8.8, color=JOURNAL_COLORS["ink"], labelpad=4)
    ax.tick_params(labelsize=7.4, colors=JOURNAL_COLORS["muted_ink"], width=0.7, length=3)
    ax.grid(True, alpha=0.55, linewidth=0.55, color=JOURNAL_COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color(JOURNAL_COLORS["ink"])
        ax.spines[side].set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a panel label in a small white box for robust visibility."""
    txt = ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=JOURNAL_COLORS["ink"],
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=JOURNAL_COLORS["light_grid"], linewidth=0.6),
        zorder=20,
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2g}", fontsize: float = 6.7) -> None:
    """Add compact value labels above bars when there are not too many bars."""
    patches = list(ax.patches)
    if len(patches) > 14:
        return
    ymax = max([p.get_height() for p in patches] + [0])
    if not np.isfinite(ymax) or ymax == 0:
        return
    for p in patches:
        h = p.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            h + 0.015 * ymax,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=JOURNAL_COLORS["muted_ink"],
        )


def clean_legend(ax: plt.Axes, loc: str = "best", ncol: int = 1) -> None:
    """Small frameless legend with consistent text color."""
    leg = ax.legend(loc=loc, fontsize=7, frameon=False, ncol=ncol)
    if leg is not None:
        for t in leg.get_texts():
            t.set_color(JOURNAL_COLORS["ink"])


def add_reference_line(ax: plt.Axes, y: Optional[float] = None, x: Optional[float] = None, label: str = "") -> None:
    """Add subtle horizontal/vertical reference line."""
    if y is not None:
        ax.axhline(y, linestyle="--", linewidth=1.0, color=JOURNAL_COLORS["muted_ink"], alpha=0.75, zorder=1)
        if label:
            ax.text(0.98, y, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=JOURNAL_COLORS["muted_ink"])
    if x is not None:
        ax.axvline(x, linestyle="--", linewidth=1.0, color=JOURNAL_COLORS["muted_ink"], alpha=0.75, zorder=1)
        if label:
            ax.text(x, 0.98, label, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.8, color=JOURNAL_COLORS["muted_ink"])


apply_publication_style()


def figure_1_conceptual_workflow() -> List[Path]:
    """Figure 1: conceptual workflow.

    This figure does not require data and is useful as a manuscript schematic.
    """
    fig, ax = plt.subplots(figsize=(12.2, 4.9), dpi=CFG.FIG_DPI)
    ax.axis("off")
    boxes = [
        ("Ordinary ML ranking", "Predict uptake and sort MOFs by score\nProblem: no calibrated risk certificate"),
        ("Conformal interval", "Use calibration residuals to form\n[L, U] around each prediction"),
        ("Confident elite", "Report only candidates whose lower bound\nexceeds the elite threshold"),
        ("Conformal anomaly", "Flag positive/negative residual surprises\nrelative to structural context"),
    ]
    xs = [0.05, 0.30, 0.55, 0.80]
    for i, ((title, body), x) in enumerate(zip(boxes, xs)):
        patch = FancyBboxPatch((x, 0.35), 0.18, 0.35, boxstyle="round,pad=0.02,rounding_size=0.025", linewidth=1.3, facecolor=[JOURNAL_COLORS["pale_blue"], JOURNAL_COLORS["pale_teal"], "#F6F3FF", JOURNAL_COLORS["pale_orange"]][i], edgecolor=COLOR_CYCLE[i])
        ax.add_patch(patch)
        ax.text(x + 0.09, 0.62, title, ha="center", va="center", fontsize=10, fontweight="bold", color=JOURNAL_COLORS["ink"])
        ax.text(x + 0.09, 0.48, body, ha="center", va="center", fontsize=8, color=JOURNAL_COLORS["muted_ink"])
        ax.text(x + 0.02, 0.75, f"({chr(97+i)})", fontsize=12, fontweight="bold", color=COLOR_CYCLE[i])
        if i < len(xs) - 1:
            arr = FancyArrowPatch((x + 0.19, 0.52), (xs[i+1] - 0.01, 0.52), arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color=JOURNAL_COLORS["muted_ink"])
            ax.add_patch(arr)
    ax.text(0.5, 0.12, "Central shift: from uncalibrated top-k lists to auditable material-discovery decisions", ha="center", fontsize=11, fontweight="bold", color=JOURNAL_COLORS["ink"])
    return save_current_figure(PATHS["figures_main"] / "Figure_1_conceptual_workflow")


def figure_2_calibration_efficiency(summary: pd.DataFrame, metrics_df: pd.DataFrame) -> List[Path]:
    """Figure 2: calibration as a screening-quality control, not a flat coverage plot."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()

    # Panel a: deviation from nominal coverage by target and split. This is more
    # informative than raw bars when all methods appear close to 0.90.
    ax = axes[0]
    if not metrics_df.empty and "test_empirical_coverage" in metrics_df:
        data = metrics_df.pivot_table(index="split_type", columns="target_key", values="test_empirical_coverage", aggfunc="mean") - CFG.MAIN_COVERAGE
        data.to_csv(PATHS["figure_data_main"] / "Figure_2a_coverage_deviation_by_split_target.csv")
        norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(data.values), vmax=np.nanmax(data.values)) if np.isfinite(data.values).any() and np.nanmin(data.values) < 0 < np.nanmax(data.values) else None
        im = ax.imshow(data.values, aspect="auto", cmap=COVERAGE_CMAP, norm=norm)
        ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(data.index, fontsize=7)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Coverage - nominal")
    style_axes(ax, "Where does nominal 90% coverage fail?", "Target", "Split")
    add_panel_label(ax, "a")

    # Panel b: efficiency heatmap. Width is target-specific, so keep targets explicit.
    ax = axes[1]
    width_col = "test_mean_interval_width"
    if not metrics_df.empty and width_col in metrics_df:
        data = metrics_df.pivot_table(index="split_type", columns="target_key", values=width_col, aggfunc="mean")
        data.to_csv(PATHS["figure_data_main"] / "Figure_2b_interval_width_by_split_target.csv")
        im = ax.imshow(np.log1p(data.values), aspect="auto", cmap=EFFICIENCY_CMAP)
        ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(data.index, fontsize=7)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + width)")
    style_axes(ax, "Efficiency cost of certification", "Target", "Split")
    add_panel_label(ax, "b")

    # Panel c: RMSE is not enough; coverage and accuracy move independently.
    ax = axes[2]
    if not metrics_df.empty and "test_rmse" in metrics_df and "test_empirical_coverage" in metrics_df:
        data = metrics_df.groupby(["model_name", "split_type"]).agg(
            rmse=("test_rmse", "mean"),
            coverage=("test_empirical_coverage", "mean"),
            width=("test_mean_interval_width", "mean") if "test_mean_interval_width" in metrics_df else ("test_rmse", "size"),
        ).reset_index()
        data.to_csv(PATHS["figure_data_main"] / "Figure_2c_rmse_coverage_screening_tradeoff.csv", index=False)
        sizes = 30 + 120 * (data["width"] / max(data["width"].max(), 1e-9))
        for split, sub in data.groupby("split_type"):
            ax.scatter(sub["rmse"], sub["coverage"], s=sizes.loc[sub.index], alpha=0.78, label=split, edgecolor="white", linewidth=0.6, zorder=3)
            for _, r in sub.iterrows():
                ax.text(r["rmse"], r["coverage"], str(r["model_name"]), fontsize=6)
        add_reference_line(ax, y=CFG.MAIN_COVERAGE, label="nominal")
        clean_legend(ax, ncol=1)
    style_axes(ax, "RMSE does not certify screening risk", "Mean RMSE", "Mean coverage")
    add_panel_label(ax, "c")

    # Panel d: standard versus adaptive conformal efficiency when available.
    ax = axes[3]
    if not metrics_df.empty and "test_adaptive_empirical_coverage" in metrics_df:
        data = metrics_df.groupby("split_type").agg(
            standard_width=("test_mean_interval_width", "mean"),
            adaptive_width=("test_adaptive_mean_interval_width", "mean"),
            standard_coverage=("test_empirical_coverage", "mean"),
            adaptive_coverage=("test_adaptive_empirical_coverage", "mean"),
        ).reset_index()
        data.to_csv(PATHS["figure_data_main"] / "Figure_2d_standard_vs_adaptive_conformal.csv", index=False)
        x = np.arange(len(data)); width = 0.35
        ax.bar(x - width/2, data["standard_width"], width=width, label="standard", color=JOURNAL_COLORS["blue"], edgecolor="white", linewidth=0.6)
        ax.bar(x + width/2, data["adaptive_width"], width=width, label="adaptive", color=JOURNAL_COLORS["teal"], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(data["split_type"], rotation=30, ha="right", fontsize=7)
        clean_legend(ax, ncol=1)
        ax2 = ax.twinx()
        ax2.plot(x, data["standard_coverage"], marker="o", linestyle="--", linewidth=1.2, color=JOURNAL_COLORS["blue"], label="standard cov.")
        ax2.plot(x, data["adaptive_coverage"], marker="s", linestyle=":", linewidth=1.4, color=JOURNAL_COLORS["teal"], label="adaptive cov.")
        ax2.axhline(CFG.MAIN_COVERAGE, linewidth=0.8, color=JOURNAL_COLORS["muted_ink"], linestyle="--", alpha=0.65)
        ax2.set_ylabel("Coverage", fontsize=8)
        ax2.tick_params(labelsize=7)
    elif not metrics_df.empty and "test_empirical_coverage" in metrics_df:
        pivot = metrics_df.pivot_table(index="split_type", columns="feature_family", values="test_empirical_coverage", aggfunc="mean")
        pivot.to_csv(PATHS["figure_data_main"] / "Figure_2d_coverage_by_split_feature.csv")
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_axes(ax, "Standard vs adaptive certificates", "Split", "Interval width")
    add_panel_label(ax, "d")
    fig.suptitle("Figure 2. Calibration is a materials-screening quality-control layer", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_2_calibration_efficiency_result_aware")


def figure_3_elite_reliability(metrics_df: pd.DataFrame, stability: pd.DataFrame) -> List[Path]:
    """Figure 3: elite screening as triage, not simple LCB reranking."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()
    qtag = int(CFG.MAIN_ELITE_Q * 100)
    recall_col = f"test_elite_q{qtag}_confident_recall"
    false_col = f"test_elite_q{qtag}_confident_false_elite_rate"
    pred_false_col = f"test_elite_q{qtag}_predicted_false_elite_rate"
    adaptive_recall_col = f"test_adaptive_elite_q{qtag}_confident_recall"
    adaptive_false_col = f"test_adaptive_elite_q{qtag}_confident_false_elite_rate"

    ax = axes[0]
    if recall_col in metrics_df:
        cols = [recall_col] + ([adaptive_recall_col] if adaptive_recall_col in metrics_df else [])
        data = metrics_df.groupby("split_type")[cols].mean()
        data.to_csv(PATHS["figure_data_main"] / "Figure_3a_confident_elite_recall_standard_adaptive.csv")
        x = np.arange(len(data)); width = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + (i - (len(cols)-1)/2) * width, data[c], width=width, label=c.replace("test_", "").replace("_", " "), color=COLOR_CYCLE[i], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=7)
        clean_legend(ax, ncol=1)
    style_axes(ax, "How many true elites can be certified?", "Split", "Recall")
    add_panel_label(ax, "a")

    ax = axes[1]
    cols = [c for c in [pred_false_col, false_col, adaptive_false_col] if c in metrics_df]
    if cols:
        data = metrics_df.groupby("split_type")[cols].mean()
        data.to_csv(PATHS["figure_data_main"] / "Figure_3b_false_elite_rates_triage.csv")
        x = np.arange(len(data)); width = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + (i - (len(cols)-1)/2) * width, data[c], width=width, label=c.replace("test_", "").replace("_", " "), color=COLOR_CYCLE[i], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=7)
        clean_legend(ax, ncol=1)
    style_axes(ax, "False-elite risk after certification", "Split", "False elite rate")
    add_panel_label(ax, "b")

    ax = axes[2]
    if not stability.empty:
        data = stability[stability["top_fraction"] == 0.05].groupby("split_type")[["jaccard_pred_mean", "jaccard_lcb_mean"]].mean()
        data.to_csv(PATHS["figure_data_main"] / "Figure_3c_shortlist_stability_pred_vs_adaptive_lcb.csv")
        x = np.arange(len(data)); width = 0.35
        ax.bar(x - width/2, data["jaccard_pred_mean"], width=width, label="predicted top-5%", color=JOURNAL_COLORS["blue"], edgecolor="white", linewidth=0.6)
        ax.bar(x + width/2, data["jaccard_lcb_mean"], width=width, label="adaptive/LCB top-5%", color=JOURNAL_COLORS["teal"], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=7)
        clean_legend(ax, ncol=1)
    style_axes(ax, "Seed-to-seed shortlist fragility", "Split", "Jaccard")
    add_panel_label(ax, "c")

    ax = axes[3]
    pred_j = "test_top5_jaccard_pred_vs_true"
    adaptive_j = "test_adaptive_top5_jaccard_lcb_vs_true"
    if pred_j in metrics_df:
        cols = [pred_j] + ([adaptive_j] if adaptive_j in metrics_df else [])
        data = metrics_df.groupby("model_name")[cols].mean()
        data.to_csv(PATHS["figure_data_main"] / "Figure_3d_top5_recovery_by_model.csv")
        x = np.arange(len(data)); width = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + (i - (len(cols)-1)/2) * width, data[c], width=width, label=c.replace("test_", "").replace("_", " "), color=COLOR_CYCLE[i], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(data.index, rotation=20, ha="right", fontsize=8)
        clean_legend(ax, ncol=1)
    elif "test_top5_ndcg_pred" in metrics_df:
        data = metrics_df.groupby("model_name")["test_top5_ndcg_pred"].mean()
        data.to_csv(PATHS["figure_data_main"] / "Figure_3d_ndcg_pred_only.csv")
        ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=20, ha="right")
    style_axes(ax, "Top-5% recovery: ranking versus certificate", "Model", "Overlap with true top-5%")
    add_panel_label(ax, "d")

    fig.suptitle("Figure 3. Elite MOF discovery becomes a risk-controlled triage problem", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_3_elite_screening_result_aware")


def select_representative_prediction_file(metrics_df: pd.DataFrame) -> Optional[Path]:
    """Choose a strong representative job for case-study figures."""
    if metrics_df.empty:
        return None
    sub = metrics_df.copy()
    # Prefer main target, geometry+topology+clusters, hgb/rf, random or topology grouped.
    score = np.zeros(len(sub))
    score += (sub.get("target_key") == CFG.MAIN_TARGET_KEY).astype(float).values * 10
    score += (sub.get("feature_family") == "geometry_topology_clusters").astype(float).values * 5
    score += (sub.get("model_name").isin(["hgb", "rf"])).astype(float).values * 3
    score += (sub.get("split_type").isin(["random", "topology_grouped"])).astype(float).values * 2
    if "test_empirical_coverage" in sub:
        score += -np.abs(pd.to_numeric(sub["test_empirical_coverage"], errors="coerce").fillna(0).values - CFG.MAIN_COVERAGE)
    sub["_score"] = score
    row = sub.sort_values("_score", ascending=False).iloc[0]
    path = find_existing_prediction_path(row["job_id"])
    return path if path is not None and path.exists() else None


def figure_4_anomaly_atlas(metrics_df: pd.DataFrame) -> List[Path]:
    path = select_representative_prediction_file(metrics_df)
    pred = pd.read_csv(path, low_memory=False) if path else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=CFG.FIG_DPI)
    axes = axes.ravel()
    tag = int(CFG.MAIN_COVERAGE * 100)

    ax = axes[0]
    if not pred.empty:
        data = pred.sort_values("residual", ascending=False).head(50)[["display_id", "residual", "p_positive_surprise"]]
        data.to_csv(PATHS["figure_data_main"] / "Figure_4a_positive_surprise_top50.csv", index=False)
        ax.scatter(range(len(data)), data["residual"].values, s=22, color=JOURNAL_COLORS["orange"], edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3)
        add_reference_line(ax, y=pred[f"q_pos_{tag}"].iloc[0], label="90% threshold")
    style_axes(ax, "Positive-surprise residuals", "Rank among top residuals", "Residual")
    add_panel_label(ax, "a")

    ax = axes[1]
    if not pred.empty:
        data = pred.sort_values("residual", ascending=True).head(50)[["display_id", "residual", "p_negative_surprise"]]
        data.to_csv(PATHS["figure_data_main"] / "Figure_4b_negative_surprise_top50.csv", index=False)
        ax.scatter(range(len(data)), data["residual"].values, s=22, color=JOURNAL_COLORS["orange"], edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3)
        add_reference_line(ax, y=-pred[f"q_neg_{tag}"].iloc[0], label="90% threshold")
    for coll in ax.collections:
        coll.set_facecolor(JOURNAL_COLORS["blue"])
    style_axes(ax, "Negative-surprise residuals", "Rank among low residuals", "Residual")
    add_panel_label(ax, "b")

    ax = axes[2]
    if not pred.empty:
        data = pred[["p_positive_surprise", "p_negative_surprise", "p_two_sided_surprise"]].copy()
        data.to_csv(PATHS["figure_data_main"] / "Figure_4c_anomaly_pvalues.csv", index=False)
        ax.hist(data["p_two_sided_surprise"].dropna().values, bins=30, alpha=0.88, color=JOURNAL_COLORS["purple"], edgecolor="white", linewidth=0.45)
        add_reference_line(ax, x=1 - CFG.MAIN_COVERAGE, label="alpha")
    style_axes(ax, "Two-sided conformal p-values", "p-value", "Count")
    add_panel_label(ax, "c")

    ax = axes[3]
    data = compute_target_anomaly_rates_from_files(metrics_df, CFG.MAIN_COVERAGE)
    if not data.empty:
        data.to_csv(PATHS["figure_data_main"] / "Figure_4d_target_anomaly_rates.csv")
        ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=30, ha="right")
    style_axes(ax, "Target-specific anomaly rate", "Target", "Fraction positive anomalies")
    add_panel_label(ax, "d")
    fig.suptitle("Figure 4. Conformal anomaly atlas", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_4_conformal_anomaly_atlas")


def _candidate_slug_variants(display_id: Any) -> List[str]:
    raw = str(display_id)
    variants = [raw, raw.replace(".cif", ""), slugify(raw), slugify(raw.replace(".cif", ""))]
    # Preserve order and uniqueness.
    out: List[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


def find_structure_image(display_id: Any) -> Optional[Path]:
    """Find a rendered structure image for a candidate, if the user provides one.

    Expected locations, relative to the script folder, include:
    structure_images/, structures_rendered/, figures/structures/, and
    data/structures_rendered/. Filenames can be either the display_id or its
    slugified form, with .png/.jpg/.jpeg/.tif/.tiff extensions.
    """
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    dirs = [CFG.PROJECT_ROOT / sub for sub in CFG.STRUCTURE_IMAGE_SUBDIRS]
    for d in dirs:
        if not d.exists():
            continue
        for stem in _candidate_slug_variants(display_id):
            for ext in exts:
                path = d / f"{stem}{ext}"
                if path.exists():
                    return path
    return None


def draw_structure_slot(ax: plt.Axes, display_id: Any, role: str, subtitle: str = "") -> None:
    """Draw either a rendered structure image or a high-IF-ready placeholder."""
    ax.axis("off")
    img_path = find_structure_image(display_id)
    if img_path is not None:
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(role, fontsize=8, fontweight="bold")
            ax.text(0.02, 0.02, str(display_id)[:35], transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.3))
            return
        except Exception:
            pass
    # Placeholder: publication team should replace these with PyMOL/VESTA/OVITO
    # renders from the matching ARC-MOF CIFs.
    rect = FancyBboxPatch((0.06, 0.12), 0.88, 0.72, boxstyle="round,pad=0.025,rounding_size=0.03",
                          linewidth=1.2, edgecolor=JOURNAL_COLORS["blue"], facecolor=JOURNAL_COLORS["pale_blue"])
    ax.add_patch(rect)
    ax.text(0.50, 0.68, role, ha="center", va="center", fontsize=8, fontweight="bold", color=JOURNAL_COLORS["ink"])
    ax.text(0.50, 0.50, "Add rendered\nMOF structure here", ha="center", va="center", fontsize=8)
    ax.text(0.50, 0.31, str(display_id)[:42], ha="center", va="center", fontsize=6, family="monospace")
    if subtitle:
        ax.text(0.50, 0.18, subtitle[:46], ha="center", va="center", fontsize=6)


def choose_structural_cases(pred: pd.DataFrame, tag: int, qtag: int) -> pd.DataFrame:
    """Select candidate structures for chemistry-facing case-study panels."""
    if pred.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    base_cols = [c for c in [
        "display_id", "target_key", "target_col", "feature_family", "model_name", "split_type", "seed",
        "y_true", "y_pred", "residual", f"L_{tag}", f"U_{tag}", f"L_adaptive_{tag}", f"U_adaptive_{tag}",
        "p_positive_surprise", "p_negative_surprise", "p_two_sided_surprise", "decision_class_main",
    ] if c in pred.columns]

    def add_case(label: str, sub: pd.DataFrame, sort_col: str, ascending: bool = True, n: int = 4):
        if sub.empty or sort_col not in sub.columns:
            return
        tmp = sub.sort_values(sort_col, ascending=ascending).head(n).copy()
        tmp["structural_case_role"] = label
        rows.append(tmp)

    add_case("positive-surprise anomaly", pred[pred.get(f"positive_anomaly_{tag}", False).astype(bool)] if f"positive_anomaly_{tag}" in pred else pd.DataFrame(), "p_positive_surprise", True, CFG.N_STRUCTURAL_CASES_PER_CLASS)
    add_case("negative-surprise anomaly", pred[pred.get(f"negative_anomaly_{tag}", False).astype(bool)] if f"negative_anomaly_{tag}" in pred else pd.DataFrame(), "p_negative_surprise", True, CFG.N_STRUCTURAL_CASES_PER_CLASS)
    if f"confident_elite_adaptive_q{qtag}" in pred.columns:
        add_case("adaptive confident elite", pred[pred[f"confident_elite_adaptive_q{qtag}"].astype(bool)], f"L_adaptive_{tag}", False, CFG.N_STRUCTURAL_CASES_PER_CLASS)
    elif f"confident_elite_q{qtag}" in pred.columns:
        add_case("confident elite", pred[pred[f"confident_elite_q{qtag}"].astype(bool)], f"L_{tag}", False, CFG.N_STRUCTURAL_CASES_PER_CLASS)
    if f"predicted_elite_q{qtag}" in pred.columns and f"true_elite_q{qtag}" in pred.columns:
        false_or_uncertain = pred[pred[f"predicted_elite_q{qtag}"].astype(bool) & ~pred[f"true_elite_q{qtag}"].astype(bool)]
        add_case("uncalibrated false/fragile elite", false_or_uncertain, "y_pred", False, CFG.N_STRUCTURAL_CASES_PER_CLASS)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Keep duplicates once, but retain the first role assigned.
    out = out.drop_duplicates("display_id", keep="first")
    cols = ["structural_case_role"] + base_cols
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()
    out["structure_image_found"] = out["display_id"].map(lambda x: str(find_structure_image(x)) if find_structure_image(x) else "")
    out["recommended_structure_action"] = "Render ARC-MOF CIF as node/linker + pore-window view; include PLD/LCD/density/void fraction annotations."
    return out


def write_structural_case_manifest(pred: pd.DataFrame, tag: int, qtag: int) -> pd.DataFrame:
    """Write the candidate manifest for structural representations."""
    cases = choose_structural_cases(pred, tag, qtag)
    if not cases.empty:
        safe_to_csv(cases, PATHS["structural_cases"] / "structural_case_study_manifest.csv")
        save_latex_table(
            cases.head(16),
            PATHS["tables_main"] / "table_structural_case_study_candidates.tex",
            caption="Candidate MOFs selected for structural case-study rendering.",
            label="tab:structural_cases",
        )
    return cases


def figure_5_chemistry_of_anomalies(enrichment: pd.DataFrame, metrics_df: pd.DataFrame) -> List[Path]:
    """Figure 5: chemical regimes and structural case-study placeholders."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()

    ax = axes[0]
    if not enrichment.empty:
        data = enrichment.head(15).copy()
        data.to_csv(PATHS["figure_data_main"] / "Figure_5a_enrichment_topology_pore_cluster.csv", index=False)
        labels = (data["group_col"].astype(str).str[:12] + ":" + data["group"].astype(str).str[:18]).values
        y = np.arange(len(data))
        ax.barh(y, data["odds_ratio"].values)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        add_reference_line(ax, x=1.0, label="no enrichment")
        ax.invert_yaxis()
    style_axes(ax, "Which pore/topology/cluster regimes are enriched?", "Odds ratio", "")
    add_panel_label(ax, "a")

    path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY) or select_representative_prediction_file(metrics_df)
    pred = pd.read_csv(path, low_memory=False) if path else pd.DataFrame()
    tag = int(CFG.MAIN_COVERAGE * 100)
    qtag = int(CFG.MAIN_ELITE_Q * 100)

    ax = axes[1]
    if not pred.empty:
        numeric_context = [c for c in pred.columns if pd.api.types.is_numeric_dtype(pred[c]) and any(h in slugify(c) for h in ["density", "asa", "ava", "avaf", "pld", "lcd", "di", "df", "dif"])]
        # Prefer chemically interpretable porosity/window axes if available.
        preferred_x = next((c for c in numeric_context if any(h in slugify(c) for h in ["density", "avaf", "void"])), numeric_context[0] if numeric_context else None)
        preferred_y = next((c for c in numeric_context if any(h in slugify(c) for h in ["pld", "df", "lcd", "di", "asa"])), numeric_context[1] if len(numeric_context) > 1 else None)
        if preferred_x and preferred_y:
            data = pred[[preferred_x, preferred_y, f"positive_anomaly_{tag}", "residual"]].dropna().copy()
            data.to_csv(PATHS["figure_data_main"] / "Figure_5b_pore_regime_positive_anomalies.csv", index=False)
            ax.scatter(data[preferred_x], data[preferred_y], s=9, alpha=0.18, color=JOURNAL_COLORS["grey"], edgecolor="none", rasterized=True)
            pos = data[data[f"positive_anomaly_{tag}"].astype(bool)]
            ax.scatter(pos[preferred_x], pos[preferred_y], s=24, alpha=0.92, color=JOURNAL_COLORS["orange"], edgecolor="white", linewidth=0.45)
            ax.set_xlabel(preferred_x[:34], fontsize=8); ax.set_ylabel(preferred_y[:34], fontsize=8)
    style_axes(ax, "Positive surprises in pore-property space", "", "")
    add_panel_label(ax, "b")

    ax = axes[2]
    if not pred.empty:
        count_items: Dict[str, int] = {}
        for label, col in [
            ("confident elite", f"confident_elite_q{qtag}"),
            ("adaptive confident elite", f"confident_elite_adaptive_q{qtag}"),
            ("positive surprise", f"positive_anomaly_{tag}"),
            ("negative surprise", f"negative_anomaly_{tag}"),
            ("two-sided anomaly", f"two_sided_anomaly_{tag}"),
        ]:
            if col in pred.columns:
                count_items[label] = int(pred[col].astype(bool).sum())
        pd.Series(count_items).to_csv(PATHS["figure_data_main"] / "Figure_5c_candidate_class_counts.csv")
        ax.bar(range(len(count_items)), list(count_items.values()))
        ax.set_xticks(range(len(count_items))); ax.set_xticklabels(list(count_items.keys()), rotation=30, ha="right", fontsize=7)
    style_axes(ax, "How many materials receive certificates?", "Class", "Count")
    add_panel_label(ax, "c")

    # Panel d: structural representations belong here. Candidate IDs are saved
    # as a table/manifest; the panel displays rendered images if the user provides
    # them, otherwise publication-ready placeholders.
    ax = axes[3]
    ax.axis("off")
    cases = write_structural_case_manifest(pred, tag, qtag) if not pred.empty else pd.DataFrame()
    pos_cases = cases[cases["structural_case_role"].str.contains("positive", case=False, na=False)].head(4) if not cases.empty else pd.DataFrame()
    if pos_cases.empty and not cases.empty:
        pos_cases = cases.head(4)
    subaxes = [ax.inset_axes([0.02 + 0.49*(i % 2), 0.52 - 0.49*(i // 2), 0.46, 0.43]) for i in range(4)]
    for i, sax in enumerate(subaxes):
        if i < len(pos_cases):
            r = pos_cases.iloc[i]
            subtitle = f"resid={r.get('residual', np.nan):.2g}; p+={r.get('p_positive_surprise', np.nan):.2g}"
            draw_structure_slot(sax, r["display_id"], "positive-surprise case", subtitle)
        else:
            draw_structure_slot(sax, "candidate CIF", "structure slot", "render selected MOF")
    ax.text(0.02, 0.02, "Replace placeholders with ARC-MOF CIF renders: node/linker view + pore-window view.", transform=ax.transAxes, fontsize=7)
    add_panel_label(ax, "d")

    fig.suptitle("Figure 5. Conformal anomalies expose chemically interpretable MOF regimes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_5_chemistry_structural_cases")


def figure_6_screening_consequence(metrics_df: pd.DataFrame) -> List[Path]:
    """Figure 6: decision consequences and structural contrast cases."""
    path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY) or select_representative_prediction_file(metrics_df)
    pred = pd.read_csv(path, low_memory=False) if path else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()
    tag = int(CFG.MAIN_COVERAGE * 100)
    qtag = int(CFG.MAIN_ELITE_Q * 100)

    ax = axes[0]
    if not pred.empty:
        counts: Dict[str, int] = {}
        for label, col in [
            ("predicted elite", f"predicted_elite_q{qtag}"),
            ("possible elite", f"possible_elite_q{qtag}"),
            ("confident elite", f"confident_elite_q{qtag}"),
            ("adaptive confident", f"confident_elite_adaptive_q{qtag}"),
            ("positive surprise", f"positive_anomaly_{tag}"),
            ("negative surprise", f"negative_anomaly_{tag}"),
        ]:
            if col in pred.columns:
                counts[label] = int(pred[col].astype(bool).sum())
        pd.Series(counts).to_csv(PATHS["figure_data_main"] / "Figure_6a_screening_triage_counts.csv")
        ax.bar(range(len(counts)), list(counts.values()), color=bar_colors(counts.keys()), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(counts))); ax.set_xticklabels(list(counts.keys()), rotation=30, ha="right", fontsize=7)
    style_axes(ax, "How calibration changes screening output", "Decision class", "Candidates")
    add_panel_label(ax, "a")

    ax = axes[1]
    if not pred.empty:
        Lcol = f"L_adaptive_{tag}" if f"L_adaptive_{tag}" in pred.columns else f"L_{tag}"
        Ucol = f"U_adaptive_{tag}" if f"U_adaptive_{tag}" in pred.columns else f"U_{tag}"
        data = pred[["y_pred", Lcol, Ucol, "y_true", "decision_class_main"]].dropna().sample(min(900, len(pred)), random_state=CFG.RANDOM_SEED)
        data.to_csv(PATHS["figure_data_main"] / "Figure_6b_uncertainty_decision_map_sample.csv", index=False)
        ax.errorbar(data["y_pred"], data["y_true"], xerr=[data["y_pred"] - data[Lcol], data[Ucol] - data["y_pred"]], fmt="o", markersize=2.2, alpha=0.24, linewidth=0.35, color=JOURNAL_COLORS["blue"], ecolor=JOURNAL_COLORS["sky"], markeredgecolor="white", markeredgewidth=0.25)
        lims = [min(data["y_pred"].min(), data["y_true"].min()), max(data["y_pred"].max(), data["y_true"].max())]
        ax.plot(lims, lims, linestyle="--", linewidth=0.9, color=JOURNAL_COLORS["muted_ink"], alpha=0.85)
    style_axes(ax, "Prediction, truth and certificate width", "Predicted uptake with interval", "True uptake")
    add_panel_label(ax, "b")

    ax = axes[2]
    if not pred.empty and f"predicted_elite_q{qtag}" in pred.columns and f"true_elite_q{qtag}" in pred.columns:
        rows = []
        for label, col in [
            ("uncalibrated predicted elite", f"predicted_elite_q{qtag}"),
            ("standard confident elite", f"confident_elite_q{qtag}"),
            ("adaptive confident elite", f"confident_elite_adaptive_q{qtag}"),
        ]:
            if col in pred.columns:
                selected = pred[col].astype(bool)
                false_rate = float((selected & ~pred[f"true_elite_q{qtag}"].astype(bool)).sum() / max(selected.sum(), 1))
                recall = float((selected & pred[f"true_elite_q{qtag}"].astype(bool)).sum() / max(pred[f"true_elite_q{qtag}"].astype(bool).sum(), 1))
                rows.append({"gate": label, "false_elite_rate": false_rate, "elite_recall": recall})
        data = pd.DataFrame(rows)
        if not data.empty:
            data.to_csv(PATHS["figure_data_main"] / "Figure_6c_risk_gate_tradeoff.csv", index=False)
            x = np.arange(len(data)); width = 0.35
            ax.bar(x - width/2, data["false_elite_rate"], width=width, label="false elite rate")
            ax.bar(x + width/2, data["elite_recall"], width=width, label="elite recall")
            ax.set_xticks(x); ax.set_xticklabels(data["gate"], rotation=30, ha="right", fontsize=7)
            clean_legend(ax, ncol=1)
    style_axes(ax, "Risk gates trade recall for reliability", "Gate", "Rate")
    add_panel_label(ax, "c")

    ax = axes[3]
    ax.axis("off")
    cases = write_structural_case_manifest(pred, tag, qtag) if not pred.empty else pd.DataFrame()
    roles = ["adaptive confident", "positive-surprise", "negative-surprise", "false/fragile"]
    subaxes = [ax.inset_axes([0.02 + 0.49*(i % 2), 0.52 - 0.49*(i // 2), 0.46, 0.43]) for i in range(4)]
    for i, sax in enumerate(subaxes):
        chosen = pd.DataFrame()
        if not cases.empty:
            role_pattern = roles[i].split("-")[0].split("/")[0]
            chosen = cases[cases["structural_case_role"].str.contains(role_pattern, case=False, na=False)].head(1)
            if chosen.empty and i < len(cases):
                chosen = cases.iloc[[i]]
        if not chosen.empty:
            r = chosen.iloc[0]
            subtitle = f"{r.get('structural_case_role', '')}; resid={r.get('residual', np.nan):.2g}"
            draw_structure_slot(sax, r["display_id"], roles[i], subtitle)
        else:
            draw_structure_slot(sax, "candidate CIF", roles[i], "render selected MOF")
    ax.text(0.02, 0.02, "Use this panel as the structural contrast figure: certified elite vs positive/negative surprise vs fragile top-k candidate.", transform=ax.transAxes, fontsize=7)
    add_panel_label(ax, "d")

    # Tables, not figure panels, should contain candidate lists.
    if not pred.empty:
        Lcol = f"L_adaptive_{tag}" if f"L_adaptive_{tag}" in pred.columns else f"L_{tag}"
        cols = [c for c in ["display_id", "y_true", "y_pred", Lcol, f"U_adaptive_{tag}", f"U_{tag}", "decision_class_main", "p_positive_surprise", "p_negative_surprise", "p_two_sided_surprise"] if c in pred.columns]
        pred.sort_values(Lcol, ascending=False).head(25)[cols].to_csv(PATHS["tables_main"] / "table_4_certificate_screening_shortlist_main_target.csv", index=False)
        save_latex_table(pred.sort_values(Lcol, ascending=False).head(15)[cols], PATHS["tables_main"] / "table_4_certificate_screening_shortlist_main_target.tex", caption="Certificate-ranked screening shortlist for the main target.", label="tab:certificate_shortlist")

    fig.suptitle("Figure 6. Screening decisions become chemically auditable certificates", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_6_screening_consequence_structural")


def create_si_figures(metrics_df: pd.DataFrame) -> List[Path]:
    outputs = []
    if metrics_df.empty:
        return outputs
    # SI Figure S1: target-wise performance summary.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=CFG.FIG_DPI)
    if "test_rmse" in metrics_df:
        data = metrics_df.groupby(["target_key", "split_type"])["test_rmse"].mean().unstack("split_type")
        data.to_csv(PATHS["figure_data_si"] / "Figure_S1_target_split_rmse.csv")
        data.plot(kind="bar", ax=ax)
        style_axes(ax, "SI Figure S1. Target-wise RMSE by split", "Target", "RMSE")
        clean_legend(ax, ncol=1)
    outputs.extend(save_current_figure(PATHS["figures_si"] / "Figure_S1_target_split_rmse"))

    # SI Figure S2: interval width by target and split.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=CFG.FIG_DPI)
    if "test_mean_interval_width" in metrics_df:
        data = metrics_df.groupby(["target_key", "split_type"])["test_mean_interval_width"].mean().unstack("split_type")
        data.to_csv(PATHS["figure_data_si"] / "Figure_S2_target_interval_width.csv")
        data.plot(kind="bar", ax=ax)
        style_axes(ax, "SI Figure S2. Interval width by target", "Target", "Mean width")
        clean_legend(ax, ncol=1)
    outputs.extend(save_current_figure(PATHS["figures_si"] / "Figure_S2_target_interval_width"))

    # SI Figure S3: coverage distribution.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=CFG.FIG_DPI)
    if "test_empirical_coverage" in metrics_df:
        metrics_df[["test_empirical_coverage"]].to_csv(PATHS["figure_data_si"] / "Figure_S3_coverage_distribution.csv", index=False)
        ax.hist(metrics_df["test_empirical_coverage"].dropna(), bins=25)
        ax.axvline(CFG.MAIN_COVERAGE, linestyle="--", linewidth=1)
        style_axes(ax, "SI Figure S3. Distribution of empirical coverage", "Coverage", "Count")
    outputs.extend(save_current_figure(PATHS["figures_si"] / "Figure_S3_coverage_distribution"))
    return outputs


def stage_make_figures(metrics_df: pd.DataFrame, aggregates: Dict[str, pd.DataFrame]) -> List[Path]:
    stage = "make_all_figures_v3_highif_structural"
    sentinel = PATHS["figures"] / "figure_generation_complete.txt"
    if is_stage_done(stage, [sentinel]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        return []

    LOGGER.info("STAGE >>> MAKE_FIGURES")
    outputs = []
    outputs.extend(figure_1_conceptual_workflow())
    outputs.extend(figure_2_calibration_efficiency(aggregates.get("summary", pd.DataFrame()), metrics_df))
    outputs.extend(figure_3_elite_reliability(metrics_df, aggregates.get("stability", pd.DataFrame())))
    outputs.extend(figure_4_anomaly_atlas(metrics_df))
    outputs.extend(figure_5_chemistry_of_anomalies(aggregates.get("enrichment", pd.DataFrame()), metrics_df))
    outputs.extend(figure_6_screening_consequence(metrics_df))
    outputs.extend(create_si_figures(metrics_df))
    sentinel.write_text("Figure generation completed at " + now(), encoding="utf-8")
    mark_stage_done(stage, [sentinel] + outputs, {"n_outputs": len(outputs)})
    return outputs



# =============================================================================
# Paper-production export helpers
# =============================================================================

def write_feature_and_target_tables(
    df: pd.DataFrame,
    target_mapping: Dict[str, Optional[str]],
    families: Dict[str, List[str]],
) -> List[Path]:
    """Write manuscript/SI-ready tables describing targets and feature families.

    These tables are deliberately generated before modelling so that the paper can
    report exactly what the script detected from the user's local CSV files. They
    also make debugging much easier when column names differ between ARC-MOF
    downloads or user-merged tables.
    """
    outputs: List[Path] = []

    target_rows = []
    for key, col in target_mapping.items():
        info = TARGET_DEFINITIONS.get(key, {})
        if col is None or col not in df.columns:
            target_rows.append({
                "target_key": key,
                "target_description": info.get("short", key),
                "detected_column": "NOT DETECTED",
                "n_non_missing": 0,
                "missing_fraction": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
            })
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        target_rows.append({
            "target_key": key,
            "target_description": info.get("short", key),
            "detected_column": col,
            "n_non_missing": int(y.notna().sum()),
            "missing_fraction": float(y.isna().mean()),
            "mean": float(y.mean()) if y.notna().any() else np.nan,
            "std": float(y.std()) if y.notna().sum() > 1 else np.nan,
            "min": float(y.min()) if y.notna().any() else np.nan,
            "median": float(y.median()) if y.notna().any() else np.nan,
            "max": float(y.max()) if y.notna().any() else np.nan,
        })
    target_df = pd.DataFrame(target_rows)
    p = PATHS["tables_si"] / "table_si_detected_targets_and_availability.csv"
    safe_to_csv(target_df, p)
    outputs.append(p)
    ptex = PATHS["tables_si"] / "table_si_detected_targets_and_availability.tex"
    save_latex_table(target_df, ptex, caption="Detected adsorption targets and target availability in the local ARC-MOF-style input tables.", label="tab:si_detected_targets")
    outputs.append(ptex)

    family_rows = []
    for fam, cols in families.items():
        for c in cols:
            if c not in df.columns:
                continue
            family_rows.append({
                "feature_family": fam,
                "column": c,
                "dtype": str(df[c].dtype),
                "n_non_missing": int(df[c].notna().sum()),
                "missing_fraction": float(df[c].isna().mean()),
                "n_unique": int(df[c].nunique(dropna=True)),
            })
    family_df = pd.DataFrame(family_rows)
    p = PATHS["tables_si"] / "table_si_feature_family_columns.csv"
    safe_to_csv(family_df, p)
    outputs.append(p)
    ptex = PATHS["tables_si"] / "table_si_feature_family_columns_preview.tex"
    save_latex_table(family_df.head(80), ptex, caption="Feature-family definitions used by the paper-production pipeline. The CSV file contains the complete list.", label="tab:si_feature_families")
    outputs.append(ptex)
    return outputs


def write_publication_manifest() -> Path:
    """Create a single CSV inventory of reusable outputs for manuscript writing."""
    rows = []
    wanted_roots = [
        PATHS["tables_main"], PATHS["tables_si"],
        PATHS["figures_main"], PATHS["figures_si"],
        PATHS["figure_data_main"], PATHS["figure_data_si"],
        PATHS["metrics"], PATHS["processed"], PATHS["audit"],
        PATHS["paper_b"], PATHS["download_guides"],
    ]
    for root in wanted_roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rows.append({
                    "category": root.name,
                    "relative_path": str(p.relative_to(PATHS["results"])),
                    "absolute_path": str(p),
                    "suffix": p.suffix.lower(),
                    "size_kb": round(p.stat().st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
    manifest = pd.DataFrame(rows)
    out = PATHS["results"] / "PUBLICATION_OUTPUT_MANIFEST.csv"
    safe_to_csv(manifest, out)
    return out


def write_minimal_environment_file() -> Path:
    """Write an environment note with the exact core packages needed.

    This is not a lockfile, but it is enough for an undergraduate student to
    recreate a working VS Code environment before running the pipeline.
    """
    out = PATHS["results"] / "PYTHON_ENVIRONMENT_NOTES.txt"
    content = f"""
Conformal MOF anomaly-screening pipeline: Python environment notes
=================================================================
Generated: {now()}

Recommended Python version
--------------------------
Python 3.10, 3.11 or 3.12. Python 3.11 is a safe choice for scikit-learn.

Install packages
----------------
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn matplotlib joblib openpyxl tqdm jinja2 jinja2

Run in VS Code / Visual Studio
------------------------------
1. Put this .py file in the same folder as the ARC-MOF CSV files, or place the
   CSV files under data_raw/arc_mof/.
2. Open the folder in VS Code.
3. Select the Python interpreter.
4. Run: python {Path(__file__).name}
5. Optional processor and model-size control:
   python {Path(__file__).name} --n_jobs 2 --model_profile balanced
   python {Path(__file__).name} --n_jobs 4 --model_profile comprehensive
   python {Path(__file__).name} --n_jobs 2 --model_profile fast --fast_test_rows 10000

Model profiles
--------------
fast:     quick smoke test; not intended for final paper claims.
balanced: default manuscript-friendly setting; recommended first full run.
comprehensive: heavier final robustness run if runtime permits. The old name paper is accepted as an alias.

Restart/resume behaviour
------------------------
The script writes stage manifests under results/manifests/. If interrupted, run
again; completed stages and completed model jobs are skipped unless FORCE_RERUN
is set to True in the Config dataclass.
""".strip()
    out.write_text(content, encoding="utf-8")
    return out

# =============================================================================
# Paper-B impact additions: external plausibility, process/stability triage,
# candidate casebooks and mechanistic positive-surprise classes
# =============================================================================

PAPER_B_FILE_REQUIREMENTS: List[Dict[str, str]] = [
    {"folder": "data_raw/arc_mof/clean_data", "file": "clean_data.csv", "status": "required", "use": "Master ARC-MOF table containing material identifiers and adsorption labels."},
    {"folder": "data_raw/arc_mof/descriptors", "file": "geometric_properties.csv", "status": "required", "use": "PLD, LCD, density, accessible surface area, pore volume and void-fraction descriptors used for process proxies and chemical interpretation."},
    {"folder": "data_raw/arc_mof/topology", "file": "all_topology_lists.csv", "status": "strongly recommended", "use": "Topology labels for topology-grouped validation and topology-surprise interpretation."},
    {"folder": "data_raw/arc_mof/clusters", "file": "geo-clusters.csv", "status": "strongly recommended", "use": "Geometry-family holdouts and pore-regime interpretation."},
    {"folder": "data_raw/arc_mof/clusters", "file": "mc-clusters.csv", "status": "strongly recommended", "use": "Metal-chemistry clusters for chemistry-grouped validation and metal-node surprise annotation."},
    {"folder": "data_raw/arc_mof/clusters", "file": "func-clusters.csv", "status": "strongly recommended", "use": "Functional-group clusters for functional chemistry holdouts and positive-surprise families."},
    {"folder": "data_raw/arc_mof/clusters", "file": "flig-clusters.csv", "status": "strongly recommended", "use": "Functional-ligand/linker clusters for chemistry holdouts and case-study tables."},
    {"folder": "data_raw/arc_mof/adsorption", "file": "post_comb_vsa-CO2.csv", "status": "required for CO2 Paper-B story", "use": "CO2 low-pressure adsorption targets, especially 0.015 and 0.15 bar."},
    {"folder": "data_raw/arc_mof/adsorption", "file": "methane.csv", "status": "recommended", "use": "CH4 5.8 and 65 bar cross-target sanity checks."},
    {"folder": "data_raw/arc_mof/process", "file": "overall_process.csv", "status": "optional but valuable", "use": "If joinable, supplies process-level working-capacity/selectivity/proxy columns."},
    {"folder": "data_raw/arc_mof/structures", "file": "ARCMOF_20241004.tar.gz", "status": "recommended for final figures", "use": "CIF source for rendering real structure panels that replace placeholders."},
    {"folder": "data_raw/core_mof_2024/metadata", "file": "ASR_data_SI_20250204.csv", "status": "Paper-B overlay", "use": "Experimental-origin activated-solvent-removed metadata and stability/provenance columns where available."},
    {"folder": "data_raw/core_mof_2024/metadata", "file": "FSR_data_SI_20250204.csv", "status": "Paper-B overlay", "use": "Experimental-origin free-solvent-removed metadata and activation-state contrasts."},
    {"folder": "data_raw/core_mof_2024/metadata", "file": "ION_data_SI_20250204.csv", "status": "Paper-B overlay", "use": "Ionic CoRE subset/provenance metadata."},
    {"folder": "data_raw/core_mof_2024/metadata", "file": "12089-recommended-screening-list.csv", "status": "Paper-B overlay", "use": "Recommended experimentally grounded screening subset for candidate plausibility labels."},
    {"folder": "data_raw/core_mof_2024/duplicate_checks", "file": "ASR_FSR_check.csv", "status": "optional", "use": "Duplicate/activation-state awareness for candidate case studies."},
    {"folder": "data_raw/core_mof_2024/water_gemc", "file": "water.zip", "status": "optional", "use": "Hydrophobicity/water-stability support when extracted to CSV by the user."},
    {"folder": "data_raw/core_mof_2024/tsa", "file": "TSA.zip", "status": "optional", "use": "Optional thermal/process/stability descriptors if extracted to CSV by the user."},
    {"folder": "data_raw/core_mof_2024/mofid", "file": "mofid-v2.zip", "status": "optional", "use": "MOFid/MOFkey identity support for conservative cross-database matching if extracted."},
    {"folder": "data_raw/mosaec_db", "file": "mosaec-db.csv or mosaec-db.xlsx", "status": "optional restricted overlay", "use": "Experimental plausibility/provenance annotation where the user has access."},
    {"folder": "data_raw/mosaec_db/descriptors", "file": "GEOM_mosaec-db.csv", "status": "optional restricted overlay", "use": "Experimental-database geometry overlay for case-study plausibility."},
    {"folder": "data_raw/mofchecker", "file": "mofchecker_results.csv", "status": "optional user-generated", "use": "Basic structural/chemical sanity flags for promoted candidates."},
]


def _case_safe_numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _find_columns_by_hints(df: pd.DataFrame, hints: Sequence[str], numeric_only: bool = False, max_cols: int = 10) -> List[str]:
    """Find dataframe columns whose slug contains any of the requested hints."""
    hits: List[str] = []
    for c in df.columns:
        if c in hits:
            continue
        slug = slugify(c)
        if any(h in slug for h in hints):
            if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().sum() < max(5, int(0.01 * len(df))):
                    continue
            if is_probably_target(c):
                continue
            hits.append(c)
            if len(hits) >= max_cols:
                break
    return hits


def _first_context_col(df: pd.DataFrame, hints: Sequence[str]) -> Optional[str]:
    hits = _find_columns_by_hints(df, hints, numeric_only=False, max_cols=1)
    return hits[0] if hits else None


def _normalised_identifier_values_from_row(row: pd.Series, preferred_cols: Optional[Sequence[str]] = None) -> set:
    keys: set = set()
    cols = list(preferred_cols or [])
    cols.extend([c for c in row.index if any(h in slugify(c) for h in ID_HINTS + ["mofkey", "mof_id", "refcode", "database_code"])])
    cols.extend(["display_id", "__join_id__", "filename", "name", "Name", "mofid", "MOFid", "mofkey", "MOFkey"])
    for c in cols:
        if c in row.index:
            val = row.get(c)
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if not sval or sval.lower() in {"nan", "none", "null"}:
                continue
            keys.add(normalize_identifier_series(pd.Series([sval])).iloc[0])
            keys.add(slugify(sval))
    return {k for k in keys if k}


def read_optional_table(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read CSV/XLSX files for optional overlays; archives are reported but not parsed."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return read_csv_robust(path, nrows=nrows)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=nrows)
    except Exception as e:
        LOGGER.warning("OPTIONAL_TABLE_READ_FAILED | %s | %s", path, e)
    return None




def resolve_project_or_data_folder(folder: Path) -> Path:
    """Resolve requirement folders against DATA_ROOT when they start with data/.

    Older documentation used project-root-relative paths such as data/raw/arc_mof/v7_zenodo_16802743.
    In this v2 script, DATA_ROOT itself is the data folder, so data/raw/... should
    resolve to DATA_ROOT/raw/....
    """
    folder = Path(folder)
    if folder.is_absolute():
        return folder
    parts = folder.parts
    if parts and parts[0].lower() in {"data", "data_raw"}:
        # If folder starts with data_raw, paths are relative to the project root;
        # if it starts with data/raw from old docs, resolve to the normalised DATA_ROOT.
        if parts[0].lower() == "data_raw":
            return Path(CFG.PROJECT_ROOT).joinpath(*parts)
        return Path(CFG.DATA_ROOT).joinpath(*parts[1:])
    # Try DATA_ROOT first because this pipeline works from the raw archive root.
    candidate = Path(CFG.DATA_ROOT) / folder
    if candidate.exists():
        return candidate
    return Path(CFG.PROJECT_ROOT) / folder

def find_existing_requirement_file(folder: Path, filename: str) -> Optional[Path]:
    """Find a required/recommended file, allowing nested folders and csv/xlsx alternatives."""
    if " or " in filename:
        names = [x.strip() for x in filename.split(" or ")]
    else:
        names = [filename]
    root = resolve_project_or_data_folder(folder)
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
        if root.exists():
            for pth in root.rglob(name):
                if pth.exists():
                    return pth
    if root.exists():
        for pth in root.rglob("*"):
            if not pth.is_file():
                continue
            stem_slug = slugify(pth.name)
            for name in names:
                name_slug = slugify(name.replace(" or ", " "))
                if name_slug and name_slug in stem_slug:
                    return pth
    return None


def write_paper_b_input_file_check() -> pd.DataFrame:
    """Write an explicit file-status table for the downloads requested by Paper B."""
    rows: List[Dict[str, Any]] = []
    for req in PAPER_B_FILE_REQUIREMENTS:
        folder = Path(req["folder"])
        found = find_existing_requirement_file(folder, req["file"])
        rows.append({
            "folder": req["folder"],
            "file_to_download_or_prepare": req["file"],
            "status": req["status"],
            "analysis_use": req["use"],
            "found": bool(found),
            "found_path": str(found) if found else "",
        })
    df = pd.DataFrame(rows)
    safe_to_csv(df, PATHS["download_guides"] / "paper_b_input_file_check.csv")
    save_latex_table(
        df[["folder", "file_to_download_or_prepare", "status", "analysis_use", "found"]],
        PATHS["download_guides"] / "paper_b_input_file_check.tex",
        caption="Input files requested for the Paper-B conformal-certificate and anomaly-screening enhancements.",
        label="tab:paper_b_input_files",
    )
    return df


def _collect_overlay_tables(root: Path, source_name: str) -> List[Tuple[str, pd.DataFrame]]:
    """Collect reasonably sized optional CSV/XLSX overlay tables from a source folder."""
    tables: List[Tuple[str, pd.DataFrame]] = []
    if not root.exists():
        return tables
    for path in sorted(list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + list(root.rglob("*.xls"))):
        if path.stat().st_size > 500 * 1024 * 1024:
            LOGGER.warning("OPTIONAL_TABLE_SKIPPED_TOO_LARGE | %s", path)
            continue
        df = read_optional_table(path)
        if df is None or df.empty:
            continue
        tables.append((f"{source_name}:{path.relative_to(root)}", df))
    return tables


def _build_identifier_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Map normalised IDs/MOFids/MOFkeys/refcodes to row numbers in an overlay table."""
    id_cols = [c for c in df.columns if any(h in slugify(c) for h in ID_HINTS + ["mofkey", "mof_id", "refcode", "database_code", "csd"])]
    if not id_cols:
        id_col = choose_id_column(df)
        id_cols = [id_col] if id_col else []
    index: Dict[str, List[int]] = {}
    if not id_cols:
        return index
    for i, row in df[id_cols].iterrows():
        keys = _normalised_identifier_values_from_row(row, id_cols)
        for k in keys:
            index.setdefault(k, []).append(int(i))
    return index


def _summarise_overlay_match(df: pd.DataFrame, indices: Sequence[int]) -> Dict[str, Any]:
    """Extract compact stability/provenance text from matched overlay rows."""
    if df.empty or not indices:
        return {}
    clean_indices = list(dict.fromkeys([i for i in indices if 0 <= i < len(df)]))
    sub = df.iloc[clean_indices]
    if sub.empty:
        return {}
    interesting = []
    for c in sub.columns:
        slug = slugify(c)
        if any(h in slug for h in ["stability", "stable", "water", "hydro", "activation", "solvent", "decomp", "temperature", "oms", "metal", "mofid", "mofkey", "doi", "refcode", "asr", "fsr", "ion", "recommend"]):
            vals = sub[c].dropna().astype(str).unique().tolist()[:3]
            if vals:
                interesting.append(f"{c}=" + "/".join(vals))
    return {"matched_rows": int(len(sub)), "metadata_summary": "; ".join(interesting[:8])}


def annotate_external_plausibility(cases: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Annotate candidate cases with CoRE/MOSAEC/MOFChecker matches when available."""
    if cases.empty or not getattr(CFG, "ENABLE_EXTERNAL_PLAUSIBILITY_OVERLAY", True):
        return cases

    overlay_sources = {
        "core_mof": PATHS.get("raw_core_mof", CFG.PROJECT_ROOT / CFG.CORE_MOF_SUBDIR),
        "mosaec": PATHS.get("raw_mosaec", CFG.PROJECT_ROOT / CFG.MOSAEC_SUBDIR),
        "mofchecker": PATHS.get("raw_mofchecker", CFG.PROJECT_ROOT / CFG.MOFCHECKER_SUBDIR),
    }
    loaded: Dict[str, List[Tuple[str, pd.DataFrame, Dict[str, List[int]]]]] = {}
    for source, root in overlay_sources.items():
        loaded[source] = []
        for table_name, table in _collect_overlay_tables(Path(root), source):
            idx = _build_identifier_index(table)
            if idx:
                loaded[source].append((table_name, table, idx))
        LOGGER.info("PAPER_B_OVERLAY_TABLES | %s | %d indexed tables", source, len(loaded[source]))

    out = cases.copy()
    for source in overlay_sources:
        out[f"{source}_match"] = False
        out[f"{source}_matched_tables"] = ""
        out[f"{source}_metadata_summary"] = ""

    for ridx, row in out.iterrows():
        keys = _normalised_identifier_values_from_row(row)
        if "row_index" in row.index and pd.notna(row.get("row_index")):
            ix = int(row.get("row_index"))
            if 0 <= ix < len(merged):
                keys |= _normalised_identifier_values_from_row(merged.iloc[ix])
        for source, tables in loaded.items():
            matched_tables: List[str] = []
            summaries: List[str] = []
            for table_name, table, idx in tables:
                matched_indices: List[int] = []
                for k in keys:
                    matched_indices.extend(idx.get(k, []))
                if matched_indices:
                    matched_tables.append(table_name)
                    summary = _summarise_overlay_match(table, matched_indices)
                    if summary.get("metadata_summary"):
                        summaries.append(summary["metadata_summary"])
            if matched_tables:
                out.at[ridx, f"{source}_match"] = True
                out.at[ridx, f"{source}_matched_tables"] = "; ".join(matched_tables[:5])
                out.at[ridx, f"{source}_metadata_summary"] = " | ".join(summaries[:3])

    def provenance_label(r: pd.Series) -> str:
        if bool(r.get("core_mof_match", False)) or bool(r.get("mosaec_match", False)):
            return "external experimental/plausibility match"
        if bool(r.get("mofchecker_match", False)):
            return "structure-check available, no external experimental match"
        return "ARC-MOF/hypothetical or unmatched in optional overlays"

    out["provenance_plausibility_status"] = out.apply(provenance_label, axis=1)
    return out


def add_context_to_cases(cases: pd.DataFrame, merged: pd.DataFrame, target_mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Add geometry/topology/cluster/target columns to case-study candidates."""
    if cases.empty:
        return cases
    out = cases.copy()
    context_specs = {
        "density": ["density"],
        "pld_or_limiting_diameter": ["pld", "limiting_pore", "df", "dif"],
        "lcd_or_largest_cavity": ["lcd", "largest_cavity", "di"],
        "accessible_surface_area": ["asa", "surface_area"],
        "accessible_volume_or_pore_volume": ["ava", "pore_volume", "void_volume"],
        "void_fraction": ["avaf", "void_fraction", "porosity"],
        "topology_label": ["topology", "topo", "rcsr", "net"],
        "geometry_cluster": ["geo_cluster", "geo", "cluster"],
        "metal_cluster": ["mc_cluster", "metal_cluster", "metal"],
        "functional_cluster": ["func_cluster", "functional"],
        "ligand_cluster": ["flig_cluster", "ligand", "linker"],
    }
    resolved_cols = {name: _first_context_col(merged, hints) for name, hints in context_specs.items()}

    for new_name, col in resolved_cols.items():
        values = []
        for _, row in out.iterrows():
            val = np.nan
            if "row_index" in row.index and pd.notna(row.get("row_index")):
                ix = int(row.get("row_index"))
                if col and 0 <= ix < len(merged) and col in merged.columns:
                    val = merged.iloc[ix][col]
            values.append(val)
        out[new_name] = values
        out[f"source_col__{new_name}"] = col or ""

    for key, col in target_mapping.items():
        if col is None or col not in merged.columns:
            continue
        vals = []
        qvals = []
        series = pd.to_numeric(merged[col], errors="coerce")
        for _, row in out.iterrows():
            val = np.nan
            qval = np.nan
            if "row_index" in row.index and pd.notna(row.get("row_index")):
                ix = int(row.get("row_index"))
                if 0 <= ix < len(merged):
                    val = pd.to_numeric(pd.Series([merged.iloc[ix][col]]), errors="coerce").iloc[0]
                    if np.isfinite(val):
                        qval = float((series <= val).mean())
            vals.append(val)
            qvals.append(qval)
        out[f"target_value__{key}"] = vals
        out[f"target_quantile__{key}"] = qvals
    return out


def add_process_stability_triage(cases: pd.DataFrame) -> pd.DataFrame:
    """Add transparent post-certification process/stability proxy labels."""
    if cases.empty or not getattr(CFG, "ENABLE_PROCESS_STABILITY_TRIAGE", True):
        return cases
    out = cases.copy()
    statuses: List[str] = []
    reasons: List[str] = []
    for _, r in out.iterrows():
        checks: List[Tuple[str, bool]] = []
        density = _case_safe_numeric(r.get("density"))
        vf = _case_safe_numeric(r.get("void_fraction"))
        pld = _case_safe_numeric(r.get("pld_or_limiting_diameter"))
        if np.isfinite(density):
            checks.append(("density window", CFG.PROCESS_PROXY_MIN_DENSITY <= density <= CFG.PROCESS_PROXY_MAX_DENSITY))
        if np.isfinite(vf):
            checks.append(("void-fraction window", CFG.PROCESS_PROXY_MIN_VOID_FRACTION <= vf <= CFG.PROCESS_PROXY_MAX_VOID_FRACTION))
        if np.isfinite(pld):
            checks.append(("PLD window", CFG.PROCESS_PROXY_MIN_PLD_A <= pld <= CFG.PROCESS_PROXY_MAX_PLD_A))
        if len(checks) < 2:
            statuses.append("insufficient process-proxy descriptors")
        elif all(v for _, v in checks):
            statuses.append("no obvious process-proxy red flag")
        else:
            statuses.append("process-proxy warning")
        reasons.append("; ".join([f"{name}={'pass' if val else 'warn'}" for name, val in checks]) or "no descriptor checks available")
    out["process_proxy_status"] = statuses
    out["process_proxy_reason"] = reasons

    stability_status: List[str] = []
    for _, r in out.iterrows():
        text = " ".join(str(r.get(c, "")) for c in out.columns if any(h in slugify(c) for h in ["stability", "stable", "water", "hydro", "activation", "decomp", "mofchecker", "core_mof_metadata", "mosaec_metadata"]))
        if "fail" in text.lower() or "unstable" in text.lower() or "warning" in text.lower():
            stability_status.append("possible stability/validity warning")
        elif text.strip():
            stability_status.append("external stability/provenance metadata available")
        else:
            stability_status.append("no external stability metadata available")
    out["stability_proxy_status"] = stability_status

    def cross_target_label(row: pd.Series) -> str:
        co2_cols = [c for c in out.columns if c.startswith("target_quantile__co2")]
        ch4_cols = [c for c in out.columns if c.startswith("target_quantile__ch4")]
        co2 = [float(row.get(c)) for c in co2_cols if pd.notna(row.get(c))]
        ch4 = [float(row.get(c)) for c in ch4_cols if pd.notna(row.get(c))]
        if len(co2) >= 2 and min(co2) >= 0.80:
            return "CO2-consistent high-quantile candidate"
        if len(co2) >= 2 and max(co2) >= 0.90 and min(co2) < 0.50:
            return "cross-pressure CO2 tension"
        if ch4 and max(ch4) >= 0.90:
            return "also high-CH4; check selectivity/process context"
        if co2 and max(co2) >= 0.80:
            return "single-CO2-target high candidate"
        return "cross-target evidence limited or ordinary"

    out["cross_target_sanity"] = out.apply(cross_target_label, axis=1)
    return out


def add_mechanistic_positive_surprise_family(cases: pd.DataFrame) -> pd.DataFrame:
    """Classify positive-surprise candidates into review-friendly mechanistic families."""
    if cases.empty:
        return cases
    out = cases.copy()
    labels: List[str] = []
    explanations: List[str] = []
    for _, r in out.iterrows():
        role = str(r.get("paper_b_case_class", r.get("structural_case_role", ""))).lower()
        vf = _case_safe_numeric(r.get("void_fraction"))
        pld = _case_safe_numeric(r.get("pld_or_limiting_diameter"))
        metal = str(r.get("metal_cluster", ""))
        topology = str(r.get("topology_label", ""))
        func = str(r.get("functional_cluster", ""))
        if "negative" in role:
            labels.append("negative-surprise / overprediction regime")
            explanations.append("Model expectation is higher than the held-out uptake; inspect blocked pores, overly generic descriptors or unstable/implausible structures.")
        elif "fragile" in role:
            labels.append("fragile elite after calibration")
            explanations.append("High predicted rank is not supported by the lower confidence bound; use as a cautionary contrast rather than a promoted candidate.")
        elif "positive" in role or bool(r.get(f"positive_anomaly_{int(CFG.MAIN_COVERAGE*100)}", False)):
            if np.isfinite(pld) and pld <= 4.0:
                labels.append("confined micropore surprise")
                explanations.append("Narrow windows may create overlapping adsorption potentials that simple global descriptors underpredict.")
            elif np.isfinite(vf) and vf <= 0.20:
                labels.append("dense / low-void surprise")
                explanations.append("The candidate performs better than expected despite low accessible volume, suggesting local binding or pore-shape effects.")
            elif metal and metal.lower() not in {"nan", "", "none"}:
                labels.append("metal-node / chemistry surprise")
                explanations.append("Metal-cluster chemistry may encode local adsorption interactions not fully captured by the model features.")
            elif topology and topology.lower() not in {"nan", "", "none"}:
                labels.append("topology surprise")
                explanations.append("Network architecture appears to contribute beyond global geometry descriptors.")
            elif func and func.lower() not in {"nan", "", "none"}:
                labels.append("functional-group surprise")
                explanations.append("Functional chemistry may create stronger local adsorption environments than expected.")
            else:
                labels.append("underencoded-local-environment surprise")
                explanations.append("Candidate is a statistically strong positive residual but needs structural inspection before a chemical mechanism is claimed.")
        elif "confident" in role:
            labels.append("certified high-uptake elite")
            explanations.append("Lower confidence bound remains above the elite threshold; promote only after plausibility/stability checks.")
        elif "possible" in role:
            labels.append("possible elite requiring follow-up")
            explanations.append("Upper bound or prediction supports interest, but evidence is not strong enough for a confident certificate.")
        else:
            labels.append("ordinary contrast candidate")
            explanations.append("Used as a high-rank or structural contrast for interpreting certified and anomalous cases.")
    out["positive_surprise_mechanistic_family"] = labels
    out["case_interpretation_short"] = explanations
    return out


def select_paper_b_casebook(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select 8--60 representative Paper-B candidate cases from a main-target job."""
    path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY) or select_representative_prediction_file(metrics_df)
    if path is None:
        return pd.DataFrame()
    pred = pd.read_csv(path, low_memory=False)
    if pred.empty:
        return pd.DataFrame()
    tag = int(CFG.MAIN_COVERAGE * 100)
    qtag = int(CFG.MAIN_ELITE_Q * 100)
    n = int(getattr(CFG, "PAPER_B_CASES_PER_CLASS", 4))
    selections: List[pd.DataFrame] = []

    def add(label: str, mask: pd.Series, sort_col: str, ascending: bool, limit: int = n) -> None:
        if sort_col not in pred.columns:
            return
        sub = pred[mask.fillna(False) if hasattr(mask, "fillna") else mask].copy()
        if sub.empty:
            return
        sub = sub.sort_values(sort_col, ascending=ascending).head(limit).copy()
        sub["paper_b_case_class"] = label
        sub["representative_prediction_file"] = path.name
        selections.append(sub)

    conf_col = f"confident_elite_adaptive_q{qtag}" if f"confident_elite_adaptive_q{qtag}" in pred.columns else f"confident_elite_q{qtag}"
    poss_col = f"possible_elite_q{qtag}"
    pred_col = f"predicted_elite_q{qtag}"
    true_col = f"true_elite_q{qtag}"
    lcb_col = f"L_adaptive_{tag}" if f"L_adaptive_{tag}" in pred.columns else f"L_{tag}"

    if conf_col in pred.columns:
        add("confident elite", pred[conf_col].astype(bool), lcb_col, False)
    if poss_col in pred.columns and conf_col in pred.columns:
        add("possible but not confident elite", pred[poss_col].astype(bool) & ~pred[conf_col].astype(bool), "y_pred", False)
    if pred_col in pred.columns and conf_col in pred.columns:
        add("fragile predicted elite", pred[pred_col].astype(bool) & ~pred[conf_col].astype(bool), "y_pred", False)
    if f"positive_anomaly_{tag}" in pred.columns:
        sort_col = "p_positive_surprise" if "p_positive_surprise" in pred.columns else "residual"
        add("positive surprise", pred[f"positive_anomaly_{tag}"].astype(bool), sort_col, True)
    if f"negative_anomaly_{tag}" in pred.columns:
        sort_col = "p_negative_surprise" if "p_negative_surprise" in pred.columns else "residual"
        add("negative surprise", pred[f"negative_anomaly_{tag}"].astype(bool), sort_col, True)
    ordinary_mask = pd.Series(True, index=pred.index)
    for col in [conf_col, poss_col, f"positive_anomaly_{tag}", f"negative_anomaly_{tag}"]:
        if col in pred.columns:
            ordinary_mask &= ~pred[col].astype(bool)
    add("ordinary high-rank contrast", ordinary_mask, "y_pred", False, limit=max(2, n // 2))

    if not selections:
        return pd.DataFrame()
    out = pd.concat(selections, ignore_index=True)
    out = out.drop_duplicates("display_id", keep="first")
    if true_col in out.columns and pred_col in out.columns:
        out["retrospective_false_elite_flag"] = out[pred_col].astype(bool) & ~out[true_col].astype(bool)
    max_rows = int(getattr(CFG, "PAPER_B_MAX_CASEBOOK_ROWS", 60))
    return out.head(max_rows).copy()


def write_paper_b_candidate_tables(casebook: pd.DataFrame) -> List[Path]:
    """Write compact main/SI tables for the Paper-B candidate story."""
    outputs: List[Path] = []
    if casebook.empty:
        return outputs
    full_csv = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_complete.csv"
    safe_to_csv(casebook, full_csv)
    outputs.append(full_csv)

    preferred = [
        "paper_b_case_class", "display_id", "y_true", "y_pred", "residual",
        f"L_adaptive_{int(CFG.MAIN_COVERAGE*100)}", f"U_adaptive_{int(CFG.MAIN_COVERAGE*100)}",
        f"L_{int(CFG.MAIN_COVERAGE*100)}", f"U_{int(CFG.MAIN_COVERAGE*100)}",
        "p_positive_surprise", "p_negative_surprise", "decision_class_main",
        "topology_label", "geometry_cluster", "metal_cluster", "functional_cluster", "ligand_cluster",
        "pld_or_limiting_diameter", "lcd_or_largest_cavity", "density", "accessible_surface_area", "accessible_volume_or_pore_volume", "void_fraction",
        "provenance_plausibility_status", "core_mof_match", "mosaec_match", "mofchecker_match",
        "process_proxy_status", "stability_proxy_status", "cross_target_sanity",
        "positive_surprise_mechanistic_family", "case_interpretation_short",
    ]
    cols = [c for c in preferred if c in casebook.columns]
    compact = casebook[cols].copy()
    compact_csv = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_compact.csv"
    safe_to_csv(compact, compact_csv)
    outputs.append(compact_csv)
    compact_tex = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_compact.tex"
    save_latex_table(
        compact.head(24),
        compact_tex,
        caption="Candidate-level Paper-B casebook linking conformal decision class, pore descriptors, external plausibility overlays and process/stability proxy labels.",
        label="tab:paper_b_candidate_casebook",
    )
    outputs.append(compact_tex)

    if "positive_surprise_mechanistic_family" in casebook:
        family = casebook.groupby(["paper_b_case_class", "positive_surprise_mechanistic_family"], dropna=False).size().reset_index(name="n_candidates")
        p = PATHS["paper_b_tables"] / "paper_b_positive_surprise_mechanistic_families.csv"
        safe_to_csv(family, p)
        outputs.append(p)
    triage_cols = [c for c in ["paper_b_case_class", "display_id", "process_proxy_status", "process_proxy_reason", "stability_proxy_status", "cross_target_sanity", "provenance_plausibility_status"] if c in casebook.columns]
    if triage_cols:
        p = PATHS["paper_b_tables"] / "paper_b_process_stability_triage.csv"
        safe_to_csv(casebook[triage_cols], p)
        outputs.append(p)
    return outputs


def figure_paper_b_plausibility_triage(casebook: pd.DataFrame) -> List[Path]:
    """Paper-B Figure: candidate classes, overlays, process filters and mechanisms."""
    if casebook.empty:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()

    ax = axes[0]
    data = casebook["paper_b_case_class"].value_counts()
    data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_candidate_class_counts.csv")
    ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=7)
    style_axes(ax, "Candidate-level decision vocabulary", "Class", "Candidates")
    add_panel_label(ax, "a")

    ax = axes[1]
    overlay_cols = [c for c in ["core_mof_match", "mosaec_match", "mofchecker_match"] if c in casebook.columns]
    if overlay_cols:
        data = pd.Series({c.replace("_match", ""): int(casebook[c].astype(bool).sum()) for c in overlay_cols})
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_external_overlay_counts.csv")
        ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=20, ha="right", fontsize=8)
    style_axes(ax, "External plausibility overlays", "Overlay", "Matched candidates")
    add_panel_label(ax, "b")

    ax = axes[2]
    if "process_proxy_status" in casebook.columns:
        data = casebook["process_proxy_status"].value_counts()
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_process_proxy_counts.csv")
        ax.barh(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=7)
        ax.invert_yaxis()
    style_axes(ax, "Post-certificate process proxy", "Candidates", "")
    add_panel_label(ax, "c")

    ax = axes[3]
    if "positive_surprise_mechanistic_family" in casebook.columns:
        data = casebook["positive_surprise_mechanistic_family"].value_counts().head(10)
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_mechanistic_family_counts.csv")
        ax.barh(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=7)
        ax.invert_yaxis()
    style_axes(ax, "Mechanistic families for cases", "Candidates", "")
    add_panel_label(ax, "d")

    fig.suptitle("Paper-B addition. From conformal certificates to chemically plausible candidate triage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["paper_b_figures"] / "PaperB_Figure_plausibility_process_mechanism_triage")


def write_paper_b_reporting_protocol_tex() -> Path:
    """Write a compact LaTeX protocol for the manuscript/SI."""
    out = PATHS["paper_b"] / "paper_b_reporting_protocol.tex"
    content = r'''
\section*{Paper-B reporting protocol: from rankings to risk certificates}

For a high-impact conformal MOF screening manuscript, the promoted candidates should not be reported as a bare top-$k$ ranking. The recommended workflow is:

\begin{enumerate}
    \item Train baseline predictive models on ARC--MOF adsorption labels and report ordinary accuracy metrics only as the first diagnostic layer.
    \item Split the data into training, calibration and test/screening pools using both random and chemically grouped splits. Treat random-split coverage as interpolation evidence and grouped-split coverage as the stronger deployment-facing evidence.
    \item Calibrate prediction intervals on the calibration split. Report predicted elites, possible elites, confident elites and fragile elites separately.
    \item Use conformal residual scores to identify positive-surprise and negative-surprise candidates. Interpret positive surprises as hypotheses about underencoded local adsorption environments, not as automatic discovery claims.
    \item Overlay each promoted or anomalous candidate with available CoRE MOF, MOSAEC and/or MOFChecker evidence. State explicitly whether the candidate is externally matched, externally checked, or unmatched/hypothetical.
    \item Apply post-certificate process and stability proxies: density window, void-fraction window, PLD window, cross-target sanity, activation/water/stability metadata where available.
    \item Promote only candidates that combine a conformal certificate, chemical interpretability, structural plausibility and no obvious process/stability red flag.
\end{enumerate}

\paragraph{Recommended headline.}
A large fraction of apparently elite MOFs can be fragile once uncertainty is calibrated; conformal lower-bound screening reduces false elite claims while residual-surprise analysis exposes chemically interesting exceptions to the model's average structure--property rules.
'''.strip()
    out.write_text(content, encoding="utf-8")
    return out


def stage_paper_b_impact_additions(metrics_df: pd.DataFrame, merged: pd.DataFrame, target_mapping: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Run all Paper-B additions requested in the attached suggestion file."""
    stage = "paper_b_impact_additions_v1"
    sentinel = PATHS["paper_b"] / "paper_b_impact_additions_complete.txt"
    if is_stage_done(stage, [sentinel]):
        LOGGER.info("SKIP_STAGE | %s", stage)
        return {"status": "skipped", "sentinel": sentinel}
    if not getattr(CFG, "ENABLE_PAPER_B_IMPACT_ADDITIONS", True):
        sentinel.write_text("Paper-B additions skipped by configuration at " + now(), encoding="utf-8")
        mark_stage_done(stage, [sentinel], {"skipped": True})
        return {"status": "disabled", "sentinel": sentinel}

    LOGGER.info("STAGE >>> PAPER_B_IMPACT_ADDITIONS")
    outputs: List[Path] = []
    file_check = write_paper_b_input_file_check()
    outputs.append(PATHS["download_guides"] / "paper_b_input_file_check.csv")
    outputs.append(PATHS["download_guides"] / "paper_b_input_file_check.tex")

    casebook = select_paper_b_casebook(metrics_df)
    if not casebook.empty:
        casebook = add_context_to_cases(casebook, merged, target_mapping)
        casebook = annotate_external_plausibility(casebook, merged)
        casebook = add_process_stability_triage(casebook)
        casebook = add_mechanistic_positive_surprise_family(casebook)
        outputs.extend(write_paper_b_candidate_tables(casebook))
        outputs.extend(figure_paper_b_plausibility_triage(casebook))
    else:
        LOGGER.warning("PAPER_B_CASEBOOK_EMPTY | no representative prediction file/candidates available")

    protocol = write_paper_b_reporting_protocol_tex()
    outputs.append(protocol)
    sentinel.write_text("Paper-B additions completed at " + now(), encoding="utf-8")
    outputs.append(sentinel)
    mark_stage_done(stage, outputs, {"n_outputs": len(outputs), "n_casebook_rows": int(len(casebook)) if not casebook.empty else 0})
    return {"status": "complete", "outputs": outputs, "file_check": file_check}



# =============================================================================
# v4 automatic QC reports, sensitivity analysis, CIF extraction and fixed figures
# =============================================================================

def write_mode_configuration_report() -> Path:
    """Save the final interpreted mode settings for auditability."""
    out = PATHS["qc"] / "run_mode_configuration.json"
    payload = {
        "save_mode": CFG.SAVE_MODE,
        "ram_mode": CFG.RAM_MODE,
        "comprehensiveness": CFG.COMPREHENSIVENESS,
        "n_jobs": CFG.N_JOBS,
        "models_to_run": CFG.MODELS_TO_RUN,
        "feature_families_to_run": CFG.FEATURE_FAMILIES_TO_RUN,
        "split_types_to_run": CFG.SPLIT_TYPES_TO_RUN,
        "coverage_levels": CFG.COVERAGE_LEVELS,
        "elite_quantiles": CFG.ELITE_QUANTILES,
        "prediction_output_mode": CFG.PREDICTION_OUTPUT_MODE,
        "fig_formats": CFG.FIG_FORMATS,
        "save_model_objects": CFG.SAVE_MODEL_OBJECTS,
        "save_merged_engineered_csv": CFG.SAVE_MERGED_ENGINEERED_CSV,
        "max_rows_for_fast_test": CFG.MAX_ROWS_FOR_FAST_TEST,
    }
    save_json(payload, out)
    return out


def feature_family_redundancy_report(families: Dict[str, List[str]], drop_duplicates: Optional[bool] = None) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Report and optionally remove feature-family duplicates.

    Reviewers can object if an ablation compares feature families that are
    actually identical after column filtering.  This routine writes an explicit
    SI table and removes exact duplicates from the model job grid unless disabled.
    """
    if drop_duplicates is None:
        drop_duplicates = bool(getattr(CFG, "DROP_REDUNDANT_FEATURE_FAMILIES", True))
    rows: List[Dict[str, Any]] = []
    kept: Dict[str, List[str]] = {}
    signatures: Dict[Tuple[str, ...], str] = {}
    for name, cols in families.items():
        sig = tuple(sorted(set(cols)))
        duplicate_of = signatures.get(sig, "")
        is_duplicate = bool(duplicate_of)
        rows.append({
            "feature_family": name,
            "n_columns": len(cols),
            "n_unique_columns": len(set(cols)),
            "is_exact_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "included_in_model_jobs": (not is_duplicate) or (not drop_duplicates),
        })
        if not is_duplicate:
            signatures[sig] = name
            kept[name] = cols
        elif not drop_duplicates:
            kept[name] = cols
    report = pd.DataFrame(rows)
    safe_to_csv(report, PATHS["qc"] / "feature_family_redundancy_report.csv")
    save_latex_table(report, PATHS["tables_si"] / "table_si_feature_family_redundancy.tex", caption="Feature-family redundancy audit and automatic inclusion/exclusion in model jobs.", label="tab:si_feature_family_redundancy")
    save_json({"kept_families": {k: v for k, v in kept.items()}}, PATHS["processed"] / "feature_families_after_redundancy_filter.json")
    return kept, report


def write_split_diagnostic_report(df: pd.DataFrame, families: Dict[str, List[str]]) -> pd.DataFrame:
    """Explain whether topology/geometry/chemistry grouped splits are available."""
    role_cols = identify_feature_columns(df, {k: None for k in TARGET_DEFINITIONS})
    rows: List[Dict[str, Any]] = []
    for split_type in ["random", "topology_grouped", "geo_cluster_grouped", "chemistry_cluster_grouped"]:
        col = choose_group_column(df, split_type, role_cols) if split_type != "random" else None
        reason = "available" if split_type == "random" or col else "unavailable"
        if split_type != "random" and not col:
            if split_type == "geo_cluster_grouped":
                hits = [c for c in df.columns if "geo" in slugify(c) and "cluster" in slugify(c)]
                reason = "no usable geo-cluster column" if not hits else "geo-cluster column failed cardinality/nonmissing checks"
            elif split_type == "topology_grouped":
                hits = [c for c in df.columns if any(h in slugify(c) for h in TOPOLOGY_HINTS)]
                reason = "no usable topology column" if not hits else "topology column failed cardinality/nonmissing checks"
            else:
                hits = [c for c in df.columns if any(h in slugify(c) for h in ["mc_cluster", "func_cluster", "flig_cluster", "metal", "functional", "linker", "cluster"])]
                reason = "no usable chemistry/cluster column" if not hits else "chemistry/cluster column failed cardinality/nonmissing checks"
        rows.append({
            "split_type": split_type,
            "selected_group_column": col or "",
            "status": "available" if (split_type == "random" or col) else "unavailable",
            "reason": reason,
            "requested_in_run": split_type in getattr(CFG, "SPLIT_TYPES_TO_RUN", ()),
        })
    out = pd.DataFrame(rows)
    safe_to_csv(out, PATHS["qc"] / "split_availability_diagnostics.csv")
    save_latex_table(out, PATHS["tables_si"] / "table_si_split_availability_diagnostics.tex", caption="Grouped-split availability diagnostics, including explicit explanation if the geo-cluster split is unavailable.", label="tab:si_split_diagnostics")
    # Write a plain text reason specifically for geo-cluster split.
    geo = out[out["split_type"] == "geo_cluster_grouped"]
    if not geo.empty:
        reason = str(geo.iloc[0]["reason"])
        (PATHS["qc"] / "geo_cluster_split_unavailable_reason.txt").write_text(reason if geo.iloc[0]["status"] != "available" else "geo_cluster_grouped split is available.", encoding="utf-8")
    return out


def zero_and_nearzero_diagnostics(df: pd.DataFrame, target_mapping: Dict[str, Optional[str]], pred: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Audit zeros in target columns, descriptors and representative decision classes."""
    rows: List[Dict[str, Any]] = []
    def add_numeric(label: str, col: str, source: str) -> None:
        if not col or col not in df.columns:
            return
        x = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": label,
            "source_column": col,
            "source": source,
            "n": int(len(x)),
            "n_nonmissing": int(x.notna().sum()),
            "n_missing": int(x.isna().sum()),
            "n_zero": int((x == 0).sum()),
            "n_near_zero_1e12": int((x.abs() <= 1e-12).sum()),
            "zero_fraction": float((x == 0).mean()) if len(x) else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "median": float(x.median()) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
            "interpretation": "target zero/descriptor zero; check structural plausibility if promoted" if source != "target" else "target zero is present in adsorption label, not automatically a code error",
        })
    for key, col in target_mapping.items():
        if col:
            add_numeric(key, col, "target")
    for c in df.columns:
        s = slugify(c)
        if any(h in s for h in ["asa", "gasa", "vasa", "ava", "avaf", "avag", "poava", "void", "pld", "lcd", "density", "di", "df", "dif", "pore_volume", "surface_area"]):
            if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().sum() > max(10, int(0.01 * len(df))):
                add_numeric(c, c, "descriptor")
    if pred is not None and not pred.empty:
        for c in pred.columns:
            if any(c.startswith(prefix) for prefix in ["predicted_elite", "possible_elite", "confident_elite", "positive_anomaly", "negative_anomaly", "two_sided_anomaly"]):
                try:
                    vals = pred[c].astype(bool)
                    rows.append({
                        "variable": c,
                        "source_column": c,
                        "source": "prediction_class",
                        "n": int(len(pred)),
                        "n_nonmissing": int(vals.notna().sum()),
                        "n_missing": 0,
                        "n_zero": int((~vals).sum()),
                        "n_near_zero_1e12": np.nan,
                        "zero_fraction": float((~vals).mean()),
                        "positive_count": int(vals.sum()),
                        "interpretation": "a zero/small positive_count can be a real conservative conformal gate, not a plotting/code failure",
                    })
                except Exception:
                    continue
    report = pd.DataFrame(rows).sort_values(["source", "variable"]).reset_index(drop=True)
    safe_to_csv(report, PATHS["qc"] / "zero_and_nearzero_diagnostics.csv")
    if not report.empty:
        save_latex_table(report.head(100), PATHS["tables_si"] / "table_si_zero_and_nearzero_diagnostics.tex", caption="Zero and near-zero diagnostics for ARC--MOF targets, pore descriptors and representative conformal decision classes.", label="tab:si_zero_nearzero")
    md_lines = ["# ZERO_VALUE_QC_REPORT", "", f"Generated: {now()}", "", "Zeros in adsorption targets/descriptors and zero selected candidates in strict conformal classes are reported explicitly. A zero confident-elite count can be a scientifically meaningful conservative certificate, not automatically a code error.", ""]
    if not report.empty:
        try:
            md_lines.append(report.to_markdown(index=False))
        except Exception:
            md_lines.append(report.to_csv(index=False))
    (PATHS["qc"] / "ZERO_VALUE_QC_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def load_representative_prediction(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Load one representative prediction file for diagnostics/casebook figures."""
    path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY) or select_representative_prediction_file(metrics_df)
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        df["representative_prediction_file"] = path.name
        return df
    except Exception:
        try:
            return pd.read_csv(path, compression="gzip", low_memory=False)
        except Exception as e:
            LOGGER.warning("REPRESENTATIVE_PREDICTION_LOAD_FAILED | %s | %s", path, e)
            return pd.DataFrame()


def filtered_sensitivity_report(pred: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc sensitivity of class counts/rates after zero-descriptor/process filters."""
    if pred.empty:
        return pd.DataFrame()
    filters: Dict[str, pd.Series] = {"all_representative_rows": pd.Series(True, index=pred.index)}
    for hint in ["asa", "avaf", "ava", "poava", "pld", "lcd", "density"]:
        cols = [c for c in pred.columns if hint in slugify(c) and pd.to_numeric(pred[c], errors="coerce").notna().sum() > 0]
        if cols:
            c = cols[0]
            x = pd.to_numeric(pred[c], errors="coerce")
            filters[f"nonzero_{slugify(c)}"] = x.notna() & (x > 1e-12)
    density_cols = [c for c in pred.columns if "density" in slugify(c)]
    pld_cols = [c for c in pred.columns if any(h in slugify(c) for h in ["pld", "di", "limiting_diameter"])]
    if density_cols:
        d = pd.to_numeric(pred[density_cols[0]], errors="coerce")
        filters["process_density_window"] = d.between(CFG.PROCESS_PROXY_MIN_DENSITY, CFG.PROCESS_PROXY_MAX_DENSITY)
    if pld_cols:
        pld = pd.to_numeric(pred[pld_cols[0]], errors="coerce")
        filters["process_pld_window"] = pld.between(CFG.PROCESS_PROXY_MIN_PLD_A, CFG.PROCESS_PROXY_MAX_PLD_A)
    class_cols = [c for c in pred.columns if any(c.startswith(prefix) for prefix in ["predicted_elite", "possible_elite", "confident_elite", "positive_anomaly", "negative_anomaly"])]
    rows: List[Dict[str, Any]] = []
    for fname, mask in filters.items():
        sub = pred[mask.fillna(False)]
        row = {"filter": fname, "n_rows": int(len(sub)), "fraction_retained": float(len(sub) / max(len(pred), 1))}
        for c in class_cols:
            try:
                row[f"n_{c}"] = int(sub[c].astype(bool).sum())
                row[f"rate_{c}"] = float(sub[c].astype(bool).mean()) if len(sub) else np.nan
            except Exception:
                pass
        rows.append(row)
    out = pd.DataFrame(rows)
    safe_to_csv(out, PATHS["qc"] / "posthoc_filter_sensitivity_report.csv")
    if not out.empty:
        save_latex_table(out, PATHS["tables_si"] / "table_si_posthoc_filter_sensitivity.tex", caption="Post-hoc sensitivity of representative conformal candidate classes after excluding zero/inaccessible-descriptor and process-window edge cases.", label="tab:si_filter_sensitivity")
    return out


def make_fixed_zero_count_figure(pred: pd.DataFrame) -> List[Path]:
    """Create a corrected Figure 5/6-style count figure with explicit zero labels."""
    if pred.empty or not getattr(CFG, "MAKE_FIXED_ZERO_COUNT_FIGURE", True):
        return []
    qtag = int(CFG.MAIN_ELITE_Q * 100)
    tag = int(CFG.MAIN_COVERAGE * 100)
    candidates = {
        "Predicted elite": f"predicted_elite_q{qtag}",
        "Possible elite": f"possible_elite_q{qtag}",
        "Confident elite": f"confident_elite_q{qtag}",
        "Adaptive confident": f"confident_elite_adaptive_q{qtag}",
        "Positive surprise": f"positive_anomaly_{tag}",
        "Negative surprise": f"negative_anomaly_{tag}",
    }
    counts = {label: int(pred[col].astype(bool).sum()) for label, col in candidates.items() if col in pred.columns}
    if not counts:
        return []
    data = pd.Series(counts)
    safe_to_csv(data.rename_axis("class").reset_index(name="count"), PATHS["figure_data_main"] / "Figure_5_6_fixed_candidate_class_counts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=CFG.FIG_DPI)
    elite = data[[i for i in data.index if "elite" in i.lower() or "confident" in i.lower()]]
    anomaly = data[[i for i in data.index if "surprise" in i.lower()]]
    for ax, subset, title in [(axes[0], elite, "Elite decision gates"), (axes[1], anomaly, "Residual-surprise classes")]:
        if subset.empty:
            continue
        y = subset.values
        ax.bar(range(len(subset)), y, color=bar_colors(subset.index), edgecolor="white", linewidth=0.7)
        ax.set_xticks(range(len(subset))); ax.set_xticklabels(subset.index, rotation=30, ha="right", fontsize=8)
        ymax = max(float(max(y)) if len(y) else 1.0, 1.0)
        ax.set_ylim(0, ymax * 1.25 + 1)
        for i, v in enumerate(y):
            ax.text(i, v + max(ymax * 0.03, 0.2), str(int(v)), ha="center", va="bottom", fontsize=8, fontweight="bold")
        style_axes(ax, title, "Class", "Count")
    log_y = data.replace(0, 1)
    axes[2].bar(range(len(data)), log_y.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.7)
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(len(data))); axes[2].set_xticklabels(data.index, rotation=30, ha="right", fontsize=8)
    for i, (label, v) in enumerate(data.items()):
        text = "0" if v == 0 else str(int(v))
        axes[2].text(i, max(float(v), 1.0) * 1.1, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    style_axes(axes[2], "Same counts on log scale; zeros labelled", "Class", "Count (zero shown as 1)")
    add_panel_label(axes[0], "a"); add_panel_label(axes[1], "b"); add_panel_label(axes[2], "c")
    fig.suptitle("Conformal candidate counts with explicit zero and near-zero classes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_current_figure(PATHS["figures_main"] / "Figure_5_6_fixed_zero_candidate_counts")


def find_cif_for_display_id(display_id: str) -> Optional[Path]:
    """Find an ARC-MOF/CoRE CIF for a selected case without indexing all CIFs."""
    if not display_id:
        return None
    did = str(display_id).strip()
    stems = {did, did.replace(".cif", ""), slugify(did).replace("_cif", "")}
    search_roots = [PATHS.get("raw_structures"), PATHS.get("raw_arc_mof") / "structures" if PATHS.get("raw_arc_mof") else None, PATHS.get("raw_core_mof") / "structures_archive" if PATHS.get("raw_core_mof") else None]
    for root in search_roots:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for stem in stems:
            for pattern in [stem + ".cif", stem + "*.cif", "*" + stem + "*.cif"]:
                try:
                    hit = next(root.rglob(pattern), None)
                    if hit and hit.exists():
                        return hit
                except Exception:
                    continue
    return None


def extract_casebook_cifs_if_possible(casebook_csv: Optional[Path] = None) -> pd.DataFrame:
    """Copy selected casebook CIFs into results/structural_case_studies/selected_cifs."""
    if casebook_csv is None:
        casebook_csv = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_complete.csv"
    if not Path(casebook_csv).exists():
        return pd.DataFrame()
    try:
        casebook = pd.read_csv(casebook_csv)
    except Exception:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for i, row in casebook.head(int(getattr(CFG, "PAPER_B_MAX_CASEBOOK_ROWS", 60))).iterrows():
        did = str(row.get("display_id", row.get("__join_id__", "")))
        cls = str(row.get("paper_b_case_class", row.get("case_class", "case")))
        cif = find_cif_for_display_id(did)
        out_path = ""
        if cif:
            name = f"{i+1:03d}_{slugify(cls, 40)}_{slugify(did, 80)}.cif"
            dest = PATHS["selected_cifs"] / name
            try:
                shutil.copy2(cif, dest)
                out_path = str(dest)
            except Exception as e:
                LOGGER.warning("CIF_COPY_FAILED | %s | %s", cif, e)
        rows.append({"rank": i + 1, "case_class": cls, "display_id": did, "source_cif": str(cif) if cif else "", "copied_cif": out_path, "recommended_panel": "main text" if i < 10 else "SI"})
    manifest = pd.DataFrame(rows)
    safe_to_csv(manifest, PATHS["structural_cases"] / "structural_case_study_manifest.csv")
    instr = ["# Structure rendering manifest", "", "Use the copied CIFs for VESTA/PyMOL/OVITO/CrystalMaker panels. Choose 6--10 visually clear cases for the main text: positive surprise, negative surprise, fragile predicted elite, possible/confident elite. Add PLD/LCD arrows, density/void-fraction labels and external provenance badges where available.", ""]
    try:
        instr.append(manifest.to_markdown(index=False))
    except Exception:
        instr.append(manifest.to_csv(index=False))
    (PATHS["structural_cases"] / "STRUCTURAL_RENDERING_MANIFEST.md").write_text("\n".join(instr), encoding="utf-8")
    return manifest


def stage_v4_qc_sensitivity_and_fixed_figures(merged: pd.DataFrame, target_mapping: Dict[str, Optional[str]], families: Dict[str, List[str]], metrics_df: pd.DataFrame) -> List[Path]:
    """One high-level stage for all automatic reviewer-facing v4 improvements."""
    outputs: List[Path] = []
    write_mode_configuration_report(); outputs.append(PATHS["qc"] / "run_mode_configuration.json")
    write_split_diagnostic_report(merged, families); outputs.append(PATHS["qc"] / "split_availability_diagnostics.csv")
    pred = load_representative_prediction(metrics_df) if metrics_df is not None and not metrics_df.empty else pd.DataFrame()
    zero_and_nearzero_diagnostics(merged, target_mapping, pred); outputs.append(PATHS["qc"] / "zero_and_nearzero_diagnostics.csv")
    if not pred.empty:
        filtered_sensitivity_report(pred); outputs.append(PATHS["qc"] / "posthoc_filter_sensitivity_report.csv")
        outputs.extend(make_fixed_zero_count_figure(pred))
    extract_casebook_cifs_if_possible(); outputs.append(PATHS["structural_cases"] / "structural_case_study_manifest.csv")
    return outputs

# =============================================================================
# Final report and run summary
# =============================================================================

def write_structural_representation_guide() -> Path:
    """Write a guide telling the student which figure panels need real structures."""
    out = PATHS["structural_cases"] / "STRUCTURAL_REPRESENTATION_GUIDE.md"
    content = f"""
# Structural representations to add for a high-impact MOF paper

The numerical pipeline can identify candidates and generate placeholders, but a
high-IF MOF/materials audience will expect real structures. Render ARC-MOF CIFs
for the candidates listed in `structural_case_study_manifest.csv`.

## Main-text panels where structures should appear

1. **Figure 5d — positive-surprise structural quartet**
   - Show 3–4 conformal positive-surprise MOFs.
   - Use a node/linker view plus a pore-window or channel view.
   - Annotate PLD/LCD, density, accessible volume or void fraction, topology and
     cluster label where available.
   - Scientific purpose: show why the anomaly is chemically surprising, not just
     statistically high-residual.

2. **Figure 6d — decision-contrast structural quartet**
   - Show one adaptive/confident elite, one positive-surprise anomaly, one
     negative-surprise anomaly, and one fragile uncalibrated top-k candidate.
   - Scientific purpose: demonstrate that conformal screening separates different
     kinds of materials decisions rather than merely re-ranking a list.

## Recommended SI structural panels

- **Figure S4**: 8–12 additional positive-surprise structures grouped by pore
  regime/topology/cluster family.
- **Figure S5**: negative-surprise structures, emphasizing blocked/inaccessible,
  overly dense, or topology/chemistry regimes where the model overpredicts.
- **Table S-Structural**: all rendered candidates with ID, target, y_true, y_pred,
  residual, conformal p-values, topology/cluster, PLD/LCD/density/void fraction.

## How to provide images to this script

Place rendered images in one of these folders next to the Python script:

- `structure_images/`
- `structures_rendered/`
- `figures/structures/`
- `data/structures_rendered/`

Use either the exact `display_id` or a slugified version as the filename, for
example `mof_12345.png`. The script will automatically replace placeholders in
Figure 5d and Figure 6d when matching images are present.
""".strip()
    out.write_text(content, encoding="utf-8")
    return out


def write_run_summary(audit: Dict[str, Any], target_mapping: Dict[str, Optional[str]], families: Dict[str, List[str]], metrics_df: pd.DataFrame) -> Path:
    path = PATHS["results"] / "RUN_SUMMARY_README.txt"
    lines = []
    lines.append("Conformal MOF Anomaly Screening - Run Summary")
    lines.append("=" * 60)
    lines.append(f"Generated: {now()}")
    lines.append(f"Project root: {PATHS['root']}")
    lines.append(f"Results dir: {PATHS['results']}")
    lines.append(f"Save mode: {CFG.SAVE_MODE}")
    lines.append(f"RAM mode: {CFG.RAM_MODE}")
    lines.append(f"Comprehensiveness: {CFG.COMPREHENSIVENESS}")
    lines.append(f"Model profile: {CFG.MODEL_PROFILE}")
    lines.append(f"N_JOBS used: {CFG.N_JOBS}")
    lines.append(f"RF settings: n_estimators={CFG.RF_N_ESTIMATORS}, max_depth={CFG.RF_MAX_DEPTH}, min_samples_leaf={CFG.RF_MIN_SAMPLES_LEAF}, max_samples={CFG.RF_MAX_SAMPLES}")
    lines.append(f"HGB settings: max_iter={CFG.HGB_MAX_ITER}, learning_rate={CFG.HGB_LEARNING_RATE}, min_samples_leaf={CFG.HGB_MIN_SAMPLES_LEAF}, early_stopping={CFG.HGB_EARLY_STOPPING}")
    lines.append("")
    lines.append("Detected targets:")
    for k, v in target_mapping.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Feature families:")
    for k, v in families.items():
        lines.append(f"  - {k}: {len(v)} columns")
    lines.append("")
    lines.append("Completed jobs:")
    lines.append(f"  - {len(metrics_df)} model/split/seed jobs")
    lines.append("")
    lines.append("Important output folders:")
    for key in ["audit", "processed", "metrics", "predictions", "models", "tables_main", "tables_si", "figures_main", "figures_si", "figure_data_main", "figure_data_si", "qc", "paper_b", "structural_cases", "download_guides", "logs"]:
        lines.append(f"  - {key}: {PATHS[key]}")
    lines.append("")
    lines.append("Manuscript figures generated:")
    lines.append("  - Figure_1_conceptual_workflow")
    lines.append("  - Figure_2_calibration_efficiency")
    lines.append("  - Figure_3_elite_shortlist_reliability")
    lines.append("  - Figure_4_conformal_anomaly_atlas")
    lines.append("  - Figure_5_chemistry_confident_anomalies")
    lines.append("  - Figure_6_screening_consequence")
    lines.append("")
    lines.append("Recommended next manual checks before writing final manuscript:")
    lines.append("  1. Inspect results/data_audit/csv_column_audit_long.csv for target mapping correctness.")
    lines.append("  2. Open results/tables/main/*.csv and verify candidate identities.")
    lines.append("  3. Inspect grouped-split results; strong claims should not rely only on random splits.")
    lines.append("  4. Use figure_data CSV files to redraw or fine-tune any panel without rerunning models.")
    lines.append("  5. Add real rendered structures to Figure 5d and Figure 6d using results/structural_case_studies/STRUCTURAL_REPRESENTATION_GUIDE.md.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path



# =============================================================================
# v5 high-impact automated additions
# =============================================================================
# These definitions intentionally override selected v4 functions above.  They do
# not remove the clean-data-free merge, Jinja2-safe LaTeX export, restart-safety,
# save/RAM/comprehensiveness modes, or the v4 QC reports.  The v5 additions focus
# on the remaining objective/reproducible upgrades requested after inspecting the
# first PUBLICATION_OUTPUT run:
#   1) stricter external-match confidence, not overclaiming fuzzy overlays;
#   2) a stronger Figure 5b based on positive-surprise rate by pore-descriptor
#      quintile rather than a visually weak scatter;
#   3) more explicit structural-rendering and candidate-prioritisation tables;
#   4) reviewer-facing submission-readiness report.
# The previously mentioned incomplete-ZIP issue is not treated as a scientific
# problem here because the user clarified that those files existed locally but
# were simply not included in the uploaded ZIP.


def _clean_identifier_token(value: Any) -> str:
    """Return a conservative identifier token for exact-ish matching."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    s = re.sub(r"\.(cif|json|txt|csv)$", "", s, flags=re.IGNORECASE)
    s = s.strip().lower()
    return s


def _identifier_match_key_sets(row: pd.Series, merged: Optional[pd.DataFrame] = None) -> Tuple[set, set]:
    """Return (exact_keys, fuzzy_keys) for conservative overlay matching.

    exact_keys are normalised raw identifiers from display_id, __join_id__,
    filename/name/refcode-style columns.  fuzzy_keys are slugified variants used
    only for lower-confidence matching.  This prevents the manuscript from
    overclaiming broad substring/provenance overlays as direct experimental
    validation.
    """
    exact: set = set()
    fuzzy: set = set()
    preferred = ["display_id", "__join_id__", "filename", "name", "Name", "MOF", "mof", "mofid", "MOFid", "mofkey", "MOFkey", "refcode", "database_code", "cif"]
    rows = [row]
    if merged is not None and "row_index" in row.index and pd.notna(row.get("row_index")):
        try:
            ix = int(row.get("row_index"))
            if 0 <= ix < len(merged):
                rows.append(merged.iloc[ix])
        except Exception:
            pass
    for r in rows:
        for c in list(preferred) + [col for col in r.index if any(h in slugify(col) for h in ID_HINTS + ["mofkey", "refcode", "database_code"] )]:
            if c not in r.index:
                continue
            token = _clean_identifier_token(r.get(c))
            if not token:
                continue
            exact.add(token)
            try:
                exact.add(normalize_identifier_series(pd.Series([token])).iloc[0])
            except Exception:
                pass
            fuzzy.add(slugify(token))
            fuzzy.add(slugify(token.replace("_clean", "")))
    # Remove exact keys from fuzzy so confidence labelling stays meaningful.
    fuzzy = {k for k in fuzzy if k and k not in exact}
    exact = {k for k in exact if k}
    return exact, fuzzy


def _overlay_source_roots() -> Dict[str, Path]:
    """Return overlay roots including CoRE 2025 when present."""
    roots = {
        "core_mof_2024": PATHS.get("raw_core_mof", CFG.PROJECT_ROOT / CFG.CORE_MOF_SUBDIR),
        "core_mof_2025": PATHS.get("raw_core_mof_2025", Path(CFG.DATA_ROOT) / getattr(CFG, "CORE_MOF_2025_SUBDIR", "core_mof_2025")),
        "mosaec": PATHS.get("raw_mosaec", CFG.PROJECT_ROOT / CFG.MOSAEC_SUBDIR),
        "mofchecker": PATHS.get("raw_mofchecker", CFG.PROJECT_ROOT / CFG.MOFCHECKER_SUBDIR),
    }
    return {k: Path(v) for k, v in roots.items() if v is not None}


def annotate_external_plausibility(cases: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Annotate candidates with strict external-match confidence.

    v4 correctly provided broad CoRE/MOSAEC/MOFChecker overlays, but the first
    publication-output review showed that this can be too easy to overinterpret:
    a broad metadata summary is useful as plausibility support, but it is not the
    same as an exact external experimental match.  v5 therefore writes both the
    old boolean overlay columns and a strict confidence vocabulary:

    - exact_id_match: candidate identifier directly matches an indexed overlay ID;
    - strong_multi_source_or_checker: no exact ID, but multiple independent
      overlays or MOFChecker-like evidence are present;
    - fuzzy_or_metadata_only: only lower-confidence slug/metadata support;
    - no_external_match: no optional overlay match.
    """
    if cases.empty or not getattr(CFG, "ENABLE_EXTERNAL_PLAUSIBILITY_OVERLAY", True):
        return cases

    overlay_sources = _overlay_source_roots()
    loaded: Dict[str, List[Tuple[str, pd.DataFrame, Dict[str, List[int]]]]] = {}
    for source, root in overlay_sources.items():
        loaded[source] = []
        if not Path(root).exists():
            LOGGER.info("PAPER_B_OVERLAY_ROOT_MISSING | %s | %s", source, root)
            continue
        # _collect_overlay_tables and _build_identifier_index are defined in the
        # v4 code above; this v5 override reuses them.
        for table_name, table in _collect_overlay_tables(Path(root), source):
            idx = _build_identifier_index(table)
            if idx:
                loaded[source].append((table_name, table, idx))
        LOGGER.info("PAPER_B_OVERLAY_TABLES | %s | %d indexed tables", source, len(loaded[source]))

    out = cases.copy()
    for source in overlay_sources:
        out[f"{source}_match"] = False
        out[f"{source}_exact_id_match"] = False
        out[f"{source}_fuzzy_match"] = False
        out[f"{source}_matched_tables"] = ""
        out[f"{source}_metadata_summary"] = ""
        out[f"{source}_match_key_used"] = ""

    strict_rows: List[Dict[str, Any]] = []
    for ridx, row in out.iterrows():
        exact_keys, fuzzy_keys = _identifier_match_key_sets(row, merged)
        sources_with_match: List[str] = []
        sources_with_exact: List[str] = []
        sources_with_fuzzy: List[str] = []
        for source, tables in loaded.items():
            matched_tables: List[str] = []
            summaries: List[str] = []
            matched_keys: List[str] = []
            exact_found = False
            fuzzy_found = False
            for table_name, table, idx in tables:
                matched_indices: List[int] = []
                for k in exact_keys:
                    hits = idx.get(k, [])
                    if hits:
                        exact_found = True
                        matched_keys.append(k)
                        matched_indices.extend(hits)
                # Fuzzy matching is intentionally secondary.  It is still useful
                # for refcodes with suffixes and slugged file names, but it is
                # labelled separately in the output table.
                if not matched_indices:
                    for k in fuzzy_keys:
                        hits = idx.get(k, [])
                        if hits:
                            fuzzy_found = True
                            matched_keys.append(k)
                            matched_indices.extend(hits)
                if matched_indices:
                    matched_tables.append(table_name)
                    summary = _summarise_overlay_match(table, matched_indices[:20])
                    if summary.get("metadata_summary"):
                        summaries.append(summary["metadata_summary"])
            if matched_tables:
                sources_with_match.append(source)
                if exact_found:
                    sources_with_exact.append(source)
                if fuzzy_found and not exact_found:
                    sources_with_fuzzy.append(source)
                out.at[ridx, f"{source}_match"] = True
                out.at[ridx, f"{source}_exact_id_match"] = bool(exact_found)
                out.at[ridx, f"{source}_fuzzy_match"] = bool(fuzzy_found and not exact_found)
                out.at[ridx, f"{source}_matched_tables"] = "; ".join(matched_tables[:8])
                out.at[ridx, f"{source}_metadata_summary"] = " | ".join(summaries[:4])
                out.at[ridx, f"{source}_match_key_used"] = "; ".join(sorted(set(matched_keys))[:8])

        n_sources = len(sources_with_match)
        exact_any = bool(sources_with_exact)
        mofchecker_any = bool(out.loc[ridx, "mofchecker_match"]) if "mofchecker_match" in out.columns else False
        if exact_any:
            confidence = "exact_id_match"
            claim_strength = "direct external identifier match; can be described as external-match support after manual ID check"
        elif n_sources >= 2 or mofchecker_any:
            confidence = "strong_multi_source_or_checker"
            claim_strength = "strong plausibility support; do not call experimental validation without manual structure/ID confirmation"
        elif n_sources == 1:
            confidence = "fuzzy_or_metadata_only"
            claim_strength = "weak plausibility support only; use as an overlay, not validation"
        else:
            confidence = "no_external_match"
            claim_strength = "ARC-MOF-only or unmatched; needs manual/provenance caution"

        out.at[ridx, "external_match_confidence"] = confidence
        out.at[ridx, "external_match_sources"] = "; ".join(sources_with_match)
        out.at[ridx, "exact_id_match_any"] = exact_any
        out.at[ridx, "fuzzy_match_any"] = bool(sources_with_fuzzy)
        out.at[ridx, "external_plausibility_tier"] = {
            "exact_id_match": "Tier 1: exact external ID support",
            "strong_multi_source_or_checker": "Tier 2: multi-source/checker plausibility",
            "fuzzy_or_metadata_only": "Tier 3: fuzzy metadata plausibility",
            "no_external_match": "Tier 4: no optional external overlay",
        }.get(confidence, "Tier 4: no optional external overlay")
        out.at[ridx, "recommended_claim_strength"] = claim_strength
        strict_rows.append({
            "paper_b_case_class": row.get("paper_b_case_class", row.get("structural_case_role", "")),
            "display_id": row.get("display_id", ""),
            "external_match_confidence": confidence,
            "external_match_sources": "; ".join(sources_with_match),
            "exact_id_match_any": exact_any,
            "fuzzy_match_any": bool(sources_with_fuzzy),
            "recommended_claim_strength": claim_strength,
        })

    def provenance_label(r: pd.Series) -> str:
        confidence = str(r.get("external_match_confidence", ""))
        if confidence == "exact_id_match":
            return "external exact-ID support"
        if confidence == "strong_multi_source_or_checker":
            return "external plausibility/checker support"
        if confidence == "fuzzy_or_metadata_only":
            return "fuzzy external metadata support only"
        return "ARC-MOF/hypothetical or unmatched in optional overlays"

    out["provenance_plausibility_status"] = out.apply(provenance_label, axis=1)
    strict = pd.DataFrame(strict_rows)
    if not strict.empty:
        safe_to_csv(strict, PATHS["paper_b_tables"] / "paper_b_external_match_strict_validation.csv")
        save_latex_table(
            strict.head(40),
            PATHS["tables_si"] / "table_si_external_match_strict_validation.tex",
            caption="Strict external-match confidence for Paper-B casebook candidates. Fuzzy or metadata-only overlays are not treated as experimental validation.",
            label="tab:si_external_match_strict_validation",
        )
    return out


def _find_best_pore_descriptor_columns(pred: pd.DataFrame) -> List[str]:
    """Choose a small set of pore descriptors for Figure 5b quintile rates."""
    if pred.empty:
        return []
    priority_groups = [
        ["avaf", "void_fraction", "porosity"],
        ["poavaf", "pore_accessible_volume_fraction"],
        ["asa", "gasa", "surface_area"],
        ["ava", "poava", "pore_volume", "accessible_volume"],
        ["pld", "df", "limiting_pore"],
        ["density"],
    ]
    chosen: List[str] = []
    used = set()
    for hints in priority_groups:
        candidates = []
        for c in pred.columns:
            if c in used:
                continue
            slug = slugify(c)
            if any(h in slug for h in hints):
                x = pd.to_numeric(pred[c], errors="coerce")
                if x.notna().sum() >= max(50, int(0.02 * len(pred))) and x.nunique(dropna=True) >= 5:
                    candidates.append((x.notna().sum(), -len(str(c)), c))
        if candidates:
            candidates.sort(reverse=True)
            col = candidates[0][2]
            chosen.append(col)
            used.add(col)
        if len(chosen) >= 4:
            break
    return chosen


def positive_surprise_quintile_rate_table(pred: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Compute positive-surprise rate by pore-descriptor quintile for Figure 5b."""
    if pred.empty or label_col not in pred.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    descriptors = _find_best_pore_descriptor_columns(pred)
    label = pred[label_col].astype(bool)
    for col in descriptors:
        x = pd.to_numeric(pred[col], errors="coerce")
        ok = x.notna()
        if ok.sum() < 50:
            continue
        try:
            bins = pd.qcut(x[ok], q=5, duplicates="drop")
        except Exception:
            continue
        for q_i, interval in enumerate(bins.cat.categories, start=1):
            mask = ok.copy()
            mask.loc[ok] = (bins == interval).values
            n = int(mask.sum())
            n_pos = int((label & mask).sum())
            rate = n_pos / max(n, 1)
            rows.append({
                "descriptor": col,
                "descriptor_slug": slugify(col),
                "quintile": q_i,
                "quintile_label": f"Q{q_i}",
                "range": str(interval),
                "n": n,
                "n_positive_surprise": n_pos,
                "positive_surprise_rate": float(rate),
                "mean_descriptor_value": float(x[mask].mean()) if n else np.nan,
                "median_descriptor_value": float(x[mask].median()) if n else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        safe_to_csv(out, PATHS["figure_data_main"] / "Figure_5b_positive_surprise_rate_by_pore_quintile.csv")
        safe_to_csv(out, PATHS["tables_si"] / "table_si_positive_surprise_rate_by_pore_quintile.csv")
    return out


def _plot_positive_surprise_quintile_panel(ax: plt.Axes, pred: pd.DataFrame, label_col: str) -> None:
    """Plot the improved Figure 5b panel."""
    table = positive_surprise_quintile_rate_table(pred, label_col)
    if table.empty:
        style_axes(ax, "Positive-surprise rate by pore quintile", "Descriptor quintile", "Rate")
        ax.text(0.5, 0.5, "No suitable pore descriptor\nfor quintile-rate panel", ha="center", va="center", transform=ax.transAxes)
        return
    # Keep up to three descriptors so the panel remains readable.
    descriptors = list(table["descriptor"].drop_duplicates())[:3]
    plot = table[table["descriptor"].isin(descriptors)].copy()
    x = np.arange(1, 6)
    width = 0.22 if len(descriptors) >= 3 else 0.32
    offsets = np.linspace(-width, width, len(descriptors)) if len(descriptors) > 1 else [0]
    for i, desc in enumerate(descriptors):
        sub = plot[plot["descriptor"] == desc].set_index("quintile").reindex(x)
        y = sub["positive_surprise_rate"].values
        ax.bar(x + offsets[i], y, width=width, label=str(desc)[:24], color=COLOR_CYCLE[i], edgecolor="white", linewidth=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x])
    clean_legend(ax, loc="best", ncol=1)
    style_axes(ax, "Positive-surprise rate by pore quintile", "Low → high descriptor quintile", "Positive-surprise rate")
    # Add a compact annotation emphasizing the reviewer-facing interpretation.
    try:
        strongest = table.sort_values("positive_surprise_rate", ascending=False).iloc[0]
        ax.text(
            0.02, 0.96,
            f"Highest: {str(strongest['descriptor'])[:18]} {strongest['quintile_label']}\nrate={strongest['positive_surprise_rate']:.2f}, n={int(strongest['n'])}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=6.7,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=JOURNAL_COLORS["light_grid"], linewidth=0.5),
        )
    except Exception:
        pass


def figure_5_chemistry_of_anomalies(enrichment: pd.DataFrame, metrics_df: pd.DataFrame) -> List[Path]:
    """Figure 5 v5: chemical regimes plus robust pore-quintile signal.

    The former Figure 5b scatter was honest but visually weak for reviewers.  v5
    replaces it with a direct positive-surprise-rate-by-quintile panel for the
    most interpretable pore descriptors, while retaining source data in CSV.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()

    ax = axes[0]
    if not enrichment.empty:
        data = enrichment.head(15).copy()
        data.to_csv(PATHS["figure_data_main"] / "Figure_5a_enrichment_topology_pore_cluster.csv", index=False)
        labels = (data["group_col"].astype(str).str[:12] + ":" + data["group"].astype(str).str[:18]).values
        y = np.arange(len(data))
        ax.barh(y, data["odds_ratio"].values, color=bar_colors(labels), edgecolor="white", linewidth=0.55)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        add_reference_line(ax, x=1.0, label="no enrichment")
        ax.invert_yaxis()
    style_axes(ax, "Which pore/topology/cluster regimes are enriched?", "Odds ratio", "")
    add_panel_label(ax, "a")

    path = select_representative_prediction_file_light(metrics_df, CFG.MAIN_TARGET_KEY) or select_representative_prediction_file(metrics_df)
    pred = pd.read_csv(path, low_memory=False) if path else pd.DataFrame()
    tag = int(CFG.MAIN_COVERAGE * 100)
    qtag = int(CFG.MAIN_ELITE_Q * 100)

    ax = axes[1]
    if not pred.empty:
        _plot_positive_surprise_quintile_panel(ax, pred, f"positive_anomaly_{tag}")
    else:
        style_axes(ax, "Positive-surprise rate by pore quintile", "Descriptor quintile", "Rate")
    add_panel_label(ax, "b")

    ax = axes[2]
    if not pred.empty:
        count_items: Dict[str, int] = {}
        for label, col in [
            ("confident elite", f"confident_elite_q{qtag}"),
            ("adaptive confident elite", f"confident_elite_adaptive_q{qtag}"),
            ("positive surprise", f"positive_anomaly_{tag}"),
            ("negative surprise", f"negative_anomaly_{tag}"),
            ("two-sided anomaly", f"two_sided_anomaly_{tag}"),
        ]:
            if col in pred.columns:
                count_items[label] = int(pred[col].astype(bool).sum())
        pd.Series(count_items).to_csv(PATHS["figure_data_main"] / "Figure_5c_candidate_class_counts.csv")
        ax.bar(range(len(count_items)), list(count_items.values()), color=bar_colors(count_items.keys()), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(count_items))); ax.set_xticklabels(list(count_items.keys()), rotation=30, ha="right", fontsize=7)
        annotate_bars(ax, fmt="{:.0f}", fontsize=7)
    style_axes(ax, "How many materials receive certificates?", "Class", "Count")
    add_panel_label(ax, "c")

    ax = axes[3]
    ax.axis("off")
    cases = write_structural_case_manifest(pred, tag, qtag) if not pred.empty else pd.DataFrame()
    pos_cases = cases[cases["structural_case_role"].str.contains("positive", case=False, na=False)].head(4) if not cases.empty else pd.DataFrame()
    if pos_cases.empty and not cases.empty:
        pos_cases = cases.head(4)
    subaxes = [ax.inset_axes([0.02 + 0.49*(i % 2), 0.52 - 0.49*(i // 2), 0.46, 0.43]) for i in range(4)]
    for i, sax in enumerate(subaxes):
        if i < len(pos_cases):
            r = pos_cases.iloc[i]
            subtitle = f"resid={r.get('residual', np.nan):.2g}; p+={r.get('p_positive_surprise', np.nan):.2g}"
            draw_structure_slot(sax, r["display_id"], "positive-surprise case", subtitle)
        else:
            draw_structure_slot(sax, "candidate CIF", "structure slot", "render selected MOF")
    ax.text(0.02, 0.02, "Replace placeholders with ARC-MOF CIF renders: node/linker view + pore-window view.", transform=ax.transAxes, fontsize=7)
    add_panel_label(ax, "d")

    fig.suptitle("Figure 5. Conformal anomalies expose chemically interpretable MOF regimes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["figures_main"] / "Figure_5_chemistry_structural_cases")


def _case_priority_score(row: pd.Series) -> float:
    """Score casebook rows for structural-rendering priority."""
    score = 0.0
    cls = str(row.get("paper_b_case_class", "")).lower()
    if "positive" in cls:
        score += 50
    if "confident" in cls:
        score += 45
    if "possible" in cls:
        score += 35
    if "fragile" in cls:
        score += 25
    if "negative" in cls:
        score += 20
    # Prefer chemically interpretable and not obviously invalid structures.
    for col in ["pld_or_limiting_diameter", "lcd_or_largest_cavity", "density", "void_fraction", "accessible_surface_area"]:
        try:
            if np.isfinite(float(row.get(col, np.nan))):
                score += 2
        except Exception:
            pass
    ext = str(row.get("external_match_confidence", ""))
    if ext == "exact_id_match":
        score += 12
    elif ext == "strong_multi_source_or_checker":
        score += 8
    elif ext == "fuzzy_or_metadata_only":
        score += 3
    proc = str(row.get("process_proxy_status", "")).lower()
    if "no obvious" in proc:
        score += 6
    if "warning" in proc:
        score -= 8
    # Strong residual/p-value cases are more visually compelling.
    try:
        score += min(10.0, abs(float(row.get("residual", 0.0))))
    except Exception:
        pass
    try:
        pval = float(row.get("p_positive_surprise", np.nan))
        if np.isfinite(pval) and pval > 0:
            score += min(8.0, -math.log10(pval + 1e-12))
    except Exception:
        pass
    return float(score)


def write_paper_b_candidate_tables(casebook: pd.DataFrame) -> List[Path]:
    """Write compact main/SI tables for the Paper-B candidate story, v5."""
    outputs: List[Path] = []
    if casebook.empty:
        return outputs
    casebook = casebook.copy()
    casebook["structure_rendering_priority_score"] = casebook.apply(_case_priority_score, axis=1)
    casebook["recommended_rendering_priority"] = pd.qcut(
        casebook["structure_rendering_priority_score"].rank(method="first"),
        q=min(3, max(1, len(casebook))),
        labels=["SI optional", "SI/main backup", "main-text priority"][-min(3, max(1, len(casebook))):],
        duplicates="drop",
    ).astype(str) if len(casebook) >= 3 else "main-text priority"
    casebook = casebook.sort_values(["structure_rendering_priority_score"], ascending=False).copy()

    full_csv = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_complete.csv"
    safe_to_csv(casebook, full_csv)
    outputs.append(full_csv)

    strict_cols = [c for c in [
        "paper_b_case_class", "display_id", "external_match_confidence", "external_match_sources",
        "exact_id_match_any", "fuzzy_match_any", "core_mof_2024_match", "core_mof_2024_exact_id_match",
        "core_mof_2025_match", "core_mof_2025_exact_id_match", "mosaec_match", "mosaec_exact_id_match",
        "mofchecker_match", "mofchecker_exact_id_match", "recommended_claim_strength",
    ] if c in casebook.columns]
    if strict_cols:
        strict_csv = PATHS["paper_b_tables"] / "paper_b_external_match_strict_validation.csv"
        safe_to_csv(casebook[strict_cols], strict_csv)
        outputs.append(strict_csv)

    preferred = [
        "paper_b_case_class", "display_id", "y_true", "y_pred", "residual",
        f"L_adaptive_{int(CFG.MAIN_COVERAGE*100)}", f"U_adaptive_{int(CFG.MAIN_COVERAGE*100)}",
        f"L_{int(CFG.MAIN_COVERAGE*100)}", f"U_{int(CFG.MAIN_COVERAGE*100)}",
        "p_positive_surprise", "p_negative_surprise", "decision_class_main",
        "topology_label", "geometry_cluster", "metal_cluster", "functional_cluster", "ligand_cluster",
        "pld_or_limiting_diameter", "lcd_or_largest_cavity", "density", "accessible_surface_area", "accessible_volume_or_pore_volume", "void_fraction",
        "external_match_confidence", "external_plausibility_tier", "recommended_claim_strength",
        "provenance_plausibility_status", "process_proxy_status", "stability_proxy_status", "cross_target_sanity",
        "positive_surprise_mechanistic_family", "case_interpretation_short", "structure_rendering_priority_score", "recommended_rendering_priority",
    ]
    cols = [c for c in preferred if c in casebook.columns]
    compact = casebook[cols].copy()
    compact_csv = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_compact.csv"
    safe_to_csv(compact, compact_csv)
    outputs.append(compact_csv)
    compact_tex = PATHS["paper_b_tables"] / "paper_b_candidate_casebook_compact.tex"
    save_latex_table(
        compact.head(24),
        compact_tex,
        caption="Candidate-level Paper-B casebook linking conformal decision class, pore descriptors, strict external-match confidence, process/stability proxy labels and rendering priority.",
        label="tab:paper_b_candidate_casebook",
    )
    outputs.append(compact_tex)

    render_cols = [c for c in [
        "recommended_rendering_priority", "paper_b_case_class", "display_id", "structure_rendering_priority_score",
        "positive_surprise_mechanistic_family", "pld_or_limiting_diameter", "lcd_or_largest_cavity", "density",
        "void_fraction", "accessible_surface_area", "external_match_confidence", "process_proxy_status",
        "case_interpretation_short",
    ] if c in casebook.columns]
    if render_cols:
        rendering_plan = casebook[render_cols].head(24).copy()
        rendering_plan["manual_rendering_instruction"] = "Render CIF in VESTA/PyMOL/OVITO: framework view + pore-window view; add PLD/LCD arrows and external/proxy badges."
        render_csv = PATHS["structural_cases"] / "v5_ranked_structure_rendering_plan.csv"
        safe_to_csv(rendering_plan, render_csv)
        outputs.append(render_csv)
        try:
            md = ["# v5 ranked structure-rendering plan", "", rendering_plan.to_markdown(index=False)]
        except Exception:
            md = ["# v5 ranked structure-rendering plan", "", rendering_plan.to_csv(index=False)]
        (PATHS["structural_cases"] / "v5_ranked_structure_rendering_plan.md").write_text("\n".join(md), encoding="utf-8")
        outputs.append(PATHS["structural_cases"] / "v5_ranked_structure_rendering_plan.md")

    if "positive_surprise_mechanistic_family" in casebook:
        family = casebook.groupby(["paper_b_case_class", "positive_surprise_mechanistic_family"], dropna=False).size().reset_index(name="n_candidates")
        p = PATHS["paper_b_tables"] / "paper_b_positive_surprise_mechanistic_families.csv"
        safe_to_csv(family, p)
        outputs.append(p)
    triage_cols = [c for c in ["paper_b_case_class", "display_id", "process_proxy_status", "process_proxy_reason", "stability_proxy_status", "cross_target_sanity", "provenance_plausibility_status", "external_match_confidence", "recommended_claim_strength"] if c in casebook.columns]
    if triage_cols:
        p = PATHS["paper_b_tables"] / "paper_b_process_stability_triage.csv"
        safe_to_csv(casebook[triage_cols], p)
        outputs.append(p)
    return outputs


def write_submission_readiness_report(metrics_df: pd.DataFrame, merged: pd.DataFrame, target_mapping: Dict[str, Optional[str]], families: Dict[str, List[str]]) -> Path:
    """Write a reviewer-facing automated readiness report.

    This is not a substitute for manual chemical interpretation. It tells the
    user which objective pieces are present and which final high-impact steps
    remain manual, especially structural rendering and mechanism assignment.
    """
    out = PATHS["qc"] / "SUBMISSION_READINESS_REPORT.md"
    lines: List[str] = []
    lines.append("# SUBMISSION_READINESS_REPORT")
    lines.append("")
    lines.append(f"Generated: {now()}")
    lines.append("")
    lines.append("## Automated checks completed")
    lines.append("")
    detected = {k: v for k, v in target_mapping.items() if v is not None}
    lines.append(f"- Detected adsorption targets: {len(detected)} / {len(TARGET_DEFINITIONS)}")
    for k, v in target_mapping.items():
        status = "OK" if v is not None else "MISSING"
        n = int(pd.to_numeric(merged[v], errors="coerce").notna().sum()) if v in merged.columns else 0
        lines.append(f"  - {k}: {status}; column={v}; nonmissing={n}")
    lines.append(f"- Feature families retained after redundancy filter: {len(families)}")
    lines.append(f"- Completed model jobs in current metrics table: {len(metrics_df) if metrics_df is not None else 0}")
    split_diag = PATHS["qc"] / "split_availability_diagnostics.csv"
    if split_diag.exists():
        try:
            sd = pd.read_csv(split_diag)
            available = sd[sd["status"].astype(str) == "available"]["split_type"].tolist()
            unavailable = sd[sd["status"].astype(str) != "available"][["split_type", "reason"]].to_dict("records")
            lines.append(f"- Available split types: {', '.join(available) if available else 'none'}")
            if unavailable:
                lines.append(f"- Unavailable split explanations: {unavailable}")
        except Exception:
            pass
    zero_report = PATHS["qc"] / "ZERO_VALUE_QC_REPORT.md"
    if zero_report.exists():
        lines.append("- Zero/near-zero QC report exists: qc_reports/ZERO_VALUE_QC_REPORT.md")
    filter_report = PATHS["qc"] / "posthoc_filter_sensitivity_report.csv"
    if filter_report.exists():
        lines.append("- Filter sensitivity report exists: qc_reports/posthoc_filter_sensitivity_report.csv")
    strict_external = PATHS["paper_b_tables"] / "paper_b_external_match_strict_validation.csv"
    if strict_external.exists():
        lines.append("- Strict external-match validation table exists: paper_b_impact_additions/tables/paper_b_external_match_strict_validation.csv")
        try:
            em = pd.read_csv(strict_external)
            if "external_match_confidence" in em.columns:
                lines.append("  - External-match confidence counts: " + json.dumps(em["external_match_confidence"].value_counts(dropna=False).to_dict(), default=str))
        except Exception:
            pass
    fig5b = PATHS["figure_data_main"] / "Figure_5b_positive_surprise_rate_by_pore_quintile.csv"
    if fig5b.exists():
        lines.append("- Improved Figure 5b source data exists: figure_data/main/Figure_5b_positive_surprise_rate_by_pore_quintile.csv")
    render_plan = PATHS["structural_cases"] / "v5_ranked_structure_rendering_plan.csv"
    if render_plan.exists():
        lines.append("- Ranked structure-rendering plan exists: structural_case_studies/v5_ranked_structure_rendering_plan.csv")
    lines.append("")
    lines.append("## Remaining manual high-impact steps")
    lines.append("")
    lines.append("1. Replace Figure 5d/Figure 6d placeholders with real CIF-rendered structural panels.")
    lines.append("2. Manually confirm exact/fuzzy CoRE/MOSAEC/MOFChecker matches before using the word validation; otherwise call them plausibility overlays.")
    lines.append("3. Choose 6--10 visually clear candidate structures from v5_ranked_structure_rendering_plan.csv for the main text.")
    lines.append("4. Interpret each positive-surprise case chemically: confinement, polar functionality, metal-node chemistry, topology, descriptor failure, or possible artefact.")
    lines.append("5. Keep the manuscript framing honest: the strongest result is risk-controlled triage and confined-pore residual exceptions, not a large population of certified elites.")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def stage_v4_qc_sensitivity_and_fixed_figures(merged: pd.DataFrame, target_mapping: Dict[str, Optional[str]], families: Dict[str, List[str]], metrics_df: pd.DataFrame) -> List[Path]:
    """v5 high-level reviewer-facing automatic improvement stage.

    This overrides the v4 function of the same name and adds a submission
    readiness report while retaining the v4 QC, sensitivity, fixed zero-count
    figure and CIF-extraction outputs.
    """
    outputs: List[Path] = []
    write_mode_configuration_report(); outputs.append(PATHS["qc"] / "run_mode_configuration.json")
    write_split_diagnostic_report(merged, families); outputs.append(PATHS["qc"] / "split_availability_diagnostics.csv")
    pred = load_representative_prediction(metrics_df) if metrics_df is not None and not metrics_df.empty else pd.DataFrame()
    zero_and_nearzero_diagnostics(merged, target_mapping, pred); outputs.append(PATHS["qc"] / "zero_and_nearzero_diagnostics.csv")
    if not pred.empty:
        filtered_sensitivity_report(pred); outputs.append(PATHS["qc"] / "posthoc_filter_sensitivity_report.csv")
        outputs.extend(make_fixed_zero_count_figure(pred))
        positive_surprise_quintile_rate_table(pred, f"positive_anomaly_{int(CFG.MAIN_COVERAGE*100)}")
        outputs.append(PATHS["figure_data_main"] / "Figure_5b_positive_surprise_rate_by_pore_quintile.csv")
    extract_casebook_cifs_if_possible(); outputs.append(PATHS["structural_cases"] / "structural_case_study_manifest.csv")
    outputs.append(write_submission_readiness_report(metrics_df, merged, target_mapping, families))
    return outputs




def figure_paper_b_plausibility_triage(casebook: pd.DataFrame) -> List[Path]:
    """Paper-B Figure, v5: candidate classes, strict overlays, process filters and mechanisms."""
    if casebook.empty:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=CFG.FIG_DPI)
    axes = axes.ravel()

    ax = axes[0]
    data = casebook["paper_b_case_class"].value_counts()
    data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_candidate_class_counts.csv")
    ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=7)
    annotate_bars(ax, fmt="{:.0f}", fontsize=7)
    style_axes(ax, "Candidate-level decision vocabulary", "Class", "Candidates")
    add_panel_label(ax, "a")

    ax = axes[1]
    if "external_match_confidence" in casebook.columns:
        data = casebook["external_match_confidence"].value_counts()
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_external_match_confidence_counts.csv")
        ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=25, ha="right", fontsize=7)
        annotate_bars(ax, fmt="{:.0f}", fontsize=7)
    else:
        overlay_cols = [c for c in ["core_mof_2024_match", "core_mof_2025_match", "core_mof_match", "mosaec_match", "mofchecker_match"] if c in casebook.columns]
        if overlay_cols:
            data = pd.Series({c.replace("_match", ""): int(casebook[c].astype(bool).sum()) for c in overlay_cols})
            data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_external_overlay_counts.csv")
            ax.bar(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
            ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=20, ha="right", fontsize=8)
            annotate_bars(ax, fmt="{:.0f}", fontsize=7)
    style_axes(ax, "Strict external-match confidence", "Overlay confidence", "Candidates")
    add_panel_label(ax, "b")

    ax = axes[2]
    if "process_proxy_status" in casebook.columns:
        data = casebook["process_proxy_status"].value_counts()
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_process_proxy_counts.csv")
        ax.barh(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=7)
        ax.invert_yaxis()
    style_axes(ax, "Post-certificate process proxy", "Candidates", "")
    add_panel_label(ax, "c")

    ax = axes[3]
    if "positive_surprise_mechanistic_family" in casebook.columns:
        data = casebook["positive_surprise_mechanistic_family"].value_counts().head(10)
        data.to_csv(PATHS["paper_b_figure_data"] / "PaperB_Figure_mechanistic_family_counts.csv")
        ax.barh(range(len(data)), data.values, color=bar_colors(data.index), edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=7)
        ax.invert_yaxis()
    style_axes(ax, "Mechanistic families for cases", "Candidates", "")
    add_panel_label(ax, "d")

    fig.suptitle("Paper-B addition. From conformal certificates to chemically plausible candidate triage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_current_figure(PATHS["paper_b_figures"] / "PaperB_Figure_plausibility_process_mechanism_triage")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    try:
        LOGGER.info("STAGE >>> DISCOVER_INPUTS")
        csv_files = discover_csv_files(CFG)
        if not csv_files:
            raise FileNotFoundError(
                "No ARC-MOF CSV files found. Expected files under DATA_ROOT/raw/arc_mof/ "
                "for example C:\\Projects\\Project_Data\\arc_core_mos_ver1\\data\\raw\\arc_mof\\v7_zenodo_16802743. "
                "You can pass --data_root C:\\Projects\\Project_Data\\arc_core_mos_ver1\\data."
            )
        LOGGER.info("CSV_FILES_FOUND | %d | %s", len(csv_files), ", ".join(p.name for p in csv_files))

        audit = stage_data_audit(csv_files)
        merged = build_merged_table(csv_files)
        merged = add_engineered_geometry_features(merged)
        safe_to_pickle_df(merged, PATHS["processed"] / "merged_modelling_table_with_engineered_features.pkl")
        if getattr(CFG, "SAVE_MERGED_ENGINEERED_CSV", False):
            safe_to_csv(merged, PATHS["processed"] / "merged_modelling_table_with_engineered_features.csv")

        target_mapping = detect_target_columns(merged)
        save_json(target_mapping, PATHS["processed"] / "detected_target_columns.json")
        LOGGER.info("TARGET_MAPPING | %s", target_mapping)

        families = build_feature_families(merged, target_mapping)
        save_json({k: v for k, v in families.items()}, PATHS["processed"] / "feature_families.json")
        LOGGER.info("FEATURE_FAMILIES | %s", {k: len(v) for k, v in families.items()})
        write_feature_and_target_tables(merged, target_mapping, families)
        write_split_diagnostic_report(merged, families)
        zero_and_nearzero_diagnostics(merged, target_mapping, None)
        write_mode_configuration_report()
        write_minimal_environment_file()

        if getattr(CFG, "AUDIT_ONLY", False):
            publication_manifest = write_publication_manifest()
            summary_path = write_run_summary(audit, target_mapping, families, pd.DataFrame())
            LOGGER.info("AUDIT_ONLY_COMPLETE | summary=%s manifest=%s", summary_path, publication_manifest)
            print("\nAUDIT ONLY DONE. Results are in:", PATHS["results"])
            print("Open:", summary_path)
            return

        if getattr(CFG, "FIGURES_ONLY", False):
            metrics_path = PATHS["metrics"] / "all_job_metrics.pkl"
            if not metrics_path.exists():
                metrics_path = PATHS["metrics"] / "all_job_metrics.csv"
            if not metrics_path.exists():
                raise FileNotFoundError("--figures_only was requested but all_job_metrics.pkl/csv was not found in the selected results directory.")
            metrics_df = pd.read_pickle(metrics_path) if metrics_path.suffix == ".pkl" else pd.read_csv(metrics_path)
            aggregates = stage_aggregate_outputs(metrics_df)
            stage_paper_b_impact_additions(metrics_df, merged, target_mapping)
            stage_v4_qc_sensitivity_and_fixed_figures(merged, target_mapping, families, metrics_df)
            stage_make_figures(metrics_df, aggregates)
            publication_manifest = write_publication_manifest()
            summary_path = write_run_summary(audit, target_mapping, families, metrics_df)
            LOGGER.info("FIGURES_ONLY_COMPLETE | summary=%s manifest=%s", summary_path, publication_manifest)
            print("\nFIGURES ONLY DONE. Results are in:", PATHS["results"])
            print("Open:", summary_path)
            return

        metrics_df = stage_run_all_models(merged, target_mapping, families)
        aggregates = stage_aggregate_outputs(metrics_df)
        paper_b_outputs = stage_paper_b_impact_additions(metrics_df, merged, target_mapping)
        stage_v4_qc_sensitivity_and_fixed_figures(merged, target_mapping, families, metrics_df)
        stage_make_figures(metrics_df, aggregates)
        structural_guide = write_structural_representation_guide()
        summary_path = write_run_summary(audit, target_mapping, families, metrics_df)
        publication_manifest = write_publication_manifest()

        LOGGER.info("STAGE >>> JOB_COMPLETE")
        LOGGER.info("Run summary: %s", summary_path)
        LOGGER.info("Publication output manifest: %s", publication_manifest)
        print("\nDONE. Results are in:", PATHS["results"])
        print("Open:", summary_path)
        print("Publication manifest:", publication_manifest)
    except Exception as e:
        msg = str(e)
        if isinstance(e, RuntimeError) and ("LOW_DISK_SPACE" in msg or "No space left" in msg):
            print("\nERROR: Disk/output limit reached before the pipeline could finish.")
            print(msg)
            print("\nRecommended cleanup: delete or move old results_balanced_models/predictions and results_balanced_models/models, then rerun with --prediction_output_mode minimal --n_jobs 1 --results_dirname results_paperB_disk_safe.")
        else:
            LOGGER.error("PIPELINE_FAILED | %s", e)
            LOGGER.error(traceback.format_exc())
            print("\nERROR: The pipeline failed. Check the latest log under results/logs/.")
        raise


if __name__ == "__main__":
    main()
