# Certificates, not Rankings: Conformal Anomaly Screening of Exceptional MOF Adsorbents — clean code package

This package contains the unique Python source files supplied for this project, separated from manuscript/Overleaf files and publication outputs.
## Authors

1. Mehrdad Asgari — Lucy Cavendish College, University of Cambridge
2. Natalia Lydia Papadantonaki — Lucy Cavendish College, University of Cambridge
3. Angelo Di Bella — Lucy Cavendish College, University of Cambridge
4. Shayan Abaei — Department of Chemical Engineering, Faculty of Engineering, University of Tehran
5. Hosein Alimardani — Faculty of Engineering, University of Tehran
6. Miguel Jorge — Department of Chemical and Process Engineering, University of Strathclyde
7. Ashleigh Fletcher — Department of Chemical and Process Engineering, University of Strathclyde

## Folder structure

- `src/` — executable/analysis Python scripts
- `requirements.txt` — third-party Python packages detected from imports
- `README.md` — this guide

## Included scripts

- `src/conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py`
- `src/prepare_final_publication_figures_tables_full.py`
- `src/finish_publication_package.py`

## Recommended use

1. Create and activate a dedicated Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Read the header/docstring of the main pipeline script before running it; the supplied scripts document expected ARC-MOF/related input filenames and output paths.
4. Keep raw databases outside this code ZIP unless a script explicitly expects them beside the script; when needed, place/copy the requested input files according to the script header or pass the corresponding command-line path option.
5. Run the main analysis pipeline first, then run publication-asset/post-processing scripts only after their required intermediate/source-data outputs exist.

## Project-specific notes

- `conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py` is the main restart-safe analysis pipeline.
- The other two scripts are publication-package/figure-table finishing utilities from the supplied project archive; some paths inside them are environment-specific and may need adjustment when reused outside the original build environment.

## Duplicate cleanup

- No exact duplicate Python scripts were found.

No generated figures, LaTeX build files, PDFs, raw databases, caches, or compiled Python files are included in this code-only package.
