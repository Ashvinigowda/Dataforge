# Team Name: DataForge

## Team

- **Team name:** DataForge
- **Members:** Kruthika S L, Ashwini L, Manasa BP
- **Track(s) addressed:** Data Insights & Visualization
- **Contact email:** kruthilokesh49@gmail.com
- **Language used:** Python

## Solution Overview

This project analyzes three years of Grade 4–6 mathematics assessment data (2022–23, 2023–24, and 2024–25) to identify trends in student performance across districts, blocks, grades, and competencies. The solution includes an interactive Streamlit dashboard, descriptive analytics, and predictive models to support educational planning and intervention.

## What's in this repo

| File / folder | Purpose |
|---|---|
| `report.pdf` | Final project report |
| `slides.pptx` | Presentation deck |
| `docs/policy_note.pdf` | Policy recommendations |
| `src/run_all.py` | Entry point for reproducing outputs |
| `requirements.txt` | Python dependencies |
| `data/` | Local dataset folder (not committed) |
| `manifest.yml` | Output manifest |
| `claims.json` | Verifiable claims list |
| `outputs/` | Generated figures, tables, dashboard outputs, and predictions |

## How to run

```bash
pip install -r requirements.txt
python src/run_all.py
```

The project expects the dataset to be placed locally inside:

```text
data/
├── 2022_23/
├── 2023_24/
└── 2024_25/
```

The dataset is intentionally excluded from this public repository.

## Notes for Reviewers

- The repository excludes the organizer dataset because it contains student assessment records.
- All paths are relative.
- Random seeds are fixed where applicable for reproducibility.
- The dashboard is implemented using Streamlit.
