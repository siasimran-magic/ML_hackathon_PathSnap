# PathSnap

GPS map matching with LightGBM and HMM/Viterbi. Run commands from this directory, containing src and api. Verified Python version: 3.12.

## Run on this laptop

```powershell
.\run_project.ps1 serve
```

Open http://localhost:8000 (API docs: http://localhost:8000/docs). The trip list includes student-test and reserved evaluator trips. Evaluator mode sends no reference route geometry or scores to the UI. An isolated public-map directory prevents access to older analysis maps.

## Set up a second laptop

Copy the source, reports, outputs/models, result tables and figures. Transfer raw data separately: Git intentionally excludes it. Inside this directory, keep `Student folder/train`, `Student folder/student_test` and `Student folder/evaluator_test`. Keep both saved model files in outputs/models.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_project.ps1 test
.\run_project.ps1 serve
```

No activation or execution-policy change is needed. To use external data, set `$env:PATHSNAP_DATA_DIR = 'D:\PathSnapData'` to the directory containing all three splits. Use the included freshly fitted model with the pinned environment, rather than the original incompatible artifact.

For a two-laptop demo, run the server on laptop A and open `http://<laptop-A-LAN-IP>:8000` on laptop B on the same trusted network. Allow the Python server through a firewall only if local policy permits it. The service has no authentication; stop it after the demo. Styles, scripts and map tiles require internet access.

## Reproduce

```powershell
.\run_project.ps1 preprocess
.\run_project.ps1 benchmark
.\run_project.ps1 tune
.\run_project.ps1 routes
.\run_project.ps1 test
.\run_project.ps1 evaluator
```

Preprocessing is needed only when data/features change. Benchmarking and tuning take several minutes. Tuning resumes outputs/bayesian_study.sqlite3 and adds 15 trials per call; CSV/JSON histories are saved after each trial. Archive the study file first for an intentionally new independent study. The original unsaved optimisation history cannot be recovered; the delivered history belongs to the fresh search. Refresh all downstream reports when changing the final model. Evaluator trips must never guide tuning or model selection.

## Preprocessing and evaluation

.track rows are longitude, latitude, timestamp. .nodes row numbers are node IDs; .arcs rows are directed from/to node IDs; .route contains zero-based reference arc IDs. Coordinates are projected to local metres. A grid index searches a nominal 40 m radius, retains up to eight candidate roads, and extracts 13 geometric/motion features. Training labels use an approximate moving-window alignment to the reference route; candidate recall is relative to these derived labels. Missing candidates are rejected explicitly rather than assigned arc 0.

All five models use three-fold GroupKFold over 80 training trips; scalers fit only the fold's training rows. Fold manifests are in outputs/grouped_cv_folds.csv. KNN uses a deterministic 15,000-row training subsample per fold, with complete validation folds. MLP has a fixed 40-epoch limit and no row-level internal validation. Tuning uses three-fold grouped CV over a fixed random subset of 40 training trips, then refits on all 80. Neither student-test nor evaluator trips enter fitting or tuning.

Candidate binary F1, candidate macro F1, observation top-1 accuracy and route arc-set F1 are distinct metrics. Top-1 groups by (trip_id, obs_id). Legacy seq_f1 keys mean arc-set overlap, ignoring traversal order and repeated arcs. Observation continuity can skip intermediate arcs; full-route continuity includes connecting arcs. Disconnected routes expose separate connected sections, a gap count and partial status rather than a false continuity claim.

## Evidence and repository

Results are in outputs/model_experimentation_results.csv, grouped_cv_folds.csv, bayesian_search_history.json, bayesian_tuning_comparison.csv, route_reconstruction_continuity_comparison.csv and evaluator/. Tests return nonzero on failures; timing varies by hardware. The consolidated PDF and genuine demo captures are under reports/.

Keep source paths unchanged in one shared repository. Each contributor should review and commit their own work under their own identity. Check `git diff --cached --name-only` before pushing. .gitignore excludes raw data, dataset caches, dynamic maps, runtimes and Python caches. No commits or pushes are performed by this task.

To rebuild the consolidated report after refreshing results, run `python scripts/build_report.py` in the pinned environment (ReportLab is included).
