# Review fixes

- Portable paths: project-relative config, optional PATHSNAP_DATA_DIR override.
- Runtime: freshly trained final artifact under pinned Python 3.12 / scikit-learn 1.4.2 / LightGBM 4.6.0; original model retained only in an ignored archive.
- Reference isolation: live maps omit reference geometry; evaluator requests never read .route labels or return scores. Analysis maps are outside the public static mount.
- UI: saved-data-driven tuning table and hyperparameters, correct arc-set F1 labels, all ten reserved evaluator trips, explicit network-gap count and partial status.
- Evaluation: all five models have three-fold grouped CV, fold-specific scaling, binary and macro F1, fold trip manifests and refreshed student-test metrics/figures.
- Optimisation: a new 15-trial search saves full JSON/CSV trial records and resumable SQLite state. The original unsaved history cannot be reconstructed.
- Routes: re-evaluated using the final artifact. Mean classifier continuity is 89.85%, matching the tuning table. Gaps in the supplied directed graph are exposed as connected sections; no continuous coordinate line is returned across a gap.
- Tests: failure exit codes, eight regression checks and twenty end-to-end checks; saved evidence under outputs/.
- Delivery: README, pinned requirements, .gitignore, PowerShell launcher, consolidated Markdown/PDF and two genuine UI screenshots.

The MLP uses a fixed 40-epoch budget and emits a convergence warning. KNN uses a 15,000-row training subsample. Both limitations are disclosed. Final offline evaluator mean arc-set F1 remains 92.33%; no evaluator tuning was performed.

No repository commits, pushes, contributor impersonation or external submission were performed. Follow the README for the two-laptop demo and repository staging guidance.
