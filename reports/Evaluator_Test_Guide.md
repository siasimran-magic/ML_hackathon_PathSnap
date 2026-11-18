# Evaluator data

All ten reserved trips are extracted under Student folder/evaluator_test, separate from training and student-test data. Run `.\run_project.ps1 evaluator` only to regenerate predictions after an intentional model or code change. Evaluation is already complete.

The frozen final model predicts without loading reference .route files. Predicted routes are persisted before labels are read for offline scoring. API requests for evaluator trips remain blind and omit both labels and scores. Reserved data must never be used for tuning or selecting another model.

Current evidence: outputs/evaluator/summary.json, per_trip_metrics.csv and predicted_routes/. Mean arc-set F1 is 92.33%; all ten exported routes are fully continuous. Arc-set F1 ignores traversal order and repeated arcs. The model hash and measured runtime are in summary.json.

See PathSnap_Technical_Report.md/PDF for refreshed model, validation and runtime details. The final artifact was freshly fitted under the pinned runtime, resolving the original serialization-version mismatch. No external submission destination or additional route-file format was supplied.
