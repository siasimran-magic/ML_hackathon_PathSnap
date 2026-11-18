"""Refresh the baseline tuning row and audit consistency of the final evidence."""
import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUT_DIR as OUT, EVALUATOR_DIR, MODELS_DIR
from src.data_loader import load_trip, parse_route
from src.metrics import compute_route_continuity


def read_csv(name):
    with (OUT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    results = json.loads((OUT / 'model_experimentation_results.json').read_text())
    folds = read_csv('grouped_cv_folds.csv')
    assert len(results) == 5 and len(folds) == 15
    for model in results:
        seen = set()
        for fold in [r for r in folds if r['model'] == model]:
            train = set(fold['train_trip_ids'].split(';'))
            validation = set(fold['validation_trip_ids'].split(';'))
            assert not train & validation and len(train | validation) == 80
            assert not seen & validation
            seen |= validation
        assert len(seen) == 80
        assert np.isfinite(results[model]['metrics']['macro_f1'])
    comparison = read_csv('bayesian_tuning_comparison.csv')
    baseline = results['LightGBM']['metrics']
    for field in comparison[0]:
        if field in baseline:
            comparison[0][field] = baseline[field]
    with (OUT / 'bayesian_tuning_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    artifact = joblib.load(MODELS_DIR / 'best_tuned_lgbm_model.joblib')
    route_rows = read_csv('route_reconstruction_continuity_comparison.csv')
    assert np.isclose(np.mean([float(r['continuity_raw']) for r in route_rows]), artifact['metrics']['route_continuity'])
    assert np.isclose(np.mean([float(r['seq_f1_raw']) for r in route_rows]), artifact['metrics']['seq_f1'])
    history = json.loads((OUT / 'bayesian_search_history.json').read_text())
    assert all(r['state'] == 'COMPLETE' and len(r['fold_scores']) == 3 for r in history)
    model_hash = hashlib.sha256((MODELS_DIR / 'best_tuned_lgbm_model.joblib').read_bytes()).hexdigest()
    evaluator = json.loads((OUT / 'evaluator/summary.json').read_text())
    assert evaluator['model_sha256'] == model_hash
    assert evaluator['ground_truth_used_for_inference'] is False
    predictions = sorted((OUT / 'evaluator/predicted_routes').glob('*.route'))
    assert len(predictions) == 10
    for file in predictions:
        trip = load_trip(EVALUATOR_DIR, file.stem, include_ground_truth=False)
        route = parse_route(file)
        assert len(route) and route.min() >= 0 and route.max() < len(trip['arcs'])
        assert compute_route_continuity(route, trip['arcs']) == 1
    assert '8 passed' in (OUT / 'review_tests.txt').read_text(encoding='utf-8')
    assert 'TOTAL TESTS: 20  |  PASSED: 20  |  FAILED: 0' in (OUT / 'end_to_end_verification.txt').read_text(encoding='utf-8')
    files = [p for directory in ['src', 'api', 'tests', 'scripts'] for p in (ROOT / directory).rglob('*.py')]
    for file in files:
        compile(file.read_text(encoding='utf-8'), str(file), 'exec')
    evidence = dict(status='passed', model_sha256=model_hash, benchmark_models=5, grouped_cv_folds=15,
                    tuning_trials=len(history), evaluator_routes_verified=10,
                    regression_checks=8, end_to_end_checks=20, python_files_compiled=len(files),
                    student_classifier_continuity=artifact['metrics']['route_continuity'],
                    python=platform.python_version(),
                    versions={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'optuna', 'fastapi']})
    (OUT / 'final_verification.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
