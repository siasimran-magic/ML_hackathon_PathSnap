"""Evaluate the frozen model: python -m src.evaluate_evaluator."""
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import EVALUATOR_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, OUTPUT_DIR
from src.data_loader import load_trip, parse_route
from src.candidate_generator import SpatialArcIndex, extract_observation_features
from src.hmm_decoder import ViterbiMapMatcher
from src.metrics import compute_route_continuity, compute_sequence_performance


def main():
    trip_ids = sorted(p.name for p in EVALUATOR_DIR.iterdir() if p.is_dir())
    if not trip_ids:
        raise ValueError('No evaluator trips found')
    for split in (TRAIN_DIR, TEST_DIR):
        overlap = set(trip_ids) & {p.name for p in split.iterdir() if p.is_dir()}
        if overlap:
            raise ValueError(f'Evaluator IDs overlap {split}: {overlap}')
    model_path = MODELS_DIR / 'best_tuned_lgbm_model.joblib'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = joblib.load(model_path)['model']
    output = OUTPUT_DIR / 'evaluator'
    routes = output / 'predicted_routes'
    routes.mkdir(parents=True, exist_ok=True)
    records = []
    for tid in trip_ids:
        started = time.perf_counter()
        # Ground truth is excluded from feature generation and decoding.
        trip = load_trip(EVALUATOR_DIR, tid, include_ground_truth=False)
        index = SpatialArcIndex(trip['arcs'], trip['node_x'], trip['node_y'])
        X, _, meta = extract_observation_features(trip, index, true_matched_arcs=None)
        probabilities = model.predict_proba(X)[:, 1]
        grouped = [[], [], [], []]
        raw = []
        for obs in range(len(trip['track'])):
            mask = meta['obs_indices'] == obs
            if not np.any(mask):
                raise ValueError(f'{tid}: observation {obs} has no candidates')
            candidates = meta['cand_arc_ids'][mask]
            probs = probabilities[mask]
            raw.append(int(candidates[np.argmax(probs)]))
            for bucket, values in zip(grouped, (candidates, probs, meta['cand_distances'][mask], X[mask, 4])):
                bucket.append(values)
        matcher = ViterbiMapMatcher(trip['arcs'], trip['node_x'], trip['node_y'], sigma_z=4.07, beta=3.0, alpha_ml=0.70)
        decoded, route = matcher.decode_trajectory(
            candidate_arcs_per_obs=grouped[0], candidate_probs_per_obs=grouped[1],
            candidate_dists_per_obs=grouped[2], candidate_proj_t_per_obs=grouped[3],
            gps_timestamps=trip['track'][:, 2], gps_x=trip['track_x'], gps_y=trip['track_y'])
        if not route or np.any(np.asarray(route) < 0) or np.any(np.asarray(route) >= len(trip['arcs'])):
            raise ValueError(f'{tid}: invalid predicted route')
        (routes / f'{tid}.route').write_text(''.join(f'{int(a)}\n' for a in route))
        elapsed = time.perf_counter() - started
        # Read labels only after predictions are persisted.
        truth = parse_route(EVALUATOR_DIR / tid / f'{tid}.route')
        score = compute_sequence_performance(route, truth)
        records.append(dict(trip_id=tid, num_gps=len(trip['track']), num_gt_arcs=len(truth),
                            predicted_arcs=len(route), inference_seconds=elapsed,
                            continuity_raw=compute_route_continuity(raw, trip['arcs']),
                            continuity_hmm=compute_route_continuity(decoded, trip['arcs']),
                            continuity_full_route=compute_route_continuity(route, trip['arcs']),
                            arc_set_f1_raw=compute_sequence_performance(raw, truth)['seq_f1'],
                            arc_set_precision=score['seq_precision'], arc_set_recall=score['seq_recall'],
                            arc_set_f1=score['seq_f1']))
        print(f'{tid}: arc-set F1={score["seq_f1"]:.4f}, {elapsed:.2f}s', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'per_trip_metrics.csv', index=False)
    summary = dict(trips=len(records), gps_points=int(frame.num_gps.sum()), model_sha256=model_hash,
                   parameters=dict(sigma_z=4.07, beta=3.0, alpha_ml=0.70),
                   mean_arc_set_f1=float(frame.arc_set_f1.mean()),
                   mean_raw_arc_set_f1=float(frame.arc_set_f1_raw.mean()),
                   mean_observation_continuity=float(frame.continuity_hmm.mean()),
                   mean_full_route_continuity=float(frame.continuity_full_route.mean()),
                   total_inference_seconds=float(frame.inference_seconds.sum()),
                   metric_note='Arc-set F1 ignores traversal order and repeated arcs; it is not order-sensitive sequence accuracy.',
                   ground_truth_used_for_inference=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
