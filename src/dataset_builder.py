import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List

from src.config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data_loader import load_trip, align_track_to_ground_truth, detect_anomalies
from src.candidate_generator import SpatialArcIndex, extract_observation_features, FEATURE_NAMES

def build_dataset(
    split_dir: Path,
    trip_ids: List[str],
    has_ground_truth: bool = True
) -> Dict[str, Any]:
    """
    Iterates through trip directories, extracts spatial candidate arcs,
    computes tabular geometric and kinematic features, and aggregates into
    a structured dataset preserving trip groupings for GroupKFold validation.
    """
    all_X = []
    all_y = []
    all_groups = []
    all_obs_ids = []
    all_cand_arcs = []
    all_cand_dists = []
    all_trip_obs_counts = {}
    
    trip_summaries = []
    
    t0 = time.time()
    print(f"Building dataset for {len(trip_ids)} trips from {split_dir}...")
    
    for idx, tid in enumerate(trip_ids):
        t_start = time.time()
        trip_data = load_trip(split_dir, tid)
        
        # Check anomalies
        anom = detect_anomalies(trip_data["track"])
        
        # Align ground truth if available
        true_arcs = align_track_to_ground_truth(trip_data) if has_ground_truth else None
        
        # Spatial indexing
        spatial_idx = SpatialArcIndex(trip_data["arcs"], trip_data["node_x"], trip_data["node_y"])
        
        # Feature extraction
        X, y, meta = extract_observation_features(trip_data, spatial_idx, true_arcs)
        
        n_pairs = len(X)
        all_X.append(X)
        if y is not None:
            all_y.append(y)
        all_groups.append(np.full(n_pairs, tid))
        all_obs_ids.append(meta["obs_indices"])
        all_cand_arcs.append(meta["cand_arc_ids"])
        all_cand_dists.append(meta["cand_distances"])
        
        all_trip_obs_counts[tid] = meta["total_obs"]
        
        trip_summaries.append({
            "trip_id": tid,
            "num_gps": meta["total_obs"],
            "num_candidates": n_pairs,
            "avg_candidates_per_pt": round(n_pairs / max(1, meta["total_obs"]), 2),
            "cand_miss_rate_pct": round(meta["cand_miss_rate"] * 100.0, 2),
            "dt_mean": anom["dt_mean"],
            "dt_max": anom["dt_max"],
            "gaps_gt_2s": anom["gaps_gt_2s"],
            "gaps_gt_5s": anom["gaps_gt_5s"],
            "max_speed_kmh": anom["max_speed_kmh"],
            "num_nodes": len(trip_data["nodes"]),
            "num_arcs": len(trip_data["arcs"]),
            "num_route_arcs": len(trip_data["route"])
        })
        
        if (idx + 1) % 10 == 0 or idx == len(trip_ids) - 1:
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(trip_ids)} trips ({elapsed:.1f}s elapsed)...")
            
    X_mat = np.vstack(all_X) if all_X else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y_vec = np.concatenate(all_y) if all_y else np.empty(0, dtype=np.int32)
    groups_vec = np.concatenate(all_groups) if all_groups else np.empty(0, dtype=object)
    obs_ids_vec = np.concatenate(all_obs_ids) if all_obs_ids else np.empty(0, dtype=np.int64)
    cand_arcs_vec = np.concatenate(all_cand_arcs) if all_cand_arcs else np.empty(0, dtype=np.int64)
    cand_dists_vec = np.concatenate(all_cand_dists) if all_cand_dists else np.empty(0, dtype=np.float32)
    
    print(f"Completed! Total rows: {len(X_mat)}, Features: {X_mat.shape[1]}")
    if len(y_vec) > 0:
        pos = int(np.sum(y_vec == 1))
        neg = int(np.sum(y_vec == 0))
        print(f"Class distribution: Positive={pos} ({pos/len(y_vec)*100:.1f}%), Negative={neg} ({neg/len(y_vec)*100:.1f}%)")
        
    return {
        "X": X_mat,
        "y": y_vec,
        "groups": groups_vec,
        "obs_ids": obs_ids_vec,
        "cand_arcs": cand_arcs_vec,
        "cand_dists": cand_dists_vec,
        "feature_names": FEATURE_NAMES,
        "trip_summaries": pd.DataFrame(trip_summaries),
        "trip_obs_counts": all_trip_obs_counts
    }

def process_and_save_all_datasets() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds and persists both training (80 trips) and student-testing (10 trips) datasets.
    """
    train_trips = sorted([d for d in os.listdir(TRAIN_DIR) if (TRAIN_DIR / d).is_dir()])
    test_trips = sorted([d for d in os.listdir(TEST_DIR) if (TEST_DIR / d).is_dir()])
    
    print(f"Found {len(train_trips)} training trips and {len(test_trips)} testing trips.")
    
    train_data = build_dataset(TRAIN_DIR, train_trips, has_ground_truth=True)
    test_data = build_dataset(TEST_DIR, test_trips, has_ground_truth=True)
    
    # Save datasets
    train_save_path = OUTPUT_DIR / "train_dataset.joblib"
    test_save_path = OUTPUT_DIR / "test_dataset.joblib"
    
    print(f"Saving train dataset to {train_save_path}...")
    joblib.dump(train_data, train_save_path, compress=3)
    
    print(f"Saving test dataset to {test_save_path}...")
    joblib.dump(test_data, test_save_path, compress=3)
    
    # Save EDA summary statistics
    eda_summary_path = OUTPUT_DIR / "eda" / "dataset_inventory.csv"
    combined_summary = pd.concat([
        train_data["trip_summaries"].assign(split="train"),
        test_data["trip_summaries"].assign(split="student_test")
    ], ignore_index=True)
    combined_summary.to_csv(eda_summary_path, index=False)
    print(f"Saved dataset inventory to {eda_summary_path}")
    
    return train_data, test_data

if __name__ == "__main__":
    process_and_save_all_datasets()
