import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.config import OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, EDA_DIR, TEST_DIR
from src.data_loader import load_trip
from src.candidate_generator import SpatialArcIndex
from src.hmm_decoder import ViterbiMapMatcher
from src.metrics import compute_route_continuity, compute_sequence_performance
from src.visualizer import (
    plot_route_continuity_and_sequence_comparison,
    generate_interactive_folium_map
)

def run_route_reconstruction_pipeline() -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 5: HMM VITERBI ROUTE RECONSTRUCTION & CONTINUITY ANALYSIS")
    print("=" * 70)
    
    # Load tuned model and test dataset
    tuned_model_data = joblib.load(MODELS_DIR / "best_tuned_lgbm_model.joblib")
    model = tuned_model_data["model"]
    test_data = joblib.load(OUTPUT_DIR / "test_dataset.joblib")
    
    X_test = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    obs_ids_test = test_data["obs_ids"]
    cand_arcs_test = test_data["cand_arcs"]
    cand_dists_test = test_data["cand_dists"]
    
    # Compute tuned model probabilities for all test candidates
    print("Computing tuned model candidate probabilities...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Project ratios are feature index 4 in FEATURE_NAMES
    proj_ts_test = X_test[:, 4]
    
    df_test = pd.DataFrame({
        "trip_id": groups_test,
        "obs_id": obs_ids_test,
        "cand_arc": cand_arcs_test,
        "cand_dist": cand_dists_test,
        "proj_t": proj_ts_test,
        "prob": y_probs
    })
    
    unique_trips = sorted(list(np.unique(groups_test)))
    print(f"Reconstructing routes across {len(unique_trips)} student testing trips...")
    
    comparison_records = []
    
    for tid in unique_trips:
        t0 = time.time()
        trip_data = load_trip(TEST_DIR, tid)
        track = trip_data["track"]
        timestamps = track[:, 2]
        gps_x = trip_data["track_x"]
        gps_y = trip_data["track_y"]
        arcs = trip_data["arcs"]
        gt_route = trip_data["route"]
        
        t_df = df_test[df_test["trip_id"] == tid]
        
        # 1. Baseline: Classifier-Only (Top-1 candidate per observation)
        idx_max = t_df.groupby("obs_id")["prob"].idxmax()
        top1_cands = t_df.loc[idx_max].sort_values(by="obs_id")
        raw_pred_arcs = list(top1_cands["cand_arc"])
        
        continuity_raw = compute_route_continuity(raw_pred_arcs, arcs)
        seq_perf_raw = compute_sequence_performance(raw_pred_arcs, gt_route)
        
        # 2. HMM + Viterbi Decoder
        # Organize candidates per observation
        cand_arcs_per_obs = []
        cand_probs_per_obs = []
        cand_dists_per_obs = []
        cand_proj_per_obs = []
        
        obs_groups = t_df.groupby("obs_id")
        total_obs = len(track)
        
        for obs_i in range(total_obs):
            if obs_i in obs_groups.groups:
                g = obs_groups.get_group(obs_i)
                cand_arcs_per_obs.append(g["cand_arc"].values)
                cand_probs_per_obs.append(g["prob"].values)
                cand_dists_per_obs.append(g["cand_dist"].values)
                cand_proj_per_obs.append(g["proj_t"].values)
            else:
                raise ValueError(f"No candidates for {tid}, observation {obs_i}")
                
        # Initialize Viterbi matcher with trip's road network
        matcher = ViterbiMapMatcher(
            arcs=arcs,
            node_x=trip_data["node_x"],
            node_y=trip_data["node_y"],
            sigma_z=4.07,
            beta=3.0,
            alpha_ml=0.70
        )
        
        decoded_obs_arcs, full_reconstructed_route = matcher.decode_trajectory(
            candidate_arcs_per_obs=cand_arcs_per_obs,
            candidate_probs_per_obs=cand_probs_per_obs,
            candidate_dists_per_obs=cand_dists_per_obs,
            candidate_proj_t_per_obs=cand_proj_per_obs,
            gps_timestamps=timestamps,
            gps_x=gps_x,
            gps_y=gps_y
        )
        
        continuity_hmm = compute_route_continuity(decoded_obs_arcs, arcs)
        seq_perf_hmm = compute_sequence_performance(full_reconstructed_route, gt_route)
        
        elapsed = time.time() - t0
        print(f"Trip {tid} ({len(track)} GPS pts, {len(gt_route)} GT arcs, {elapsed:.2f}s):")
        print(f"  Classifier-Only: Continuity = {continuity_raw*100:.2f}%, Seq-F1 = {seq_perf_raw['seq_f1']*100:.2f}%")
        print(f"  HMM Viterbi:     Continuity = {continuity_hmm*100:.2f}%, Seq-F1 = {seq_perf_hmm['seq_f1']*100:.2f}%")
        
        comparison_records.append({
            "trip_id": tid,
            "num_gps": len(track),
            "num_gt_arcs": len(gt_route),
            "continuity_raw": continuity_raw,
            "continuity_hmm": continuity_hmm,
            "continuity_full_route": compute_route_continuity(full_reconstructed_route, arcs),
            "disconnected_transition_count": len(matcher.disconnected_transitions),
            "seq_f1_raw": seq_perf_raw["seq_f1"],
            "seq_f1_hmm": seq_perf_hmm["seq_f1"],
            "seq_prec_hmm": seq_perf_hmm["seq_precision"],
            "seq_rec_hmm": seq_perf_hmm["seq_recall"],
            "reconstructed_arc_count": len(full_reconstructed_route)
        })
        
        # Generate interactive map for each test trip with both routes and live stats
        generate_interactive_folium_map(
            trip_data=trip_data,
            pred_raw_arcs=raw_pred_arcs,
            pred_hmm_arcs=full_reconstructed_route,
            save_path=EDA_DIR / f"folium_map_comparison_{tid}.html",
            continuity_pct=continuity_hmm * 100.0,
            seq_f1_pct=seq_perf_hmm["seq_f1"] * 100.0
        )
            
    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(OUTPUT_DIR / "route_reconstruction_continuity_comparison.csv", index=False)
    
    # Plot final comparative graph
    plot_route_continuity_and_sequence_comparison(df_comp)
    
    print("\n" + "=" * 70)
    print("ROUTE CONTINUITY & SEQUENCE ACCURACY COMPARISON SUMMARY:")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print(f"\nMean Classifier-Only Continuity: {df_comp['continuity_raw'].mean() * 100:.2f} %")
    print(f"Mean HMM Viterbi Continuity:     {df_comp['continuity_hmm'].mean() * 100:.2f} %")
    print(f"Mean Classifier-Only Arc-set F1:     {df_comp['seq_f1_raw'].mean() * 100:.2f} %")
    print(f"Mean HMM Viterbi Arc-set F1:         {df_comp['seq_f1_hmm'].mean() * 100:.2f} %")
    
    return df_comp

if __name__ == "__main__":
    run_route_reconstruction_pipeline()
