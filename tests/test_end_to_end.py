import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODELS_DIR, EDA_DIR
from src.data_loader import (
    load_trip,
    project_coordinates,
    unproject_coordinates,
    point_to_segment_distance,
    detect_anomalies,
    align_track_to_ground_truth
)
from src.candidate_generator import SpatialArcIndex, extract_observation_features, FEATURE_NAMES
from src.network_analyzer import analyze_road_network
from src.hmm_decoder import ViterbiMapMatcher
from src.metrics import (
    compute_classification_metrics,
    compute_observation_top1_accuracy,
    compute_route_continuity,
    compute_sequence_performance
)
from src.visualizer import generate_interactive_folium_map
from api.main import app

def run_e2e_tests():
    print("=" * 80)
    print("                 PATHSNAP END-TO-END VERIFICATION SUITE                 ")
    print("=" * 80)
    
    test_results = []
    
    def record_test(name: str, passed: bool, details: str = ""):
        status_str = "PASS" if passed else "FAIL"
        test_results.append((name, status_str, details))
        mark = "✓" if passed else "✗"
        print(f"[{mark}] {name:<55} -> {status_str} {details}")

    # --------------------------------------------------------------------------
    # SUITE 1: Data Parsing, Geodesic Projection & Anomaly Detection
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 1: Data Ingestion & Geographic Geometry] ---")
    try:
        trip = load_trip(TEST_DIR, "00000000")
        has_all_keys = all(k in trip for k in ["track", "nodes", "arcs", "route", "track_x", "track_y", "node_x", "node_y"])
        record_test("1.1 File Ingestion (.track, .nodes, .arcs, .route)", has_all_keys and len(trip["track"]) == 2503)
        
        # Test metric projection round-trip
        orig_lon, orig_lat = float(trip["track"][0, 0]), float(trip["track"][0, 1])
        x, y = project_coordinates(np.array([orig_lon]), np.array([orig_lat]), orig_lon, orig_lat)
        un_lon, un_lat = unproject_coordinates(x, y, orig_lon, orig_lat)
        proj_accurate = np.isclose(orig_lon, un_lon[0], atol=1e-7) and np.isclose(orig_lat, un_lat[0], atol=1e-7)
        record_test("1.2 Local Metric Projection & Round-trip Inversion", proj_accurate, f"Delta < 1e-7 deg")
        
        # Test anomaly detection
        anom = detect_anomalies(trip["track"])
        anom_valid = anom["num_points"] == 2503 and anom["dt_mean"] > 0 and "gaps_gt_2s" in anom
        record_test("1.3 Temporal Gap & Speed Anomaly Detection", anom_valid, f"Mean dt={anom['dt_mean']:.2f}s, Gaps>2s={anom['gaps_gt_2s']}")
    except Exception as e:
        record_test("Suite 1 Ingestion & Projection", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 2: Spatial Indexing & Candidate Generation
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 2: Spatial Indexing & Candidate Retrieval] ---")
    try:
        spatial_idx = SpatialArcIndex(trip["arcs"], trip["node_x"], trip["node_y"])
        # Query first GPS point
        px, py = float(trip["track_x"][0]), float(trip["track_y"][0])
        cands, dists, proj_ts = spatial_idx.query_point(px, py)
        idx_valid = len(cands) > 0 and len(dists) == len(cands) and np.all(dists <= 45.0)
        record_test("2.1 2D Spatial Grid Indexing (40m search disk)", idx_valid, f"Found {len(cands)} candidates for ping 0")
        
        # Test point-to-segment distance calculation
        ux, uy = np.array([0.0]), np.array([0.0])
        vx, vy = np.array([10.0]), np.array([0.0])
        px_t, py_t = np.array([5.0]), np.array([3.0])
        dist, t = point_to_segment_distance(px_t, py_t, ux, uy, vx, vy)
        p2s_correct = np.isclose(dist[0], 3.0) and np.isclose(t[0], 0.5)
        record_test("2.2 Orthogonal Point-to-Segment Math & Projection Ratio", p2s_correct, f"dist={dist[0]:.1f}m, t={t[0]:.1f}")
    except Exception as e:
        record_test("Suite 2 Spatial Indexing", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 3: Feature Engineering & Target Alignment
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 3: Feature Extraction & Ground-Truth Alignment] ---")
    try:
        true_arcs = align_track_to_ground_truth(trip)
        alignment_valid = len(true_arcs) == len(trip["track"]) and np.all(true_arcs >= 0)
        record_test("3.1 Ground-Truth Moving Window Trajectory Alignment", alignment_valid, f"Aligned {len(true_arcs)} observations")
        
        X, y, meta = extract_observation_features(trip, spatial_idx, true_matched_arcs=true_arcs)
        feat_valid = X.shape[1] == len(FEATURE_NAMES) and not np.any(np.isnan(X)) and not np.any(np.isinf(X))
        miss_rate_ok = meta["cand_miss_rate"] < 0.05
        record_test("3.2 13-Feature Extraction (No NaNs/Infs, valid dims)", feat_valid, f"Shape: {X.shape}")
        record_test("3.3 Candidate Miss Rate Benchmark (< 5%)", miss_rate_ok, f"Miss rate: {meta['cand_miss_rate']*100:.2f}%")
    except Exception as e:
        record_test("Suite 3 Feature Extraction", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 4: Machine Learning Model Artifacts & Probabilities
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 4: Tuned Machine Learning Model Artifacts] ---")
    try:
        model_path = MODELS_DIR / "best_tuned_lgbm_model.joblib"
        scaler_path = MODELS_DIR / "feature_scaler.joblib"
        files_exist = model_path.exists() and scaler_path.exists()
        record_test("4.1 Model & Scaler Artifact Existence", files_exist)
        
        model_data = joblib.load(model_path)
        clf = model_data["model"]
        y_probs = clf.predict_proba(X[:100])[:, 1]
        probs_calibrated = np.all((y_probs >= 0.0) & (y_probs <= 1.0))
        record_test("4.2 Tuned LightGBM Probability Inference", probs_calibrated, f"Sample mean prob={np.mean(y_probs):.4f}")
    except Exception as e:
        record_test("Suite 4 ML Artifacts", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 5: HMM Viterbi Decoder & Route Reconstruction
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 5: HMM Viterbi Sequence Reconstruction] ---")
    try:
        # Organize candidates for trip 00000000
        all_probs = clf.predict_proba(X)[:, 1]
        obs_indices = meta["obs_indices"]
        cand_arcs = meta["cand_arc_ids"]
        cand_dists = meta["cand_distances"]
        proj_ts = X[:, 4]
        
        cand_arcs_per_obs = []
        cand_probs_per_obs = []
        cand_dists_per_obs = []
        cand_proj_per_obs = []
        
        for i in range(len(trip["track"])):
            mask = (obs_indices == i)
            if np.any(mask):
                cand_arcs_per_obs.append(cand_arcs[mask])
                cand_probs_per_obs.append(all_probs[mask])
                cand_dists_per_obs.append(cand_dists[mask])
                cand_proj_per_obs.append(proj_ts[mask])
            else:
                cand_arcs_per_obs.append(np.array([0]))
                cand_probs_per_obs.append(np.array([1.0]))
                cand_dists_per_obs.append(np.array([0.0]))
                cand_proj_per_obs.append(np.array([0.5]))
                
        matcher = ViterbiMapMatcher(
            arcs=trip["arcs"],
            node_x=trip["node_x"],
            node_y=trip["node_y"],
            sigma_z=4.07,
            beta=3.0,
            alpha_ml=0.70
        )
        
        t0 = time.time()
        decoded_obs_arcs, full_route = matcher.decode_trajectory(
            candidate_arcs_per_obs=cand_arcs_per_obs,
            candidate_probs_per_obs=cand_probs_per_obs,
            candidate_dists_per_obs=cand_dists_per_obs,
            candidate_proj_t_per_obs=cand_proj_per_obs,
            gps_timestamps=trip["track"][:, 2],
            gps_x=trip["track_x"],
            gps_y=trip["track_y"]
        )
        dec_time = time.time() - t0
        
        continuity = compute_route_continuity(decoded_obs_arcs, trip["arcs"])
        seq_perf = compute_sequence_performance(full_route, trip["route"])
        
        continuity_ok = continuity >= 0.98
        seq_f1_ok = seq_perf["seq_f1"] >= 0.90
        record_test("5.1 Viterbi Decoding Runtime (< 3.0s)", dec_time < 3.0, f"Latency: {dec_time:.2f}s")
        record_test("5.2 Reconstructed Route Continuity (>= 98.0%)", continuity_ok, f"Continuity: {continuity*100:.2f}%")
        record_test("5.3 Reference Arc-set F1 (>= 90.0%)", seq_f1_ok, f"Arc-set F1: {seq_perf['seq_f1']*100:.2f}%")
    except Exception as e:
        record_test("Suite 5 HMM Viterbi", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 6: Metrics & Whole-Partition Grouping Integrity
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 6: Partition Grouping & Metric Integrity] ---")
    try:
        # Test whole-partition grouping: 2 trips with colliding obs_ids 0..2
        mock_trip_ids = np.array(["trip_A", "trip_A", "trip_B", "trip_B"])
        mock_obs_ids = np.array([0, 0, 0, 0])  # Same obs_id across trips
        mock_cand_arcs = np.array([10, 11, 20, 21])
        mock_y_true = np.array([1, 0, 0, 1])
        mock_prob = np.array([0.9, 0.1, 0.2, 0.8])
        
        top1_res = compute_observation_top1_accuracy(mock_obs_ids, mock_cand_arcs, mock_y_true, mock_prob, trip_ids=mock_trip_ids)
        grouping_correct = top1_res["total_obs"] == 2 and top1_res["top1_correct"] == 2 and np.isclose(top1_res["top1_accuracy"], 1.0)
        record_test("6.1 Whole-Partition Observation Grouping by (trip_id, obs_id)", grouping_correct, f"Evaluated distinct obs: {top1_res['total_obs']}")
    except Exception as e:
        record_test("Suite 6 Metrics", False, str(e))

    # --------------------------------------------------------------------------
    # SUITE 7: FastAPI Backend & Live Inference Endpoints
    # --------------------------------------------------------------------------
    print("\n--- [SUITE 7: FastAPI Endpoints & Dynamic Map Generation] ---")
    try:
        client = TestClient(app)
        
        # 1. Health check
        h_res = client.get("/health")
        h_ok = h_res.status_code == 200 and h_res.json()["status"] == "healthy"
        record_test("7.1 GET /health (Health & Status Check)", h_ok, f"Status: {h_res.status_code}")
        
        # 2. UI Root
        ui_res = client.get("/")
        ui_ok = ui_res.status_code == 200 and "PathSnap" in ui_res.text
        record_test("7.2 GET / (Interactive Web UI Delivery)", ui_ok, f"Status: {ui_res.status_code}")
        
        # 3. Dynamic Trip Prediction with custom parameters
        p_res = client.post("/predict/trip/00000000?sigma_z=4.07&alpha_ml=0.70")
        p_ok = p_res.status_code == 200
        p_data = p_res.json()
        pred_valid = (
            p_ok and 
            p_data["trip_id"] == "00000000" and
            p_data["route_continuity_pct"] >= 98.0 and
            p_data["arc_set_f1_pct"] >= 90.0 and
            p_data["num_reconstructed_arcs"] > 0 and
            "map_url" in p_data and
            len(p_data["reconstructed_route"]) > 0
        )
        record_test("7.3 POST /predict/trip/{trip_id} (Dynamic Prediction & Map URL)", pred_valid, 
                    f"Cont={p_data.get('route_continuity_pct')}% | Seq-F1={p_data.get('arc_set_f1_pct')}%")
        
        # 4. Check dynamic map file on disk
        map_path = OUTPUT_DIR / "public_maps" / p_data["map_url"].split("/")[-1].split("?")[0]
        map_exists = map_path.exists()
        # Verify no CartoDB watermarks in the dynamic map
        map_content = map_path.read_text(encoding="utf-8")
        zero_watermarks = "carto.com/basemaps/apikey" not in map_content and "cartocdn.com" not in map_content
        has_osm = "tile.openstreetmap.org" in map_content
        record_test("7.4 Dynamic Folium Map Generation (Zero Watermarks, OSM default)", map_exists and zero_watermarks and has_osm)
        
        # 5. Raw Trajectory Prediction
        sample_track = trip["track"][:25].tolist()
        sample_nodes = trip["nodes"].tolist()
        sample_arcs = trip["arcs"].tolist()
        raw_payload = {
            "track": sample_track,
            "nodes": sample_nodes,
            "arcs": sample_arcs,
            "sigma_z": 4.07,
            "beta": 3.0,
            "alpha_ml": 0.70
        }
        raw_res = client.post("/predict/raw", json=raw_payload)
        raw_ok = raw_res.status_code == 200 and raw_res.json()["num_reconstructed_arcs"] > 0
        record_test("7.5 POST /predict/raw (Arbitrary JSON Trajectory Ingestion)", raw_ok, f"Arcs: {raw_res.json().get('num_reconstructed_arcs')}")
        
        # 6. Invalid Trip ID Error Handling
        err_res = client.post("/predict/trip/nonexistent_trip_999")
        err_ok = err_res.status_code == 404
        record_test("7.6 404 Error Guard for Unregistered Trips", err_ok, f"HTTP {err_res.status_code} caught cleanly")
    except Exception as e:
        record_test("Suite 7 FastAPI Endpoints", False, str(e))

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                           VERIFICATION SUMMARY                         ")
    print("=" * 80)
    passed_count = sum(1 for _, status, _ in test_results if status == "PASS")
    total_count = len(test_results)
    
    for name, status, details in test_results:
        print(f"  {status:<6} | {name:<55} | {details}")
        
    print("-" * 80)
    print(f"TOTAL TESTS: {total_count}  |  PASSED: {passed_count}  |  FAILED: {total_count - passed_count}")
    print(f"CHECK RESULT: {'ALL CHECKS PASSED' if passed_count == total_count else 'FAILURES DETECTED'}")
    print("=" * 80)
    return total_count - passed_count

if __name__ == "__main__":
    sys.exit(1 if run_e2e_tests() else 0)
