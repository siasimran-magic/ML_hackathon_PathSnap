import sys
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status, Query
import json
import re
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import MODELS_DIR, TEST_DIR, DATA_DIR, TRAIN_DIR, OUTPUT_DIR
from src.data_loader import load_trip, project_coordinates, parse_route
from src.candidate_generator import SpatialArcIndex, extract_observation_features
from src.hmm_decoder import ViterbiMapMatcher
from src.metrics import compute_route_continuity, compute_sequence_performance
from src.visualizer import generate_interactive_folium_map
from api.schemas import (
    HealthResponse,
    PredictTripRequest,
    PredictRawRequest,
    ReconstructedRouteResponse
)

app = FastAPI(
    title="PathSnap Route Reconstruction API",
    description="High-Precision Machine Learning & HMM/Viterbi Route Reconstruction Engine for Trajectory Map-Matching",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static figures and EDA maps
EDA_DIR = OUTPUT_DIR / "public_maps"
EDA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
if EDA_DIR.exists():
    app.mount("/eda", StaticFiles(directory=str(EDA_DIR)), name="eda")
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

TEMPLATES_DIR = BASE_DIR / "api" / "templates"

# Global model state
MODEL_DATA = None
SCALER = None

def get_loaded_model():
    global MODEL_DATA, SCALER
    if MODEL_DATA is None:
        model_path = MODELS_DIR / "best_tuned_lgbm_model.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Tuned model artifact not found at {model_path}.")
        MODEL_DATA = joblib.load(model_path)
    if SCALER is None:
        scaler_path = MODELS_DIR / "feature_scaler.joblib"
        if scaler_path.exists():
            SCALER = joblib.load(scaler_path)
    return MODEL_DATA, SCALER

@app.get("/", tags=["UI"])
def serve_ui():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "PathSnap Map-Matching & Route Reconstruction API",
        "documentation": "/docs",
        "status": "online"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        service="PathSnap Pipeline",
        model_name="LightGBM Classifier (Bayesian Tuned)",
        model_family="Gradient Boosting",
        feature_count=13,
        uptime_status="online"
    )

@app.post("/predict/trip/{trip_id}", response_model=ReconstructedRouteResponse, tags=["Inference"])
def predict_stored_trip(
    trip_id: str,
    sigma_z: float = Query(4.07, gt=0),
    beta: float = Query(3.0, gt=0),
    alpha_ml: float = Query(0.70, ge=0, le=1)
):
    """
    Executes route reconstruction inference for a specified trip ID (e.g. from student_test or evaluator_test)
    without revealing or leaking any ground-truth route sequence.
    """
    t0 = time.time()
    
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise HTTPException(status_code=404, detail="Unknown trip ID")
    # Locate trip directory
    search_dirs = [TEST_DIR, DATA_DIR / "evaluator_test", TRAIN_DIR]
    trip_dir = None
    for d in search_dirs:
        if (d / trip_id).exists():
            trip_dir = d
            break
            
    if trip_dir is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip ID '{trip_id}' not found in registered datasets."
        )
        
    try:
        model_data, scaler = get_loaded_model()
        clf = model_data["model"]
        
        # Load trip components (without loading .route for inference)
        trip = load_trip(trip_dir, trip_id, include_ground_truth=False)
        track = trip["track"]
        timestamps = track[:, 2]
        gps_x = trip["track_x"]
        gps_y = trip["track_y"]
        arcs = trip["arcs"]
        
        # Spatial candidate search
        spatial_idx = SpatialArcIndex(arcs, trip["node_x"], trip["node_y"])
        X, _, meta = extract_observation_features(trip, spatial_idx, true_matched_arcs=None)
        
        # Classifier probability inference
        y_probs = clf.predict_proba(X)[:, 1]
        
        # Organize per observation
        cand_arcs_per_obs = []
        cand_probs_per_obs = []
        cand_dists_per_obs = []
        cand_proj_per_obs = []
        
        obs_indices = meta["obs_indices"]
        cand_arcs = meta["cand_arc_ids"]
        cand_dists = meta["cand_distances"]
        proj_ts = X[:, 4]
        
        total_obs = len(track)
        for i in range(total_obs):
            mask = (obs_indices == i)
            if np.any(mask):
                cand_arcs_per_obs.append(cand_arcs[mask])
                cand_probs_per_obs.append(y_probs[mask])
                cand_dists_per_obs.append(cand_dists[mask])
                cand_proj_per_obs.append(proj_ts[mask])
            else:
                raise ValueError(f"No candidate roads for observation {i}")
                
        # HMM Viterbi Decoding
        matcher = ViterbiMapMatcher(
            arcs=arcs,
            node_x=trip["node_x"],
            node_y=trip["node_y"],
            sigma_z=sigma_z,
            beta=beta,
            alpha_ml=alpha_ml
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
        
        continuity = compute_route_continuity(full_reconstructed_route, arcs)
        
        # Arc-set F1 calculation (evaluated against ground truth if present)
        gt_route = (parse_route(trip_dir / trip_id / f"{trip_id}.route")
                    if trip_dir != DATA_DIR / "evaluator_test" else np.array([], dtype=np.int64))
        if len(gt_route) > 0:
            seq_perf = compute_sequence_performance(full_reconstructed_route, gt_route)
            accuracy_pct = round(seq_perf["seq_f1"] * 100.0, 2)
            seq_prec = round(seq_perf["seq_precision"] * 100.0, 2)
            seq_rec = round(seq_perf["seq_recall"] * 100.0, 2)
            has_gt = True
        else:
            accuracy_pct = None
            seq_prec = None
            seq_rec = None
            has_gt = False
            
        # Extract reconstructed route node coordinates for downstream consumers/UI
        nodes = trip["nodes"]
        route_coords = []
        for arc_id in full_reconstructed_route:
            u, v = arcs[arc_id]
            route_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
        if len(full_reconstructed_route) > 0:
            last_v = arcs[full_reconstructed_route[-1], 1]
            route_coords.append([float(nodes[last_v, 1]), float(nodes[last_v, 0])])
            
        # Dynamically generate interactive map displaying the CURRENT REQUEST route
        dynamic_map_filename = f"dynamic_map_{trip_id}_{uuid4().hex}.html"
        dynamic_map_path = EDA_DIR / dynamic_map_filename
        generate_interactive_folium_map(
            trip_data=trip,
            pred_raw_arcs=None,
            pred_hmm_arcs=full_reconstructed_route,
            save_path=dynamic_map_path,
            continuity_pct=continuity * 100.0,
            seq_f1_pct=accuracy_pct,
            include_ground_truth=False,
            show_ground_truth=False
        )
        map_url = f"/eda/{dynamic_map_filename}?t={int(time.time() * 1000)}"
        elapsed = time.time() - t0
        
        segments = []
        for arc in full_reconstructed_route:
            if not segments or arcs[segments[-1][-1], 1] != arcs[arc, 0]:
                segments.append([])
            segments[-1].append(int(arc))
        return ReconstructedRouteResponse(
            trip_id=trip_id,
            status="partial" if matcher.disconnected_transitions else "success",
            disconnected_transition_count=len(matcher.disconnected_transitions),
            route_segments=segments,
            num_gps_observations=total_obs,
            num_reconstructed_arcs=len(full_reconstructed_route),
            reconstructed_route=full_reconstructed_route,
            route_continuity_pct=round(continuity * 100.0, 2),
            arc_set_f1_pct=accuracy_pct,
            sequence_precision_pct=seq_prec,
            sequence_recall_pct=seq_rec,
            has_ground_truth=has_gt,
            processing_time_sec=round(elapsed, 3),
            map_url=map_url,
            route_coordinates=None if matcher.disconnected_transitions else route_coords,
            message=("Route contains disconnected sections in the supplied network." if matcher.disconnected_transitions else "Route reconstructed successfully via LightGBM + Viterbi HMM.")
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/predict/raw", response_model=ReconstructedRouteResponse, tags=["Inference"])
def predict_raw_trajectory(req: PredictRawRequest):
    """
    Accepts raw JSON coordinates (track, nodes, arcs) and executes real-time route reconstruction.
    """
    t0 = time.time()
    try:
        model_data, _ = get_loaded_model()
        clf = model_data["model"]
        
        track = np.array(req.track, dtype=np.float64)
        nodes = np.array(req.nodes, dtype=np.float64)
        arcs = np.array(req.arcs, dtype=np.int64)
        
        if track.ndim != 2 or track.shape[1] != 3 or nodes.ndim != 2 or nodes.shape[1] != 2 or arcs.ndim != 2 or arcs.shape[1] != 2:
            raise HTTPException(status_code=422, detail="Expected track rows of 3, node rows of 2, and arc rows of 2 values")
        if not np.isfinite(track).all() or not np.isfinite(nodes).all():
            raise HTTPException(status_code=422, detail="Coordinates and timestamps must be finite")
        if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
            raise HTTPException(status_code=422, detail="Arc node index out of range")
        if (np.abs(track[:, 0]) > 180).any() or (np.abs(track[:, 1]) > 90).any() or (np.abs(nodes[:, 0]) > 180).any() or (np.abs(nodes[:, 1]) > 90).any():
            raise HTTPException(status_code=422, detail="Longitude or latitude out of range")
        if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
            raise HTTPException(status_code=400, detail="Empty track, nodes, or arcs provided.")
            
        origin_lon = float(np.mean(track[:, 0]))
        origin_lat = float(np.mean(track[:, 1]))
        
        track_x, track_y = project_coordinates(track[:, 0], track[:, 1], origin_lon, origin_lat)
        node_x, node_y = project_coordinates(nodes[:, 0], nodes[:, 1], origin_lon, origin_lat)
        
        trip_data = {
            "track": track,
            "track_x": track_x,
            "track_y": track_y,
            "nodes": nodes,
            "node_x": node_x,
            "node_y": node_y,
            "arcs": arcs,
            "route": np.array([], dtype=np.int64)
        }
        
        spatial_idx = SpatialArcIndex(arcs, node_x, node_y)
        X, _, meta = extract_observation_features(trip_data, spatial_idx, true_matched_arcs=None)
        
        y_probs = clf.predict_proba(X)[:, 1]
        
        cand_arcs_per_obs = []
        cand_probs_per_obs = []
        cand_dists_per_obs = []
        cand_proj_per_obs = []
        
        obs_indices = meta["obs_indices"]
        cand_arcs = meta["cand_arc_ids"]
        cand_dists = meta["cand_distances"]
        proj_ts = X[:, 4]
        
        total_obs = len(track)
        for i in range(total_obs):
            mask = (obs_indices == i)
            if np.any(mask):
                cand_arcs_per_obs.append(cand_arcs[mask])
                cand_probs_per_obs.append(y_probs[mask])
                cand_dists_per_obs.append(cand_dists[mask])
                cand_proj_per_obs.append(proj_ts[mask])
            else:
                raise ValueError(f"No candidate roads for observation {i}")
                
        matcher = ViterbiMapMatcher(
            arcs=arcs,
            node_x=node_x,
            node_y=node_y,
            sigma_z=req.sigma_z,
            beta=req.beta,
            alpha_ml=req.alpha_ml
        )
        
        decoded_obs_arcs, full_reconstructed_route = matcher.decode_trajectory(
            candidate_arcs_per_obs=cand_arcs_per_obs,
            candidate_probs_per_obs=cand_probs_per_obs,
            candidate_dists_per_obs=cand_dists_per_obs,
            candidate_proj_t_per_obs=cand_proj_per_obs,
            gps_timestamps=track[:, 2],
            gps_x=track_x,
            gps_y=track_y
        )
        
        continuity = compute_route_continuity(full_reconstructed_route, arcs)
        
        # Extract coordinates for UI
        route_coords = []
        for arc_id in full_reconstructed_route:
            u, v = arcs[arc_id]
            route_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
        if len(full_reconstructed_route) > 0:
            last_v = arcs[full_reconstructed_route[-1], 1]
            route_coords.append([float(nodes[last_v, 1]), float(nodes[last_v, 0])])
            
        # Dynamically generate interactive map for raw input
        dynamic_map_filename = f"dynamic_map_raw_{uuid4().hex}.html"
        dynamic_map_path = EDA_DIR / dynamic_map_filename
        generate_interactive_folium_map(
            trip_data=trip_data,
            pred_raw_arcs=None,
            pred_hmm_arcs=full_reconstructed_route,
            save_path=dynamic_map_path,
            continuity_pct=continuity * 100.0,
            seq_f1_pct=None,
            include_ground_truth=False,
            show_ground_truth=False
        )
        map_url = f"/eda/{dynamic_map_filename}?t={int(time.time() * 1000)}"
        elapsed = time.time() - t0
        
        segments = []
        for arc in full_reconstructed_route:
            if not segments or arcs[segments[-1][-1], 1] != arcs[arc, 0]:
                segments.append([])
            segments[-1].append(int(arc))
        return ReconstructedRouteResponse(
            trip_id="raw_input",
            status="partial" if matcher.disconnected_transitions else "success",
            disconnected_transition_count=len(matcher.disconnected_transitions),
            route_segments=segments,
            num_gps_observations=total_obs,
            num_reconstructed_arcs=len(full_reconstructed_route),
            reconstructed_route=full_reconstructed_route,
            route_continuity_pct=round(continuity * 100.0, 2),
            arc_set_f1_pct=None,
            sequence_precision_pct=None,
            sequence_recall_pct=None,
            has_ground_truth=False,
            processing_time_sec=round(elapsed, 3),
            map_url=map_url,
            route_coordinates=None if matcher.disconnected_transitions else route_coords,
            message="Raw trajectory reconstructed successfully."
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trips", tags=["UI"])
def list_trips():
    records = []
    for split, directory in (("student_test", TEST_DIR), ("evaluator_test", DATA_DIR / "evaluator_test")):
        if directory.exists():
            records.extend(dict(trip_id=p.name, split=split) for p in sorted(directory.iterdir())
                           if p.is_dir() and re.fullmatch(r"[0-9]{8}", p.name))
    return records


@app.get("/results", tags=["UI"])
def saved_results():
    benchmark = OUTPUT_DIR / "model_experimentation_results.json"
    comparison = OUTPUT_DIR / "bayesian_tuning_comparison.csv"
    import pandas as pd
    data = get_loaded_model()[0]
    return dict(benchmarks=json.loads(benchmark.read_text()) if benchmark.exists() else {},
                tuning=pd.read_csv(comparison).fillna(0).to_dict(orient="records") if comparison.exists() else [],
                best_params=data.get("best_params", {}))
