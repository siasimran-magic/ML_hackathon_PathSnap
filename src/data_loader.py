import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from src.config import EARTH_RADIUS_METERS, GAP_THRESHOLD_SECONDS, LARGE_GAP_THRESHOLD_SECONDS

def project_coordinates(lon: np.ndarray, lat: np.ndarray, origin_lon: float, origin_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project angular geographical coordinates (longitude, latitude in degrees)
    to a metric Cartesian coordinate system (x, y in meters) using an equidistant
    cylindrical projection centered at the local origin.
    Preserves distance metrics accurately at local city scales.
    """
    lat0_rad = np.radians(origin_lat)
    lon0_rad = np.radians(origin_lon)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = EARTH_RADIUS_METERS * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS_METERS * (lat_rad - lat0_rad)
    return x, y

def unproject_coordinates(x: np.ndarray, y: np.ndarray, origin_lon: float, origin_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert metric Cartesian coordinates (x, y in meters) back to angular (lon, lat in degrees).
    """
    lat0_rad = np.radians(origin_lat)
    lon0_rad = np.radians(origin_lon)
    
    lat_rad = (y / EARTH_RADIUS_METERS) + lat0_rad
    lon_rad = (x / (EARTH_RADIUS_METERS * np.cos(lat0_rad))) + lon0_rad
    
    return np.degrees(lon_rad), np.degrees(lat_rad)

def point_to_segment_distance(px: np.ndarray, py: np.ndarray, 
                              ux: np.ndarray, uy: np.ndarray, 
                              vx: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the minimum Euclidean distance from 2D points (px, py)
    to line segments (ux, uy) -> (vx, vy) in metric coordinates,
    and returns (distance, projection_ratio_t).
    Vectorized across all points or pairs.
    """
    dx = vx - ux
    dy = vy - uy
    l2 = dx * dx + dy * dy
    
    # Handle zero-length segments
    safe_l2 = np.where(l2 == 0, 1.0, l2)
    t = ((px - ux) * dx + (py - uy) * dy) / safe_l2
    t = np.clip(t, 0.0, 1.0)
    t = np.where(l2 == 0, 0.0, t)
    
    proj_x = ux + t * dx
    proj_y = uy + t * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t

def parse_track(path: Path) -> np.ndarray:
    """
    Parses .track file.
    Format: longitude, latitude, timestamp
    Validates range and plausibility.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                lon, lat, ts = float(parts[0]), float(parts[1]), float(parts[2])
                # Integrity checks
                if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                    data.append([lon, lat, ts])
    return np.array(data, dtype=np.float64)

def parse_nodes(path: Path) -> np.ndarray:
    """
    Parses .nodes file.
    Format: longitude, latitude
    Line index represents 0-based node ID.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                data.append([lon, lat])
    return np.array(data, dtype=np.float64)

def parse_arcs(path: Path) -> np.ndarray:
    """
    Parses .arcs file.
    Format: from_node_id, to_node_id
    Line index represents 0-based arc ID.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                data.append([u, v])
    return np.array(data, dtype=np.int64)

def parse_route(path: Path) -> np.ndarray:
    """
    Parses .route file.
    Format: sequence of ground-truth arc IDs.
    """
    if not path.exists():
        return np.array([], dtype=np.int64)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(int(line))
    return np.array(data, dtype=np.int64)

def load_trip(trip_dir: Path, trip_id: str) -> Dict[str, Any]:
    """
    Ingests and validates all components of a single trip:
    track, nodes, arcs, and route (if available).
    """
    t_path = trip_dir / trip_id / f"{trip_id}.track"
    n_path = trip_dir / trip_id / f"{trip_id}.nodes"
    a_path = trip_dir / trip_id / f"{trip_id}.arcs"
    r_path = trip_dir / trip_id / f"{trip_id}.route"
    
    track = parse_track(t_path)
    nodes = parse_nodes(n_path)
    arcs = parse_arcs(a_path)
    route = parse_route(r_path)
    
    # Calculate origin centroid for local projection
    origin_lon = float(np.mean(track[:, 0])) if len(track) > 0 else float(np.mean(nodes[:, 0]))
    origin_lat = float(np.mean(track[:, 1])) if len(track) > 0 else float(np.mean(nodes[:, 1]))
    
    # Project track and nodes to Cartesian meters
    track_x, track_y = project_coordinates(track[:, 0], track[:, 1], origin_lon, origin_lat)
    node_x, node_y = project_coordinates(nodes[:, 0], nodes[:, 1], origin_lon, origin_lat)
    
    return {
        "trip_id": trip_id,
        "track": track, # [lon, lat, time]
        "track_x": track_x,
        "track_y": track_y,
        "nodes": nodes, # [lon, lat]
        "node_x": node_x,
        "node_y": node_y,
        "arcs": arcs,   # [u, v]
        "route": route, # [arc_id, ...]
        "origin_lon": origin_lon,
        "origin_lat": origin_lat
    }

def detect_anomalies(track: np.ndarray) -> Dict[str, Any]:
    """
    Calculates sampling intervals, identifies irregular gaps beyond nominal 1 Hz,
    and detects speed anomalies and GPS noise characteristics.
    """
    if len(track) < 2:
        return {
            "num_points": len(track),
            "dt_min": 0.0, "dt_mean": 0.0, "dt_max": 0.0, "dt_std": 0.0,
            "gaps_gt_2s": 0, "gaps_gt_5s": 0,
            "max_speed_kmh": 0.0, "mean_speed_kmh": 0.0,
            "speed_anomalies_gt_120kmh": 0
        }
        
    times = track[:, 2]
    dt = np.diff(times)
    
    # Calculate geodesic displacement between consecutive points
    lon, lat = track[:, 0], track[:, 1]
    origin_lon, origin_lat = float(np.mean(lon)), float(np.mean(lat))
    x, y = project_coordinates(lon, lat, origin_lon, origin_lat)
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.hypot(dx, dy)
    
    # Safe speed in km/h
    safe_dt = np.where(dt <= 0, 1e-3, dt)
    speed_mps = dist / safe_dt
    speed_kmh = speed_mps * 3.6
    
    return {
        "num_points": len(track),
        "dt_min": float(np.min(dt)),
        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_max": float(np.max(dt)),
        "dt_std": float(np.std(dt)),
        "gaps_gt_2s": int(np.sum(dt > GAP_THRESHOLD_SECONDS)),
        "gaps_gt_5s": int(np.sum(dt > LARGE_GAP_THRESHOLD_SECONDS)),
        "mean_speed_kmh": float(np.mean(speed_kmh)),
        "max_speed_kmh": float(np.max(speed_kmh)),
        "speed_anomalies_gt_120kmh": int(np.sum(speed_kmh > 120.0))
    }

def align_track_to_ground_truth(trip_data: Dict[str, Any]) -> np.ndarray:
    """
    Determines the ground-truth arc for each GPS observation along the trajectory
    by projecting points onto the ground-truth route sequence monotonically.
    """
    route = trip_data["route"]
    if len(route) == 0:
        return np.full(len(trip_data["track"]), -1, dtype=np.int64)
        
    arcs = trip_data["arcs"]
    node_x = trip_data["node_x"]
    node_y = trip_data["node_y"]
    track_x = trip_data["track_x"]
    track_y = trip_data["track_y"]
    
    route_u = arcs[route, 0]
    route_v = arcs[route, 1]
    ru_x, ru_y = node_x[route_u], node_y[route_u]
    rv_x, rv_y = node_x[route_v], node_y[route_v]
    
    n_track = len(track_x)
    n_route = len(route)
    
    # Vectorized distance matrix from each track point to each route arc
    # For large routes/tracks, compute in chunks or windowed
    matched_arcs = np.zeros(n_track, dtype=np.int64)
    curr_r = 0
    
    for i in range(n_track):
        px, py = track_x[i], track_y[i]
        # Search window from max(0, curr_r - 2) to min(n_route, curr_r + 25)
        w_start = max(0, curr_r - 2)
        w_end = min(n_route, curr_r + 30)
        
        # Calculate distances to window arcs
        w_ux, w_uy = ru_x[w_start:w_end], ru_y[w_start:w_end]
        w_vx, w_vy = rv_x[w_start:w_end], rv_y[w_start:w_end]
        
        dx = w_vx - w_ux
        dy = w_vy - w_uy
        l2 = dx * dx + dy * dy
        safe_l2 = np.where(l2 == 0, 1.0, l2)
        t = np.clip(((px - w_ux) * dx + (py - w_uy) * dy) / safe_l2, 0.0, 1.0)
        proj_x = w_ux + t * dx
        proj_y = w_uy + t * dy
        dists = np.hypot(px - proj_x, py - proj_y)
        
        best_local_idx = int(np.argmin(dists))
        best_global_r = w_start + best_local_idx
        
        curr_r = best_global_r
        matched_arcs[i] = route[best_global_r]
        
    return matched_arcs
