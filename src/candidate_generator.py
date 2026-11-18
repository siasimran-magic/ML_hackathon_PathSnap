import numpy as np
from typing import List, Dict, Any, Tuple
from src.config import (
    CANDIDATE_SEARCH_RADIUS_METERS,
    CANDIDATE_CELL_SIZE_METERS,
    MAX_CANDIDATES_PER_OBS,
    MIN_CANDIDATES_PER_OBS
)

class SpatialArcIndex:
    """
    High-performance 2D spatial grid index for road network arcs.
    Enables sub-millisecond candidate arc queries with exact perpendicular distance calculation.
    """
    def __init__(self, arcs: np.ndarray, node_x: np.ndarray, node_y: np.ndarray, cell_size: float = CANDIDATE_CELL_SIZE_METERS):
        self.arcs = arcs
        self.node_x = node_x
        self.node_y = node_y
        self.cell_size = cell_size
        
        self.ux = node_x[arcs[:, 0]]
        self.uy = node_y[arcs[:, 0]]
        self.vx = node_x[arcs[:, 1]]
        self.vy = node_y[arcs[:, 1]]
        
        self.dx = self.vx - self.ux
        self.dy = self.vy - self.uy
        self.arc_lengths = np.hypot(self.dx, self.dy)
        self.arc_headings = np.degrees(np.arctan2(self.dy, self.dx)) % 360.0
        
        # Build spatial grid
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        min_ax = np.minimum(self.ux, self.vx)
        max_ax = np.maximum(self.ux, self.vx)
        min_ay = np.minimum(self.uy, self.vy)
        max_ay = np.maximum(self.uy, self.vy)
        
        min_cxs = np.floor(min_ax / self.cell_size).astype(int)
        max_cxs = np.floor(max_ax / self.cell_size).astype(int)
        min_cys = np.floor(min_ay / self.cell_size).astype(int)
        max_cys = np.floor(max_ay / self.cell_size).astype(int)
        
        for a in range(len(arcs)):
            for cx in range(min_cxs[a], max_cxs[a] + 1):
                for cy in range(min_cys[a], max_cys[a] + 1):
                    cell = (cx, cy)
                    if cell not in self.grid:
                        self.grid[cell] = []
                    self.grid[cell].append(a)

    def query_point(self, px: float, py: float, search_radius: float = CANDIDATE_SEARCH_RADIUS_METERS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Queries all candidate arcs within search_radius of (px, py).
        Returns:
            sorted_arc_ids: ndarray of candidate arc indices
            sorted_distances: ndarray of perpendicular distances (meters)
            sorted_proj_ratios: ndarray of projection fractions t in [0, 1]
        """
        cell_rad = int(np.ceil(search_radius / self.cell_size))
        pcx = int(np.floor(px / self.cell_size))
        pcy = int(np.floor(py / self.cell_size))
        
        cand_set = set()
        for cx in range(pcx - cell_rad, pcx + cell_rad + 1):
            for cy in range(pcy - cell_rad, pcy + cell_rad + 1):
                cell = (cx, cy)
                if cell in self.grid:
                    cand_set.update(self.grid[cell])
                    
        cand_list = list(cand_set)
        if len(cand_list) == 0:
            # Fallback search with doubled radius if isolated
            cand_set = set()
            for cx in range(pcx - cell_rad * 2, pcx + cell_rad * 2 + 1):
                for cy in range(pcy - cell_rad * 2, pcy + cell_rad * 2 + 1):
                    cell = (cx, cy)
                    if cell in self.grid:
                        cand_set.update(self.grid[cell])
            cand_list = list(cand_set)
            
        if len(cand_list) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            
        c_ux = self.ux[cand_list]
        c_uy = self.uy[cand_list]
        c_dx = self.dx[cand_list]
        c_dy = self.dy[cand_list]
        c_l2 = c_dx * c_dx + c_dy * c_dy
        safe_l2 = np.where(c_l2 == 0, 1.0, c_l2)
        
        t = np.clip(((px - c_ux) * c_dx + (py - c_uy) * c_dy) / safe_l2, 0.0, 1.0)
        t = np.where(c_l2 == 0, 0.0, t)
        
        proj_x = c_ux + t * c_dx
        proj_y = c_uy + t * c_dy
        dists = np.hypot(px - proj_x, py - proj_y)
        
        order = np.argsort(dists)
        sorted_cand = np.array(cand_list, dtype=np.int64)[order]
        sorted_dist = dists[order]
        sorted_t = t[order]
        
        # Filter within search_radius, retaining at least MIN_CANDIDATES_PER_OBS and at most MAX_CANDIDATES_PER_OBS
        within_r = sorted_dist <= search_radius
        num_within = int(np.sum(within_r))
        
        k = max(MIN_CANDIDATES_PER_OBS, min(num_within, MAX_CANDIDATES_PER_OBS))
        k = min(k, len(sorted_cand))
        
        return sorted_cand[:k], sorted_dist[:k], sorted_t[:k]

def extract_observation_features(
    trip_data: Dict[str, Any],
    spatial_index: SpatialArcIndex,
    true_matched_arcs: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extracts tabular features for all (observation, candidate) pairs for a single trip.
    Returns:
        X: Feature matrix of shape (N_pairs, N_features)
        y: Binary labels (1 = ground-truth arc, 0 = non-matching candidate), or None if test trip
        metadata: Dictionary containing obs_ids, candidate_arc_ids, candidate_counts, candidate_miss_count
    """
    track_x = trip_data["track_x"]
    track_y = trip_data["track_y"]
    track = trip_data["track"]
    times = track[:, 2]
    n_pts = len(track_x)
    
    # Compute trajectory headings and speeds
    dx_track = np.diff(track_x, prepend=track_x[0])
    dy_track = np.diff(track_y, prepend=track_y[0])
    dt_track = np.diff(times, prepend=times[0] - 1.0)
    dt_track = np.where(dt_track <= 0, 1.0, dt_track)
    
    disp_track = np.hypot(dx_track, dy_track)
    speed_mps = disp_track / dt_track
    heading_track = np.degrees(np.arctan2(dy_track, dx_track)) % 360.0
    
    # Forward difference for heading when speed > 1 m/s, otherwise inherit
    for i in range(n_pts - 1):
        if disp_track[i] < 0.5 and i < n_pts - 1:
            heading_track[i] = heading_track[i + 1]
            
    features = []
    labels = []
    obs_indices = []
    cand_arc_ids = []
    cand_distances = []
    
    cand_misses = 0
    total_obs = n_pts
    
    has_labels = true_matched_arcs is not None and len(true_matched_arcs) == n_pts
    
    for i in range(n_pts):
        px, py = track_x[i], track_y[i]
        cands, dists, proj_ts = spatial_index.query_point(px, py)
        
        true_arc = true_matched_arcs[i] if has_labels else -1
        
        if has_labels:
            if true_arc not in cands:
                cand_misses += 1
                
        for rank, (cand_arc, dist, proj_t) in enumerate(zip(cands, dists, proj_ts)):
            arc_len = float(spatial_index.arc_lengths[cand_arc])
            arc_head = float(spatial_index.arc_headings[cand_arc])
            
            # Heading difference in [0, 180]
            raw_diff = abs(heading_track[i] - arc_head) % 360.0
            heading_diff = min(raw_diff, 360.0 - raw_diff)
            cos_heading_diff = float(np.cos(np.radians(heading_diff)))
            
            # Distance to endpoints
            ux_val, uy_val = spatial_index.ux[cand_arc], spatial_index.uy[cand_arc]
            vx_val, vy_val = spatial_index.vx[cand_arc], spatial_index.vy[cand_arc]
            dist_to_u = float(np.hypot(px - ux_val, py - uy_val))
            dist_to_v = float(np.hypot(px - vx_val, py - vy_val))
            
            feat_row = [
                float(dist),                    # 0: perpendicular distance (m)
                float(heading_diff),            # 1: heading diff degrees [0, 180]
                float(cos_heading_diff),        # 2: cos(heading diff) [-1, 1]
                float(arc_len),                 # 3: segment length (m)
                float(proj_t),                  # 4: projection ratio along arc [0, 1]
                float(dist_to_u),               # 5: distance to start node u (m)
                float(dist_to_v),               # 6: distance to end node v (m)
                float(speed_mps[i]),            # 7: current GPS speed (m/s)
                float(dt_track[i]),             # 8: delta time from prev point (s)
                float(disp_track[i]),           # 9: displacement from prev point (m)
                float(rank + 1),                # 10: spatial distance rank (1 = closest)
                float(1.0 if dist < 10.0 else 0.0), # 11: proximity flag < 10m
                float(1.0 if heading_diff < 30.0 else 0.0) # 12: heading alignment flag < 30 deg
            ]
            
            features.append(feat_row)
            obs_indices.append(i)
            cand_arc_ids.append(cand_arc)
            cand_distances.append(dist)
            
            if has_labels:
                labels.append(1 if cand_arc == true_arc else 0)
                
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if has_labels else None
    
    metadata = {
        "obs_indices": np.array(obs_indices, dtype=np.int64),
        "cand_arc_ids": np.array(cand_arc_ids, dtype=np.int64),
        "cand_distances": np.array(cand_distances, dtype=np.float32),
        "total_obs": total_obs,
        "cand_misses": cand_misses,
        "cand_miss_rate": (cand_misses / total_obs) if total_obs > 0 else 0.0
    }
    
    return X, y, metadata

FEATURE_NAMES = [
    "dist_to_arc_m",
    "heading_diff_deg",
    "cos_heading_diff",
    "arc_length_m",
    "proj_ratio",
    "dist_to_u_m",
    "dist_to_v_m",
    "gps_speed_mps",
    "delta_t_s",
    "delta_dist_m",
    "spatial_rank",
    "is_within_10m",
    "is_aligned_30deg"
]
