import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from src.config import GAP_THRESHOLD_SECONDS, LARGE_GAP_THRESHOLD_SECONDS

class ViterbiMapMatcher:
    """
    Hidden Markov Model (HMM) with Viterbi Decoding for Road Network Route Reconstruction.
    Integrates classifier emission probabilities with road-network topological transition dynamics.
    Handles temporal GPS gaps and records disconnected transitions explicitly.
    """
    def __init__(
        self,
        arcs: np.ndarray,
        node_x: np.ndarray,
        node_y: np.ndarray,
        sigma_z: float = 4.07,
        beta: float = 3.0,
        alpha_ml: float = 0.65,
        max_dijkstra_dist: float = 1500.0
    ):
        if sigma_z <= 0 or beta <= 0 or not 0 <= alpha_ml <= 1:
            raise ValueError("Invalid decoder parameters")
        self.arcs = arcs
        self.node_x = node_x
        self.node_y = node_y
        self.sigma_z = sigma_z
        self.beta = beta
        self.alpha_ml = alpha_ml
        self.max_dijkstra_dist = max_dijkstra_dist
        
        # Precompute arc lengths
        ux = node_x[arcs[:, 0]]
        uy = node_y[arcs[:, 0]]
        vx = node_x[arcs[:, 1]]
        vy = node_y[arcs[:, 1]]
        self.arc_lengths = np.hypot(vx - ux, vy - uy)
        
        # Build NetworkX DiGraph for shortest path lookups
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(node_x)))
        edge_data = [(int(arcs[i, 0]), int(arcs[i, 1]), float(self.arc_lengths[i])) for i in range(len(arcs))]
        self.G.add_weighted_edges_from(edge_data)
        
        # Cache for shortest path queries (u, v) -> distance
        self._sp_cache: Dict[Tuple[int, int, float], float] = {}
        self._sp_path_cache: Dict[Tuple[int, int], List[int]] = {}
        self._ss_cache: Dict[Tuple[int, float], Dict[int, float]] = {}
        
        # Map directed edges (u, v) -> arc_id
        self.edge_to_arc: Dict[Tuple[int, int], int] = {}
        for a_id in range(len(arcs)):
            u, v = int(arcs[a_id, 0]), int(arcs[a_id, 1])
            self.edge_to_arc[(u, v)] = a_id

    def get_shortest_path_dist(self, u: int, v: int, max_dist_bound: float = 180.0) -> float:
        """Returns shortest path distance between nodes u and v, with caching and Euclidean bounding."""
        if u == v:
            return 0.0
        pair = (u, v, max_dist_bound)
        if pair in self._sp_cache:
            return self._sp_cache[pair]
            
        # Euclidean distance lower bound
        eucl_dist = float(np.hypot(self.node_x[u] - self.node_x[v], self.node_y[u] - self.node_y[v]))
        if eucl_dist > max_dist_bound:
            self._sp_cache[pair] = eucl_dist + 1000.0
            return self._sp_cache[pair]
            
        # Cache of single-source distances within radius
        source_key = (u, max_dist_bound)
        if source_key not in self._ss_cache:
            try:
                self._ss_cache[source_key] = nx.single_source_dijkstra_path_length(
                    self.G, u, cutoff=max_dist_bound, weight="weight"
                )
            except Exception:
                self._ss_cache[source_key] = {}
                
        d = self._ss_cache[source_key].get(v, eucl_dist + 1000.0)
        self._sp_cache[pair] = d
        return d

    def get_shortest_path_nodes(self, u: int, v: int) -> List[int]:
        """Returns shortest path node sequence between u and v, with caching."""
        if u == v:
            return [u]
        pair = (u, v)
        if pair in self._sp_path_cache:
            return self._sp_path_cache[pair]
            
        try:
            path = nx.dijkstra_path(self.G, u, v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
            
        self._sp_path_cache[pair] = path
        return path

    def decode_trajectory(
        self,
        candidate_arcs_per_obs: List[np.ndarray],
        candidate_probs_per_obs: List[np.ndarray],
        candidate_dists_per_obs: List[np.ndarray],
        candidate_proj_t_per_obs: List[np.ndarray],
        gps_timestamps: np.ndarray,
        gps_x: np.ndarray,
        gps_y: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Runs Viterbi dynamic programming in log-space.
        Returns:
            decoded_observation_arcs: arc ID for each observation
            reconstructed_full_route: ordered arcs with reachable transitions bridged;
                inspect disconnected_transitions before assuming full continuity
        """
        T = len(candidate_arcs_per_obs)
        if T == 0:
            return [], []
            
        if any(len(items) != T for items in (candidate_probs_per_obs, candidate_dists_per_obs, candidate_proj_t_per_obs)):
            raise ValueError("Candidate arrays have inconsistent observation counts")
        for t, candidates in enumerate(candidate_arcs_per_obs):
            if not len(candidates):
                raise ValueError(f"No candidates for observation {t}")
            if any(len(items[t]) != len(candidates) for items in (candidate_probs_per_obs, candidate_dists_per_obs, candidate_proj_t_per_obs)):
                raise ValueError(f"Candidate lengths disagree at observation {t}")
        # Viterbi tables
        # V[t] is array of log-probabilities for candidates at time t
        # backpointer[t] is array of parent candidate indices at time t-1
        V = []
        backpointers = []
        
        # Step 0: Initial state log probabilities
        cands_0 = candidate_arcs_per_obs[0]
        probs_0 = candidate_probs_per_obs[0]
        dists_0 = candidate_dists_per_obs[0]
        
        K0 = len(cands_0)
        v0 = np.zeros(K0, dtype=np.float64)
        
        eps = 1e-12
        for k in range(K0):
            p_ml = max(eps, float(probs_0[k]))
            d_spatial = float(dists_0[k])
            
            # Hybrid log emission
            log_emit_spatial = -0.5 * (d_spatial / self.sigma_z) ** 2
            log_emit_ml = np.log(p_ml)
            v0[k] = self.alpha_ml * log_emit_ml + (1.0 - self.alpha_ml) * log_emit_spatial
            
        V.append(v0)
        
        # Steps 1 to T-1
        for t in range(1, T):
            cands_prev = candidate_arcs_per_obs[t - 1]
            cands_curr = candidate_arcs_per_obs[t]
            
            probs_curr = candidate_probs_per_obs[t]
            dists_curr = candidate_dists_per_obs[t]
            proj_t_prev = candidate_proj_t_per_obs[t - 1]
            proj_t_curr = candidate_proj_t_per_obs[t]
            
            K_prev = len(cands_prev)
            K_curr = len(cands_curr)
            
            # Great circle / Euclidean distance between GPS observations
            dt = float(gps_timestamps[t] - gps_timestamps[t - 1])
            d_gps = float(np.hypot(gps_x[t] - gps_x[t - 1], gps_y[t] - gps_y[t - 1]))
            
            # Adaptive beta for time gaps
            beta_eff = self.beta * max(1.0, np.sqrt(max(1.0, dt)))
            
            vt = np.zeros(K_curr, dtype=np.float64)
            bp_t = np.zeros(K_curr, dtype=np.int32)
            
            for j in range(K_curr):
                arc_curr = cands_curr[j]
                u_curr = int(self.arcs[arc_curr, 0])
                v_curr = int(self.arcs[arc_curr, 1])
                len_curr = float(self.arc_lengths[arc_curr])
                t_curr = float(proj_t_curr[j])
                
                # Hybrid log emission for candidate j
                p_ml = max(eps, float(probs_curr[j]))
                d_spatial = float(dists_curr[j])
                log_emit = self.alpha_ml * np.log(p_ml) + (1.0 - self.alpha_ml) * (-0.5 * (d_spatial / self.sigma_z) ** 2)
                
                best_score = -np.inf
                best_parent = 0
                
                for i in range(K_prev):
                    arc_prev = cands_prev[i]
                    u_prev = int(self.arcs[arc_prev, 0])
                    v_prev = int(self.arcs[arc_prev, 1])
                    len_prev = float(self.arc_lengths[arc_prev])
                    t_prev = float(proj_t_prev[i])
                    
                    # Network routing distance
                    if arc_prev == arc_curr:
                        # Same arc
                        d_net = abs(t_curr - t_prev) * len_prev
                    elif v_prev == u_curr:
                        # Direct connected transition
                        d_net = (1.0 - t_prev) * len_prev + t_curr * len_curr
                    else:
                        # Dijkstra path from v_prev to u_curr
                        d_middle = self.get_shortest_path_dist(v_prev, u_curr)
                        d_net = (1.0 - t_prev) * len_prev + d_middle + t_curr * len_curr
                        
                    # Transition probability
                    log_trans = -abs(d_net - d_gps) / beta_eff
                    
                    candidate_score = V[t - 1][i] + log_trans
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_parent = i
                        
                vt[j] = best_score + log_emit
                bp_t[j] = best_parent
                
            if not np.isfinite(vt).any():
                raise ValueError(f"Disconnected candidate network at observation {t}")
            V.append(vt)
            backpointers.append(bp_t)
            
        # Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        decoded_indices = [best_last_idx]
        
        for t in range(T - 1, 0, -1):
            parent_idx = int(backpointers[t - 1][decoded_indices[-1]])
            decoded_indices.append(parent_idx)
            
        decoded_indices.reverse()
        
        # Map indices to arc IDs
        decoded_obs_arcs = [int(candidate_arcs_per_obs[t][decoded_indices[t]]) for t in range(T)]
        
        self.disconnected_transitions = []
        # Bridge transitions when a directed path exists; record remaining gaps.
        full_route = []
        for t in range(len(decoded_obs_arcs)):
            arc = decoded_obs_arcs[t]
            if len(full_route) == 0:
                full_route.append(arc)
            elif full_route[-1] == arc:
                continue
            else:
                prev_arc = full_route[-1]
                v_prev = int(self.arcs[prev_arc, 1])
                u_curr = int(self.arcs[arc, 0])
                
                if v_prev == u_curr:
                    full_route.append(arc)
                else:
                    # Bridge gap via Dijkstra shortest path nodes
                    sp_nodes = self.get_shortest_path_nodes(v_prev, u_curr)
                    if not sp_nodes:
                        self.disconnected_transitions.append((prev_arc, arc))
                    if len(sp_nodes) >= 2:
                        for step_i in range(len(sp_nodes) - 1):
                            edge_pair = (sp_nodes[step_i], sp_nodes[step_i + 1])
                            if edge_pair in self.edge_to_arc:
                                bridge_arc = self.edge_to_arc[edge_pair]
                                if full_route[-1] != bridge_arc:
                                    full_route.append(bridge_arc)
                    full_route.append(arc)
                    
        return decoded_obs_arcs, full_route
