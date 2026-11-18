import numpy as np
import networkx as nx
from typing import Dict, Any, List
from pathlib import Path

def analyze_road_network(nodes: np.ndarray, arcs: np.ndarray, node_x: np.ndarray, node_y: np.ndarray) -> Dict[str, Any]:
    """
    Performs comprehensive structural and graph-topological analysis of the road network:
    - Node & Arc cardinality
    - Isolated nodes
    - Weakly and Strongly Connected Components
    - One-way vs two-way road reciprocity
    - Road segment physical length distribution (meters)
    """
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    
    # Calculate physical arc lengths
    ux, uy = node_x[arcs[:, 0]], node_y[arcs[:, 0]]
    vx, vy = node_x[arcs[:, 1]], node_y[arcs[:, 1]]
    arc_lengths = np.hypot(vx - ux, vy - uy)
    
    # Construct NetworkX DiGraph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # Add arcs with length weights
    edge_list = [(int(arcs[i, 0]), int(arcs[i, 1]), float(arc_lengths[i])) for i in range(n_arcs)]
    G.add_weighted_edges_from(edge_list)
    
    # Degree analysis
    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])
    total_degrees = in_degrees + out_degrees
    
    isolates = np.sum(total_degrees == 0)
    dead_ends = np.sum((in_degrees > 0) & (out_degrees == 0)) + np.sum((in_degrees == 0) & (out_degrees > 0))
    
    # Connected components
    num_wcc = nx.number_weakly_connected_components(G)
    wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
    largest_wcc_size = max(wcc_sizes) if wcc_sizes else 0
    wcc_coverage = largest_wcc_size / max(1, n_nodes)
    
    num_scc = nx.number_strongly_connected_components(G)
    
    # Arc reciprocity (two-way roads)
    # Check if (v, u) exists for edge (u, v)
    arc_set = set(map(tuple, arcs))
    reciprocal_count = sum(1 for (u, v) in arc_set if (v, u) in arc_set)
    reciprocity = reciprocal_count / max(1, len(arc_set))
    
    return {
        "num_nodes": n_nodes,
        "num_arcs": n_arcs,
        "num_isolates": int(isolates),
        "num_dead_ends": int(dead_ends),
        "num_wcc": int(num_wcc),
        "num_scc": int(num_scc),
        "largest_wcc_size": int(largest_wcc_size),
        "wcc_coverage_pct": float(wcc_coverage * 100.0),
        "reciprocity_pct": float(reciprocity * 100.0),
        "mean_arc_length_m": float(np.mean(arc_lengths)) if len(arc_lengths) > 0 else 0.0,
        "median_arc_length_m": float(np.median(arc_lengths)) if len(arc_lengths) > 0 else 0.0,
        "max_arc_length_m": float(np.max(arc_lengths)) if len(arc_lengths) > 0 else 0.0,
        "min_arc_length_m": float(np.min(arc_lengths)) if len(arc_lengths) > 0 else 0.0,
    }
