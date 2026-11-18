import os
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.config import FIGURES_DIR, EDA_DIR

# Set clean professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def plot_confusion_matrices(results: Dict[str, Any], y_test: np.ndarray, save_path: Path = None):
    """
    Plots side-by-side or grid confusion matrices for all evaluated models.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrices_all_models.png"
        
    model_names = list(results.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]
        
    for i, name in enumerate(model_names):
        ax = axes[i]
        y_prob = results[name]["y_prob"]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalized by row (recall per class)
        cm_norm = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))
        
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-match (0)", "Match (1)"])
        ax.set_yticklabels(["Non-match (0)", "Match (1)"])
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Ground Truth")
            
        # Annotate text
        for r in range(2):
            for c in range(2):
                cnt = cm[r, c]
                pct = cm_norm[r, c] * 100.0
                text_color = "white" if cm_norm[r, c] > 0.5 else "black"
                ax.text(c, r, f"{cnt:,}\n({pct:.1f}%)", ha="center", va="center",
                        color=text_color, fontweight="semibold", fontsize=10)
                
    fig.suptitle("Candidate Arc Classification: Normalized Confusion Matrices", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")

def plot_timing_comparison(results: Dict[str, Any], save_path: Path = None):
    """
    Plots comparative bar charts for training time (seconds) and inference speed (samples/second).
    """
    if save_path is None:
        save_path = FIGURES_DIR / "timing_and_speed_comparison.png"
        
    names = list(results.keys())
    train_times = [results[m]["train_time_sec"] for m in names]
    infer_speeds = [results[m]["infer_speed_samples_sec"] for m in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    
    # Train time
    colors1 = ["#2b5c8f", "#3c8dbc", "#46a546", "#f39c12", "#9b59b6"][:len(names)]
    bars1 = ax1.bar(names, train_times, color=colors1, edgecolor="black", alpha=0.88, width=0.55)
    ax1.set_title("Training Wall-Clock Time (Lower is Faster)", fontweight="bold")
    ax1.set_ylabel("Training Time (Seconds)")
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_yscale("log")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # Inference speed
    colors2 = ["#34495e", "#16a085", "#27ae60", "#e67e22", "#8e44ad"][:len(names)]
    bars2 = ax2.bar(names, infer_speeds, color=colors2, edgecolor="black", alpha=0.88, width=0.55)
    ax2.set_title("Inference Throughput (Higher is Faster)", fontweight="bold")
    ax2.set_ylabel("Throughput (Candidate Pairs / Second)")
    ax2.set_xticklabels(names, rotation=20, ha="right")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.02, f"{yval:,.0f}/s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    fig.suptitle("Computational Efficiency & Scalability Across 5 Algorithm Families", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved timing comparison to {save_path}")

def plot_roc_pr_curves(results: Dict[str, Any], y_test: np.ndarray, save_path: Path = None):
    """
    Plots ROC curves and Precision-Recall curves across all 5 models.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "roc_and_pr_curves_all_models.png"
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # Baseline PR ratio
    baseline_pr = float(np.mean(y_test))
    
    for i, (name, res) in enumerate(results.items()):
        y_prob = res["y_prob"]
        color = palette[i % len(palette)]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = res["metrics"]["roc_auc"]
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2.2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = res["metrics"]["pr_auc"]
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=2.2)
        
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (0.50)")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.legend(loc="lower right", frameon=True)
    
    ax2.plot([0, 1], [baseline_pr, baseline_pr], "k--", alpha=0.6, label=f"Class Prevalence ({baseline_pr:.2f})")
    ax2.set_title("Precision-Recall (PR) Curve", fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.legend(loc="lower left", frameon=True)
    
    fig.suptitle("Discrimination Performance on Candidate Arc Selection", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC/PR curves to {save_path}")

def plot_metrics_comparison_across_models(results: Dict[str, Any], save_path: Path = None):
    """
    Plots a multi-metric radar or grouped bar chart comparing the 5 models across:
    Accuracy, Precision, Recall, F1, Observation Top-1 Accuracy, and ROC-AUC.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "metric_comparison_5_models.png"
        
    metric_keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "top1_accuracy", "roc_auc"]
    metric_labels = ["Candidate accuracy", "Precision", "Recall", "Binary F1", "Macro F1", "Obs Top-1 Acc", "ROC-AUC"]
    
    names = list(results.keys())
    data = []
    for m in names:
        row = [results[m]["metrics"][k] for k in metric_keys]
        data.append(row)
        
    df_metrics = pd.DataFrame(data, index=names, columns=metric_labels)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(13, 6))
    
    x = np.arange(len(metric_labels))
    width = 0.15
    palette = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
    
    for i, name in enumerate(names):
        offset = (i - len(names) / 2 + 0.5) * width
        vals = df_metrics.loc[name].values
        rects = ax.bar(x + offset, vals, width, label=name, color=palette[i % len(palette)], edgecolor="black", alpha=0.88)
        
    ax.set_ylabel("Score [0.0 - 1.0]", fontweight="bold")
    ax.set_title("Performance Benchmarking Across 5 Algorithmic Families (Student Test Set)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="semibold")
    ax.set_ylim([0.0, 1.08])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", frameon=True, ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved metric comparison to {save_path}")

def plot_route_continuity_and_sequence_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path = None
):
    """
    Plots Route Continuity and Arc-set F1 comparison before vs after HMM Viterbi decoding.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "route_continuity_classifier_vs_hmm.png"
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Continuity
    x = np.arange(len(comparison_df))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, comparison_df["continuity_raw"] * 100, width, label="Classifier-Only (Top-1)", color="#e74c3c", alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x + width/2, comparison_df["continuity_hmm"] * 100, width, label="Classifier + HMM Viterbi", color="#2ecc71", alpha=0.85, edgecolor="black")
    
    ax1.set_ylabel("Route Continuity (%)", fontweight="bold")
    ax1.set_title("Sequential Road Network Continuity", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["trip_id"], rotation=45, ha="right")
    ax1.set_ylim([0, 105])
    ax1.legend(loc="lower right", frameon=True)
    
    # Arc-set F1
    rects3 = ax2.bar(x - width/2, comparison_df["seq_f1_raw"] * 100, width, label="Classifier-Only (Top-1)", color="#e67e22", alpha=0.85, edgecolor="black")
    rects4 = ax2.bar(x + width/2, comparison_df["seq_f1_hmm"] * 100, width, label="Classifier + HMM Viterbi", color="#3498db", alpha=0.85, edgecolor="black")
    
    ax2.set_ylabel("Route Arc-set F1 (%)", fontweight="bold")
    ax2.set_title("Ground-Truth Route Arc-set F1-Score", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df["trip_id"], rotation=45, ha="right")
    ax2.set_ylim([0, 105])
    ax2.legend(loc="lower right", frameon=True)
    
    fig.suptitle("Impact of HMM Viterbi Sequential Decoding on Topological Coherence", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved route continuity comparison to {save_path}")

def generate_interactive_folium_map(
    trip_data: Dict[str, Any],
    pred_raw_arcs: List[int] = None,
    pred_hmm_arcs: List[int] = None,
    save_path: Path = None,
    continuity_pct: float = None,
    seq_f1_pct: float = None,
    include_ground_truth: bool = False,
    show_ground_truth: bool = False
) -> str:
    """
    Generates a high-precision, interactive Folium/Leaflet visualization map:
    - Multiple base map layers (OpenStreetMap, CartoDB Light/Dark, Esri Satellite)
    - Auto-fitted bounding box fitting the entire trajectory
    - Local road network graph geometry (corridor arcs)
    - Raw noisy GPS observations connected by a subtle cyan sensor trace
    - Ground-truth traversed route (gold)
    - Classifier-only predictions (red dashed)
    - Classifier + HMM Viterbi reconstructed route (emerald green)
    - Animated start/finish marker pins
    - Floating glassmorphism HUD legend with live trip diagnostics
    - Fullscreen & Distance measurement controls
    """
    import branca
    from folium import plugins
    
    if save_path is None:
        save_path = EDA_DIR / f"interactive_map_{trip_data['trip_id']}.html"
        
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data.get("route", np.array([]))
    trip_id = trip_data.get("trip_id", "Unknown")
    
    track_lons = track[:, 0]
    track_lats = track[:, 1]
    
    min_lat, max_lat = float(np.min(track_lats)), float(np.max(track_lats))
    min_lon, max_lon = float(np.min(track_lons)), float(np.max(track_lons))
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    # Initialize Folium Map with OpenStreetMap default
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None
    )
    
    # Layer 0: 100% Free Base Maps with ZERO API Keys and ZERO Watermarks
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="Standard Streets (OpenStreetMap)",
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom",
        name="Esri World Street Map (Clean Vector)",
        show=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Satellite Imagery (Esri)",
        show=False,
        control=True
    ).add_to(m)
    
    # Layer 1: Road Network Arcs (Grey)
    net_group = folium.FeatureGroup(name="1. Road Network Graph (Grey)", show=True)
    pad = 0.015
    net_lines = []
    for a in range(len(arcs)):
        u, v = arcs[a]
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        if ((min_lat - pad <= u_lat <= max_lat + pad and min_lon - pad <= u_lon <= max_lon + pad) or
            (min_lat - pad <= v_lat <= max_lat + pad and min_lon - pad <= v_lon <= max_lon + pad)):
            net_lines.append([[float(u_lat), float(u_lon)], [float(v_lat), float(v_lon)]])
            if len(net_lines) >= 8000:
                break
    if net_lines:
        folium.PolyLine(
            locations=net_lines,
            color="#64748b",
            weight=2.0,
            opacity=0.6,
            tooltip="Road Network Segment"
        ).add_to(net_group)
    net_group.add_to(m)
    
    # Layer 2: Raw Noisy GPS Telemetry (Cyan)
    gps_group = folium.FeatureGroup(name="2. Noisy GPS Observations (Cyan)", show=True)
    # Connected cyan trace
    trace_coords = [[float(track[i, 1]), float(track[i, 0])] for i in range(len(track))]
    folium.PolyLine(
        locations=trace_coords,
        color="#06b6d4",
        weight=2.0,
        opacity=0.6,
        dash_array="2, 4"
    ).add_to(gps_group)
    
    # Clickable GPS sample points
    step = max(1, len(track) // 180)
    for i in range(0, len(track), step):
        lon_i, lat_i, ts = track[i]
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 11px;">
            <b>GPS Ping #{i}</b><br/>
            Time: {ts:.1f}s<br/>
            Coord: {lat_i:.6f}, {lon_i:.6f}
        </div>
        """
        folium.CircleMarker(
            location=[float(lat_i), float(lon_i)],
            radius=3.5,
            color="#0891b2",
            fill=True,
            fill_color="#22d3ee",
            fill_opacity=0.85,
            weight=1.0,
            popup=folium.Popup(popup_html, max_width=180)
        ).add_to(gps_group)
    gps_group.add_to(m)
    
    # Layer 3: Ground Truth Route (Amber Gold)
    # Layer 3: Reference Ground Truth Route (Comparison)
    if include_ground_truth and len(route) > 0:
        gt_group = folium.FeatureGroup(name="3. Reference Ground Truth (Comparison)", show=show_ground_truth)
        gt_lines = []
        for r_arc in route:
            u, v = arcs[r_arc]
            gt_lines.append([[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]])
        if gt_lines:
            folium.PolyLine(
                locations=gt_lines,
                color="#f59e0b",
                weight=5.5,
                opacity=0.80,
                tooltip="Reference Ground Truth Traversed Route"
            ).add_to(gt_group)
        gt_group.add_to(m)
        
    # Layer 4: Classifier-Only Predictions (Red Dashed)
    if pred_raw_arcs is not None and len(pred_raw_arcs) > 0:
        raw_group = folium.FeatureGroup(name="4. Classifier-Only Baseline (Red Dashed)", show=False)
        raw_lines = [[[float(nodes[arcs[arc_id, 0], 1]), float(nodes[arcs[arc_id, 0], 0])],
                      [float(nodes[arcs[arc_id, 1], 1]), float(nodes[arcs[arc_id, 1], 0])]]
                     for arc_id in pred_raw_arcs]
        if raw_lines:
            folium.PolyLine(
                locations=raw_lines,
                color="#ef4444",
                weight=3.5,
                dash_array="6, 6",
                opacity=0.85,
                tooltip="Classifier-Only Route (Disjointed)"
            ).add_to(raw_group)
        raw_group.add_to(m)
        
    # Layer 5: Reconstructed Route (Current Request)
    if pred_hmm_arcs is not None and len(pred_hmm_arcs) > 0:
        hmm_group = folium.FeatureGroup(name="5. Reconstructed Route (Current Request)", show=True)
        hmm_lines = [[[float(nodes[arcs[arc_id, 0], 1]), float(nodes[arcs[arc_id, 0], 0])],
                      [float(nodes[arcs[arc_id, 1], 1]), float(nodes[arcs[arc_id, 1], 0])]]
                     for arc_id in pred_hmm_arcs]
        if hmm_lines:
            folium.PolyLine(
                locations=hmm_lines,
                color="#10b981",
                weight=5.5,
                opacity=0.95,
                tooltip="Current Request Reconstructed Route"
            ).add_to(hmm_group)
        hmm_group.add_to(m)
        
    # Start and Destination Pins
    start_lat, start_lon = float(track[0, 1]), float(track[0, 0])
    end_lat, end_lon = float(track[-1, 1]), float(track[-1, 0])
    
    folium.Marker(
        location=[start_lat, start_lon],
        popup="<b>Trip Origin</b>",
        tooltip="Trip Origin (Start)",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        location=[end_lat, end_lon],
        popup="<b>Trip Destination</b>",
        tooltip="Trip Destination (End)",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
    ).add_to(m)
    
    # Auto-fit bounds to make the entire trajectory visible with margin
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])
    
    # Plugins
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    
    # On-Map Floating HUD Legend
    legend_items = []
    if pred_hmm_arcs is not None and len(pred_hmm_arcs) > 0:
        legend_items.append("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 4px; background: #10b981; border-radius: 2px;"></span>
                <span style="font-weight: 600; color: #e2e8f0;">Reconstructed Route (Current Request)</span>
            </div>
        """)
    if include_ground_truth and len(route) > 0:
        legend_items.append("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 4px; background: #f59e0b; border-radius: 2px;"></span>
                <span style="color: #cbd5e1;">Reference Route (Comparison)</span>
            </div>
        """)
    if pred_raw_arcs is not None and len(pred_raw_arcs) > 0:
        legend_items.append("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 3px; border-top: 2px dashed #ef4444;"></span>
                <span style="color: #cbd5e1;">Classifier-Only Baseline</span>
            </div>
        """)
    legend_items.append(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; background: #06b6d4; border-radius: 50%;"></span>
            <span style="color: #94a3b8;">Noisy GPS Observations ({len(track):,} pts)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 2px; background: #64748b; border-radius: 1px;"></span>
            <span style="color: #94a3b8;">Road Network Corridors</span>
        </div>
    """)
    
    stats_box = ""
    if continuity_pct is not None or seq_f1_pct is not None:
        c_val = f"{continuity_pct:.1f}%" if continuity_pct is not None else "N/A"
        f_val = f"{seq_f1_pct:.1f}%" if seq_f1_pct is not None else "N/A"
        stats_box = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; padding: 6px 8px; background: rgba(30, 41, 59, 0.6); border-radius: 8px;">
            <div>
                <span style="color: #94a3b8; font-size: 10px;">Continuity</span><br/>
                <span style="color: #34d399; font-weight: 700; font-size: 13px;">{c_val}</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 10px;">Arc-set F1</span><br/>
                <span style="color: #38bdf8; font-weight: 700; font-size: 13px;">{f_val}</span>
            </div>
        </div>
        """
        
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 24px; 
        left: 24px; 
        z-index: 9999; 
        background: rgba(15, 23, 42, 0.88); 
        backdrop-filter: blur(10px);
        padding: 14px 18px; 
        border-radius: 14px; 
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        color: #f1f5f9; 
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 12px;
        line-height: 1.5;
        min-width: 250px;
    ">
        <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: #38bdf8; display: flex; justify-content: space-between; align-items: center;">
            <span>PathSnap Inspector</span>
            <span style="font-size: 10px; background: rgba(56, 189, 248, 0.2); padding: 2px 6px; border-radius: 4px; color: #7dd3fc;">Trip {trip_id}</span>
        </div>
        {stats_box}
        <div style="display: flex; flex-direction: column; gap: 4px;">
            {''.join(legend_items)}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(str(save_path))
    print(f"Saved enhanced interactive Folium map to {save_path}")
    return str(save_path)
