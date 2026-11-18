import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, average_precision_score, confusion_matrix
)

def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes candidate-level classification metrics:
    Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC, PR-AUC.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Clip probabilities for stable log loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_pred_prob, eps, 1.0 - eps)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        ll = float(log_loss(y_true, y_prob_clipped))
    except Exception:
        ll = float("nan")
        
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        roc_auc = float("nan")
        
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        pr_auc = float("nan")
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "log_loss": ll,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def compute_observation_top1_accuracy(
    obs_ids: np.ndarray,
    cand_arcs: np.ndarray,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    trip_ids: np.ndarray = None
) -> Dict[str, float]:
    """
    Computes observation-level Top-1 accuracy across the entire partition.
    When trip_ids is provided, candidates are grouped by (trip_id, obs_id) so that
    observations from different trips with identical local indices are never conflated.
    """
    if trip_ids is not None:
        df = pd.DataFrame({
            "trip_id": trip_ids,
            "obs_id": obs_ids,
            "cand_arc": cand_arcs,
            "y_true": y_true,
            "prob": y_pred_prob
        })
        idx_max = df.groupby(["trip_id", "obs_id"])["prob"].idxmax()
    else:
        df = pd.DataFrame({
            "obs_id": obs_ids,
            "cand_arc": cand_arcs,
            "y_true": y_true,
            "prob": y_pred_prob
        })
        idx_max = df.groupby("obs_id")["prob"].idxmax()
        
    top_candidates = df.loc[idx_max]
    total = len(idx_max)
    top1_correct = int((top_candidates["y_true"] == 1).sum())
    top1_accuracy = float(top1_correct / max(1, total))
    
    return {
        "top1_accuracy": top1_accuracy,
        "top1_correct": top1_correct,
        "total_obs": total
    }

def compute_route_continuity(
    predicted_arcs_seq: List[int],
    arcs_array: np.ndarray
) -> float:
    """
    Measures the percentage of consecutive transitions that are continuous:
    Either arc_t == arc_{t+1} (remaining on same road segment), or
    end node of arc_t == start node of arc_{t+1} (valid topological edge connection).
    """
    if len(predicted_arcs_seq) < 2:
        return 1.0
        
    continuous_transitions = 0
    total_transitions = len(predicted_arcs_seq) - 1
    
    u_nodes = arcs_array[:, 0]
    v_nodes = arcs_array[:, 1]
    
    for t in range(total_transitions):
        a_curr = predicted_arcs_seq[t]
        a_next = predicted_arcs_seq[t + 1]
        
        if a_curr == a_next:
            continuous_transitions += 1
        elif v_nodes[a_curr] == u_nodes[a_next]:
            continuous_transitions += 1
            
    return float(continuous_transitions / max(1, total_transitions))

def compute_sequence_performance(
    predicted_arcs_seq: List[int],
    ground_truth_route: np.ndarray
) -> Dict[str, float]:
    """
    Computes precision, recall and F1 over unique arc sets.
    Ignores traversal order and repeated arcs. Legacy seq_* keys are retained.
    """
    # Compress consecutive duplicates to obtain traversed route sequence
    compressed_pred = []
    for arc in predicted_arcs_seq:
        if len(compressed_pred) == 0 or compressed_pred[-1] != arc:
            compressed_pred.append(arc)
            
    pred_set = set(compressed_pred)
    true_set = set(ground_truth_route)
    
    if len(pred_set) == 0 or len(true_set) == 0:
        return {"seq_precision": 0.0, "seq_recall": 0.0, "seq_f1": 0.0}
        
    intersection = len(pred_set.intersection(true_set))
    seq_prec = float(intersection / len(pred_set))
    seq_rec = float(intersection / len(true_set))
    seq_f1 = float(2 * seq_prec * seq_rec / (seq_prec + seq_rec)) if (seq_prec + seq_rec) > 0 else 0.0
    
    return {
        "seq_precision": seq_prec,
        "seq_recall": seq_rec,
        "seq_f1": seq_f1
    }
