import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from src.metrics import compute_classification_metrics, compute_observation_top1_accuracy

from src.config import OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED, TEST_DIR
from src.data_loader import load_trip
from src.models import get_model_instances, train_and_eval_single_model
from src.metrics import compute_route_continuity, compute_sequence_performance
from src.visualizer import (
    plot_confusion_matrices,
    plot_timing_comparison,
    plot_roc_pr_curves,
    plot_metrics_comparison_across_models
)

def run_5_model_experimentation() -> Dict[str, Any]:
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 70, flush=True)
    print("PHASE 2 & 3: 5-MODEL BENCHMARKING & RIGOROUS METRIC EVALUATION", flush=True)
    print("=" * 70, flush=True)
    
    # Load processed datasets
    train_path = OUTPUT_DIR / "train_dataset.joblib"
    test_path = OUTPUT_DIR / "test_dataset.joblib"
    
    print(f"Loading train dataset from {train_path}...", flush=True)
    train_data = joblib.load(train_path)
    print(f"Loading test dataset from {test_path}...", flush=True)
    test_data = joblib.load(test_path)
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]
    
    X_test = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    test_obs_ids = test_data["obs_ids"]
    test_cand_arcs = test_data["cand_arcs"]
    
    print(f"Train set: {X_train.shape[0]:,} rows across {len(np.unique(groups_train))} trips.", flush=True)
    print(f"Test set:  {X_test.shape[0]:,} rows across {len(np.unique(groups_test))} trips.", flush=True)
    
    # Fit StandardScaler STRICTLY on training data only (Leakage prevention)
    print("Fitting StandardScaler strictly on training set...", flush=True)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, OUTPUT_DIR / "models" / "feature_scaler.joblib")
    
    models = get_model_instances()
    results = {}
    cv_records = []
    
    # Load raw trip data for test trips to compute route continuity and sequence metrics
    unique_test_trips = sorted(list(np.unique(groups_test)))
    test_trip_objects = {tid: load_trip(TEST_DIR, tid) for tid in unique_test_trips}
    
    for name, model in models.items():
        print(f"\nEvaluating Model: {name}...", flush=True)
        
        # Every validation fold consists of whole trips. Scaling is fit within each fold.
        for fold, (tr, val) in enumerate(GroupKFold(n_splits=3).split(X_train, y_train, groups_train), 1):
            fold_scaler = StandardScaler().fit(X_train[tr])
            fold_result = train_and_eval_single_model(
                name, clone(model), X_train[tr], y_train[tr], X_train[val], y_train[val],
                train_data["obs_ids"][val], train_data["cand_arcs"][val],
                scaler=fold_scaler, fit_subsample_limit=15000 if "Neighbors" in name else None,
                groups_test=groups_train[val])
            cv_records.append(dict(model=name, fold=fold, train_trips=len(np.unique(groups_train[tr])),
                validation_trips=len(np.unique(groups_train[val])),
                train_trip_ids=";".join(np.unique(groups_train[tr])),
                validation_trip_ids=";".join(np.unique(groups_train[val])), **fold_result["metrics"]))
            pd.DataFrame(cv_records).to_csv(OUTPUT_DIR / "grouped_cv_folds.csv", index=False)
            print(f"  Grouped CV fold {fold}: macro F1={fold_result['metrics']['macro_f1']:.4f}", flush=True)

        # Subsample limit for KNN fitting to avoid multi-hour kd-tree overhead on 1M rows
        fit_subsample = 15000 if "Neighbors" in name else None
        
        eval_res = train_and_eval_single_model(
            model_name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_obs_ids=test_obs_ids,
            test_cand_arcs=test_cand_arcs,
            scaler=scaler,
            fit_subsample_limit=fit_subsample,
            groups_test=groups_test
        )
        
        # Now compute trip-level Route Continuity and arc-set F1 on test trips
        y_prob = eval_res["y_prob"]
        df_test = pd.DataFrame({
            "trip_id": groups_test,
            "obs_id": test_obs_ids,
            "cand_arc": test_cand_arcs,
            "prob": y_prob
        })
        
        continuities = []
        seq_precisions = []
        seq_recalls = []
        seq_f1s = []
        
        for tid in unique_test_trips:
            t_df = df_test[df_test["trip_id"] == tid]
            t_obj = test_trip_objects[tid]
            
            # For each obs_id, pick arc with max prob
            idx_max = t_df.groupby("obs_id")["prob"].idxmax()
            best_cands = t_df.loc[idx_max].sort_values(by="obs_id")
            pred_arcs = list(best_cands["cand_arc"])
            
            # Continuity
            cont = compute_route_continuity(pred_arcs, t_obj["arcs"])
            continuities.append(cont)
            
            # Sequence metrics
            seq_m = compute_sequence_performance(pred_arcs, t_obj["route"])
            seq_precisions.append(seq_m["seq_precision"])
            seq_recalls.append(seq_m["seq_recall"])
            seq_f1s.append(seq_m["seq_f1"])
            
        mean_cont = float(np.mean(continuities))
        mean_seq_prec = float(np.mean(seq_precisions))
        mean_seq_rec = float(np.mean(seq_recalls))
        mean_seq_f1 = float(np.mean(seq_f1s))
        
        eval_res["metrics"]["route_continuity"] = mean_cont
        eval_res["metrics"]["seq_precision"] = mean_seq_prec
        eval_res["metrics"]["seq_recall"] = mean_seq_rec
        eval_res["metrics"]["seq_f1"] = mean_seq_f1
        eval_res["metrics"]["cand_miss_rate"] = float(test_data["trip_summaries"]["cand_miss_rate_pct"].mean() / 100.0)
        
        cv_frame = pd.DataFrame([r for r in cv_records if r["model"] == name])
        eval_res["metrics"]["cv_macro_f1_mean"] = float(cv_frame.macro_f1.mean())
        eval_res["metrics"]["cv_macro_f1_std"] = float(cv_frame.macro_f1.std(ddof=0))
        eval_res["metrics"]["cv_top1_mean"] = float(cv_frame.top1_accuracy.mean())
        results[name] = eval_res
        
        print(f"  Train Time:       {eval_res['train_time_sec']:.2f} s")
        print(f"  Inference Speed:  {eval_res['infer_speed_samples_sec']:,.0f} samples/s")
        print(f"  Obs Top-1 Acc:    {eval_res['metrics']['top1_accuracy']*100:.2f} %")
        print(f"  Candidate F1:     {eval_res['metrics']['f1']*100:.2f} %")
        print(f"  Log Loss:         {eval_res['metrics']['log_loss']:.4f}")
        print(f"  ROC-AUC:          {eval_res['metrics']['roc_auc']:.4f}")
        print(f"  Route Continuity: {mean_cont*100:.2f} %")
        print(f"  Arc-set F1:      {mean_seq_f1*100:.2f} %")
        
    # Generate all required visualization graphs
    print("\nGenerating all required comparative graphs...")
    plot_confusion_matrices(results, y_test)
    plot_timing_comparison(results)
    plot_roc_pr_curves(results, y_test)
    plot_metrics_comparison_across_models(results)
    
    # Save results table
    summary_rows = []
    for name, res in results.items():
        m = res["metrics"]
        summary_rows.append({
            "model": name,
            "family": {
                "Logistic Regression": "Linear / Probabilistic",
                "k-Nearest Neighbors": "Instance-based",
                "Random Forest": "Tree-based (Bagging)",
                "LightGBM": "Gradient Boosting",
                "Neural Network (MLP)": "Neural Network"
            }.get(name, "Other"),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "macro_f1": m["macro_f1"],
            "cv_macro_f1_mean": m["cv_macro_f1_mean"],
            "cv_macro_f1_std": m["cv_macro_f1_std"],
            "cv_top1_mean": m["cv_top1_mean"],
            "log_loss": m["log_loss"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "top1_accuracy": m["top1_accuracy"],
            "cand_miss_rate": m["cand_miss_rate"],
            "route_continuity": m["route_continuity"],
            "seq_f1": m["seq_f1"],
            "train_time_sec": res["train_time_sec"],
            "infer_speed_samples_sec": res["infer_speed_samples_sec"]
        })
        
    df_results = pd.DataFrame(summary_rows)
    df_results.to_csv(OUTPUT_DIR / "model_experimentation_results.csv", index=False)
    
    # Convert metrics for JSON export
    json_results = {
        name: {
            "metrics": res["metrics"],
            "train_time_sec": res["train_time_sec"],
            "infer_time_sec": res["infer_time_sec"],
            "infer_speed_samples_sec": res["infer_speed_samples_sec"]
        } for name, res in results.items()
    }
    with open(OUTPUT_DIR / "model_experimentation_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
        
    print("\n" + "=" * 70)
    print("FINAL 5-MODEL BENCHMARK TABLE:")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    return results

if __name__ == "__main__":
    run_5_model_experimentation()
