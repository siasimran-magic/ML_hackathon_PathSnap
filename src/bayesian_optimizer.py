import os
import sys
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from src.config import OUTPUT_DIR, MODELS_DIR, RANDOM_SEED, TEST_DIR
from src.data_loader import load_trip
from src.metrics import (
    compute_classification_metrics,
    compute_observation_top1_accuracy,
    compute_route_continuity,
    compute_sequence_performance
)

# Silence optuna logs to avoid clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials: int = 15) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 4: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA TPE)")
    print("=" * 70)
    
    train_data = joblib.load(OUTPUT_DIR / "train_dataset.joblib")
    test_data = joblib.load(OUTPUT_DIR / "test_dataset.joblib")
    scaler = joblib.load(OUTPUT_DIR / "models" / "feature_scaler.joblib")
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]
    obs_ids_train = train_data["obs_ids"]
    cand_arcs_train = train_data["cand_arcs"]
    
    X_test = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    obs_ids_test = test_data["obs_ids"]
    cand_arcs_test = test_data["cand_arcs"]
    
    # Subsample training data for rapid cross-validation search if needed
    # (e.g. 200k rows across diverse trips preserves group structure while speeding up trials 5x)
    unique_trips = np.unique(groups_train)
    np.random.seed(RANDOM_SEED)
    cv_trips = np.random.choice(unique_trips, size=min(40, len(unique_trips)), replace=False)
    cv_mask = np.isin(groups_train, cv_trips)
    
    X_cv = X_train[cv_mask]
    y_cv = y_train[cv_mask]
    groups_cv = groups_train[cv_mask]
    obs_ids_cv = obs_ids_train[cv_mask]
    cand_arcs_cv = cand_arcs_train[cv_mask]
    
    gkf = GroupKFold(n_splits=3)
    
    print(f"Running Optuna Bayesian Optimization over {len(cv_trips)} training trips ({len(X_cv):,} candidate pairs)...")
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": RANDOM_SEED,
            "n_jobs": 2,
            "verbosity": -1
        }
        
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_cv, y_cv, groups=groups_cv):
            X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
            X_val, y_val = X_cv[val_idx], y_cv[val_idx]
            obs_val = obs_ids_cv[val_idx]
            cands_val = cand_arcs_cv[val_idx]
            groups_val = groups_cv[val_idx]
            
            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            
            y_prob_val = clf.predict_proba(X_val)[:, 1]
            top1 = compute_observation_top1_accuracy(
                obs_val, cands_val, y_val, y_prob_val, trip_ids=groups_val
            )["top1_accuracy"]
            fold_scores.append(top1)
            trial.set_user_attr("fold_scores", fold_scores.copy())
            
        return float(np.mean(fold_scores))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage="sqlite:///" + (OUTPUT_DIR / "bayesian_study.sqlite3").as_posix(),
        study_name="pathsnap_grouped_tpe", load_if_exists=True)
    def save_history(study, trial):
        study.trials_dataframe().to_csv(OUTPUT_DIR / "bayesian_search_history.csv", index=False)
        history = [dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                        fold_scores=t.user_attrs.get("fold_scores", []),
                        started=t.datetime_start.isoformat() if t.datetime_start else None,
                        finished=t.datetime_complete.isoformat() if t.datetime_complete else None)
                   for t in study.trials]
        (OUTPUT_DIR / "bayesian_search_history.json").write_text(json.dumps(history, indent=2))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[save_history])
    
    print("\nBayesian Optimization Complete!")
    print(f"Best Trial Score (Mean Group CV Obs Top-1 Accuracy): {study.best_value * 100:.2f} %")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Refit tuned model on FULL 80 training trips
    print("\nRefitting tuned model on full 80 training trips...")
    best_params = {
        **study.best_params,
        "random_state": RANDOM_SEED,
        "n_jobs": 2,
        "verbosity": -1
    }
    tuned_model = LGBMClassifier(**best_params)
    t0 = time.time()
    tuned_model.fit(X_train, y_train)
    refit_time = time.time() - t0
    print(f"Refit finished in {refit_time:.2f}s.")
    
    # Evaluate on student-testing trips
    print("Evaluating tuned model on 10 student-testing trips...")
    t1 = time.time()
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t1
    infer_speed = len(X_test) / max(1e-5, infer_time)
    
    clf_m = compute_classification_metrics(y_test, y_prob_tuned)
    top1_m = compute_observation_top1_accuracy(
        obs_ids_test, cand_arcs_test, y_test, y_prob_tuned, trip_ids=groups_test
    )
    
    # Compute route continuity and arc-set F1
    unique_test_trips = sorted(list(np.unique(groups_test)))
    test_trip_objects = {tid: load_trip(TEST_DIR, tid) for tid in unique_test_trips}
    
    df_test = pd.DataFrame({
        "trip_id": groups_test,
        "obs_id": obs_ids_test,
        "cand_arc": cand_arcs_test,
        "prob": y_prob_tuned
    })
    
    conts, seq_f1s = [], []
    for tid in unique_test_trips:
        t_df = df_test[df_test["trip_id"] == tid]
        t_obj = test_trip_objects[tid]
        idx_max = t_df.groupby("obs_id")["prob"].idxmax()
        pred_arcs = list(t_df.loc[idx_max].sort_values(by="obs_id")["cand_arc"])
        conts.append(compute_route_continuity(pred_arcs, t_obj["arcs"]))
        seq_f1s.append(compute_sequence_performance(pred_arcs, t_obj["route"])["seq_f1"])
        
    mean_cont = float(np.mean(conts))
    mean_seq_f1 = float(np.mean(seq_f1s))
    
    tuned_metrics = {
        **clf_m,
        **top1_m,
        "route_continuity": mean_cont,
        "seq_f1": mean_seq_f1,
        "refit_time_sec": float(refit_time),
        "infer_speed_samples_sec": float(infer_speed)
    }
    
    # Save best tuned model
    best_model_path = MODELS_DIR / "best_tuned_lgbm_model.joblib"
    joblib.dump({
        "model": tuned_model,
        "best_params": best_params,
        "metrics": tuned_metrics,
        "study_trials": len(study.trials),
        "runtime_versions": {"sklearn": __import__("sklearn").__version__, "lightgbm": __import__("lightgbm").__version__}
    }, best_model_path)
    print(f"Saved best tuned model artifact to {best_model_path}")
    
    # Save comparison with untuned baseline
    baseline_results = joblib.load(OUTPUT_DIR / "train_dataset.joblib") # check results json
    results_json_path = OUTPUT_DIR / "model_experimentation_results.json"
    if results_json_path.exists():
        with open(results_json_path) as f:
            all_res = json.load(f)
        untuned_m = all_res.get("LightGBM", {}).get("metrics", {})
    else:
        untuned_m = {}
        
    comp_df = pd.DataFrame([
        {
            "version": "Untuned LightGBM (Default)",
            "accuracy": untuned_m.get("accuracy", 0.0),
            "precision": untuned_m.get("precision", 0.0),
            "recall": untuned_m.get("recall", 0.0),
            "f1": untuned_m.get("f1", 0.0),
            "macro_f1": untuned_m.get("macro_f1", 0.0),
            "top1_accuracy": untuned_m.get("top1_accuracy", 0.0),
            "log_loss": untuned_m.get("log_loss", 0.0),
            "roc_auc": untuned_m.get("roc_auc", 0.0),
            "route_continuity": untuned_m.get("route_continuity", 0.0),
            "seq_f1": untuned_m.get("seq_f1", 0.0)
        },
        {
            "version": "Bayesian-Tuned LightGBM (Optuna)",
            "accuracy": tuned_metrics["accuracy"],
            "precision": tuned_metrics["precision"],
            "recall": tuned_metrics["recall"],
            "f1": tuned_metrics["f1"],
            "macro_f1": tuned_metrics["macro_f1"],
            "top1_accuracy": tuned_metrics["top1_accuracy"],
            "log_loss": tuned_metrics["log_loss"],
            "roc_auc": tuned_metrics["roc_auc"],
            "route_continuity": tuned_metrics["route_continuity"],
            "seq_f1": tuned_metrics["seq_f1"]
        }
    ])
    
    comp_save_path = OUTPUT_DIR / "bayesian_tuning_comparison.csv"
    comp_df.to_csv(comp_save_path, index=False)
    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION COMPARISON:")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    
    return {
        "best_params": best_params,
        "tuned_metrics": tuned_metrics,
        "comp_df": comp_df
    }

if __name__ == "__main__":
    run_bayesian_optimization()
