import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from src.config import RANDOM_SEED
from src.metrics import (
    compute_classification_metrics,
    compute_observation_top1_accuracy
)

def get_model_instances() -> Dict[str, Any]:
    """
    Returns 5 distinct model instances from distinct algorithmic families:
    1. Linear / Probabilistic: Logistic Regression
    2. Instance-based: k-Nearest Neighbors
    3. Tree-based / Bagging: Random Forest
    4. Gradient Boosting: LightGBM
    5. Neural Network: Multi-Layer Perceptron (MLP)
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=300,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        ),
        "k-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=2
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=12,
            min_samples_split=10,
            random_state=RANDOM_SEED,
            n_jobs=2
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=2,
            verbosity=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            batch_size=2048,
            max_iter=40,
            early_stopping=False,
            random_state=RANDOM_SEED
        )
    }

def train_and_eval_single_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_obs_ids: np.ndarray,
    test_cand_arcs: np.ndarray,
    scaler: StandardScaler = None,
    fit_subsample_limit: int = None,
    groups_test: np.ndarray = None
) -> Dict[str, Any]:
    """
    Trains a single model and evaluates candidate classification,
    observation Top-1 accuracy, and measures training time and inference latency.
    """
    # Scaling if needed (KNN and MLP benefit heavily from scaled features)
    if scaler is not None:
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = X_train
        X_te = X_test
        
    # Subsample for very slow fitters (e.g. KNN instance-based on 1M rows)
    if fit_subsample_limit is not None and len(X_tr) > fit_subsample_limit:
        np.random.seed(RANDOM_SEED)
        sub_indices = np.random.choice(len(X_tr), fit_subsample_limit, replace=False)
        X_fit = X_tr[sub_indices]
        y_fit = y_train[sub_indices]
    else:
        X_fit = X_tr
        y_fit = y_train
        
    # Training time
    t0 = time.time()
    model.fit(X_fit, y_fit)
    train_time = time.time() - t0
    
    # Inference time
    t1 = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        decision = model.decision_function(X_te)
        y_prob = 1.0 / (1.0 + np.exp(-decision))
    infer_time = time.time() - t1
    infer_speed_samples_sec = len(X_te) / max(1e-5, infer_time)
    
    # Compute candidate classification metrics
    clf_metrics = compute_classification_metrics(y_test, y_prob)
    
    # Compute observation Top-1 accuracy with whole-partition trip grouping
    top1_metrics = compute_observation_top1_accuracy(
        test_obs_ids, test_cand_arcs, y_test, y_prob, trip_ids=groups_test
    )
    
    return {
        "model_name": model_name,
        "model": model,
        "train_time_sec": float(train_time),
        "infer_time_sec": float(infer_time),
        "infer_speed_samples_sec": float(infer_speed_samples_sec),
        "y_prob": y_prob,
        "metrics": {
            **clf_metrics,
            **top1_metrics
        }
    }
