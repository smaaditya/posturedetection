"""
IMU Spinal Posture Detection — Dual-Model Pipeline
====================================================
Trains two Random Forest classifiers simultaneously:
  • Model 1: Posture  → cervical lordosis, lumbar lordosis, normal,
                         scoliosis, thoracic kyphosis
  • Model 2: Activity → sitting, standing, walking

Usage:
  python posture_pipeline.py --mode train   --data IMU_Spinal_Balanced.csv
  python posture_pipeline.py --mode predict --input sample_input.json
  python posture_pipeline.py --mode predict --input sample_input.csv
  python posture_pipeline.py --mode evaluate --data IMU_Spinal_Balanced.csv
"""

import argparse
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'C1x',  'C1y',  'C1z',
    'C7x',  'C7y',  'C7z',
    'T5x',  'T5y',  'T5z',
    'T12x', 'T12y', 'T12z',
    'L5x',  'L5y',  'L5z',
    'cervical_angle', 'thoracic_angle', 'lumbar_angle', 'scoliosis_angle'
]

MODEL_DIR      = os.path.dirname(os.path.abspath(__file__))
POSTURE_MODEL  = os.path.join(MODEL_DIR, 'rf_posture.pkl')
ACTIVITY_MODEL = os.path.join(MODEL_DIR, 'rf_activity.pkl')
POSTURE_ENC    = os.path.join(MODEL_DIR, 'le_posture.pkl')
ACTIVITY_ENC   = os.path.join(MODEL_DIR, 'le_activity.pkl')
META_FILE      = os.path.join(MODEL_DIR, 'model_meta.json')

RF_PARAMS = dict(n_estimators=100, random_state=42, n_jobs=-1)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def banner(title: str):
    w = 65
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def load_models():
    """Load saved models and encoders. Raises if not found."""
    for path in [POSTURE_MODEL, ACTIVITY_MODEL, POSTURE_ENC, ACTIVITY_ENC]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run with --mode train first."
            )
    with open(POSTURE_MODEL,  'rb') as f: rf_pos = pickle.load(f)
    with open(ACTIVITY_MODEL, 'rb') as f: rf_act = pickle.load(f)
    with open(POSTURE_ENC,    'rb') as f: le_pos = pickle.load(f)
    with open(ACTIVITY_ENC,   'rb') as f: le_act = pickle.load(f)
    return rf_pos, rf_act, le_pos, le_act


def validate_features(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


# ─── Train ────────────────────────────────────────────────────────────────────

def train(data_path: str):
    banner("TRAINING — Dual-Model Pipeline")

    print(f"\n  Loading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    validate_features(df)
    X = df[FEATURE_COLS].values

    # ── Posture model ──────────────────────────────────────────────────────
    print("\n  [1/2] Training POSTURE classifier …")
    le_pos = LabelEncoder()
    y_pos  = le_pos.fit_transform(df['posture'])
    rf_pos = RandomForestClassifier(**RF_PARAMS)
    rf_pos.fit(X, y_pos)
    print(f"        Classes : {list(le_pos.classes_)}")
    print(f"        Train acc: {rf_pos.score(X, y_pos)*100:.2f}%")

    # ── Activity model ─────────────────────────────────────────────────────
    print("\n  [2/2] Training ACTIVITY classifier …")
    le_act = LabelEncoder()
    y_act  = le_act.fit_transform(df['activity'])
    rf_act = RandomForestClassifier(**RF_PARAMS)
    rf_act.fit(X, y_act)
    print(f"        Classes : {list(le_act.classes_)}")
    print(f"        Train acc: {rf_act.score(X, y_act)*100:.2f}%")

    # ── Save ───────────────────────────────────────────────────────────────
    print("\n  Saving models …")
    with open(POSTURE_MODEL,  'wb') as f: pickle.dump(rf_pos, f)
    with open(ACTIVITY_MODEL, 'wb') as f: pickle.dump(rf_act, f)
    with open(POSTURE_ENC,    'wb') as f: pickle.dump(le_pos, f)
    with open(ACTIVITY_ENC,   'wb') as f: pickle.dump(le_act, f)

    # Feature importance summary
    fi_pos = pd.Series(rf_pos.feature_importances_, index=FEATURE_COLS)
    fi_act = pd.Series(rf_act.feature_importances_, index=FEATURE_COLS)

    meta = {
        'feature_cols':      FEATURE_COLS,
        'posture_classes':   list(le_pos.classes_),
        'activity_classes':  list(le_act.classes_),
        'posture_top5_features':  fi_pos.nlargest(5).to_dict(),
        'activity_top5_features': fi_act.nlargest(5).to_dict(),
    }
    with open(META_FILE, 'w') as f: json.dump(meta, f, indent=2)

    print(f"\n  ✓ rf_posture.pkl  saved")
    print(f"  ✓ rf_activity.pkl saved")
    print(f"  ✓ model_meta.json saved")

    print("\n  Top-5 features — POSTURE model:")
    for feat, val in fi_pos.nlargest(5).items():
        bar = "█" * int(val * 150)
        print(f"    {feat:<22} {val:.4f}  {bar}")

    print("\n  Top-5 features — ACTIVITY model:")
    for feat, val in fi_act.nlargest(5).items():
        bar = "█" * int(val * 150)
        print(f"    {feat:<22} {val:.4f}  {bar}")

    print("\n  ✓ Training complete.\n")


# ─── Predict ──────────────────────────────────────────────────────────────────

def predict(input_path: str):
    banner("INFERENCE — Dual-Model Prediction")

    rf_pos, rf_act, le_pos, le_act = load_models()

    # Load input — supports JSON (dict or list) and CSV
    if input_path.endswith('.json'):
        with open(input_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        df_in = pd.DataFrame(raw)
    elif input_path.endswith('.csv'):
        df_in = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be .json or .csv")

    validate_features(df_in)
    X_in = df_in[FEATURE_COLS].values

    pos_preds   = le_pos.inverse_transform(rf_pos.predict(X_in))
    pos_proba   = rf_pos.predict_proba(X_in)
    act_preds   = le_act.inverse_transform(rf_act.predict(X_in))
    act_proba   = rf_act.predict_proba(X_in)

    results = []
    print(f"\n  {'#':<5} {'POSTURE':<25} {'CONF':>6}   {'ACTIVITY':<12} {'CONF':>6}")
    print("  " + "─" * 60)

    for i in range(len(X_in)):
        pos_conf = pos_proba[i].max() * 100
        act_conf = act_proba[i].max() * 100

        # Per-class probabilities
        pos_dist = {cls: round(float(p)*100, 1)
                    for cls, p in zip(le_pos.classes_, pos_proba[i])}
        act_dist = {cls: round(float(p)*100, 1)
                    for cls, p in zip(le_act.classes_, act_proba[i])}

        print(f"  {i+1:<5} {pos_preds[i]:<25} {pos_conf:>5.1f}%"
              f"   {act_preds[i]:<12} {act_conf:>5.1f}%")

        results.append({
            'row':             i + 1,
            'posture':         pos_preds[i],
            'posture_confidence_pct': round(pos_conf, 2),
            'posture_probabilities':  pos_dist,
            'activity':        act_preds[i],
            'activity_confidence_pct': round(act_conf, 2),
            'activity_probabilities':  act_dist,
        })

    out_path = input_path.replace('.json', '_predictions.json') \
                         .replace('.csv',  '_predictions.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Predictions saved → {out_path}\n")
    return results


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(data_path: str):
    banner("EVALUATION — 5-Fold Cross-Validation + Leave-One-Person-Out")

    df = pd.read_csv(data_path)
    validate_features(df)

    # Use real rows only for honest evaluation
    real = df[df['timestamp'] != 'synthetic'].copy()
    print(f"\n  Dataset  : {data_path}")
    print(f"  All rows : {len(df):,}  |  Real rows only: {len(real):,}")

    X_real = real[FEATURE_COLS].values
    cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for target_col, label in [('posture', 'POSTURE'), ('activity', 'ACTIVITY')]:
        print(f"\n  {'─'*60}")
        print(f"  MODEL: {label}")
        print(f"  {'─'*60}")

        le = LabelEncoder()
        y  = le.fit_transform(real[target_col])
        rf = RandomForestClassifier(**RF_PARAMS)

        # 5-fold CV
        scores = cross_val_score(rf, X_real, y, cv=cv5, scoring='accuracy')
        print(f"\n  5-Fold CV Accuracy : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

        # Per-class report (fit once for classification_report)
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(rf, X_real, y, cv=cv5)
        print(f"\n  Per-class report:\n")
        print(classification_report(y, y_pred, target_names=le.classes_, digits=3))

        # Leave-One-Person-Out
        print(f"  Leave-One-Person-Out CV:")
        print(f"  {'Person':<10} {'Acc':>8}")
        print(f"  {'─'*20}")
        lopo_accs = []
        for pid in sorted(real['person_id'].unique()):
            tr = real[real['person_id'] != pid]
            te = real[real['person_id'] == pid]
            X_tr = tr[FEATURE_COLS].values;  y_tr = le.transform(tr[target_col])
            X_te = te[FEATURE_COLS].values;  y_te = le.transform(te[target_col])
            rf.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, rf.predict(X_te))
            lopo_accs.append(acc)
            print(f"  Person {pid:<4}  {acc*100:>7.2f}%")
        print(f"  {'─'*20}")
        print(f"  {'Mean':<10} {np.mean(lopo_accs)*100:>7.2f}%")

    print("\n  ✓ Evaluation complete.\n")


# ─── Predict Single Row (callable API) ────────────────────────────────────────

def predict_single(feature_dict: dict) -> dict:
    """
    Callable function for embedding in other scripts or APIs.

    Parameters
    ----------
    feature_dict : dict
        Keys must match FEATURE_COLS exactly. Values are floats.

    Returns
    -------
    dict with keys: posture, posture_confidence_pct,
                    posture_probabilities, activity,
                    activity_confidence_pct, activity_probabilities
    """
    rf_pos, rf_act, le_pos, le_act = load_models()

    row = np.array([[feature_dict[c] for c in FEATURE_COLS]])

    pos_proba = rf_pos.predict_proba(row)[0]
    act_proba = rf_act.predict_proba(row)[0]

    return {
        'posture':                  le_pos.classes_[np.argmax(pos_proba)],
        'posture_confidence_pct':   round(pos_proba.max() * 100, 2),
        'posture_probabilities':    {cls: round(float(p)*100, 1)
                                     for cls, p in zip(le_pos.classes_, pos_proba)},
        'activity':                 le_act.classes_[np.argmax(act_proba)],
        'activity_confidence_pct':  round(act_proba.max() * 100, 2),
        'activity_probabilities':   {cls: round(float(p)*100, 1)
                                     for cls, p in zip(le_act.classes_, act_proba)},
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IMU Spinal — Dual Posture + Activity Detection Pipeline'
    )
    parser.add_argument('--mode',  required=True,
                        choices=['train', 'predict', 'evaluate'],
                        help='Pipeline mode')
    parser.add_argument('--data',  help='Path to training/evaluation CSV')
    parser.add_argument('--input', help='Path to prediction input (.json or .csv)')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            parser.error("--data is required for train mode")
        train(args.data)

    elif args.mode == 'predict':
        if not args.input:
            parser.error("--input is required for predict mode")
        predict(args.input)

    elif args.mode == 'evaluate':
        if not args.data:
            parser.error("--data is required for evaluate mode")
        evaluate(args.data)
