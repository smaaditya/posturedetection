"""
IMU Spinal — Row Predictor
===========================
Enter any Excel row number (2–10227) and get the
posture + activity prediction for that row's sensor values.

Usage:
    python row_predict.py
    python row_predict.py --row 42
    python row_predict.py --row 42 --data IMU_Spinal_Balanced.csv
"""

import argparse
import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV  = os.path.join(SCRIPT_DIR, 'IMU_Spinal_Balanced.csv')
POSTURE_MODEL  = os.path.join(SCRIPT_DIR, 'rf_posture.pkl')
ACTIVITY_MODEL = os.path.join(SCRIPT_DIR, 'rf_activity.pkl')
POSTURE_ENC    = os.path.join(SCRIPT_DIR, 'le_posture.pkl')
ACTIVITY_ENC   = os.path.join(SCRIPT_DIR, 'le_activity.pkl')

FEATURE_COLS = [
    'C1x',  'C1y',  'C1z',
    'C7x',  'C7y',  'C7z',
    'T5x',  'T5y',  'T5z',
    'T12x', 'T12y', 'T12z',
    'L5x',  'L5y',  'L5z',
    'cervical_angle', 'thoracic_angle', 'lumbar_angle', 'scoliosis_angle'
]

# Posture → colour code (ANSI)
POSTURE_COLORS = {
    'normal':            '\033[92m',   # green
    'cervical lordosis': '\033[93m',   # yellow
    'lumbar lordosis':   '\033[94m',   # blue
    'thoracic kyphosis': '\033[91m',   # red
    'scoliosis':         '\033[95m',   # magenta
}
ACTIVITY_COLORS = {
    'sitting':  '\033[96m',   # cyan
    'standing': '\033[92m',   # green
    'walking':  '\033[93m',   # yellow
}
RESET  = '\033[0m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
RED    = '\033[91m'

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def bar(value, width=30, filled='█', empty='░'):
    n = round(value / 100 * width)
    return filled * n + empty * (width - n)

def conf_color(pct):
    if pct >= 90: return GREEN
    if pct >= 70: return YELLOW
    return RED

def load_models():
    missing = [p for p in [POSTURE_MODEL, ACTIVITY_MODEL, POSTURE_ENC, ACTIVITY_ENC]
               if not os.path.exists(p)]
    if missing:
        print(f"\n{RED}✗ Missing model files:{RESET}")
        for m in missing:
            print(f"    {m}")
        print(f"\n  Run training first:")
        print(f"  {DIM}python posture_pipeline.py --mode train --data IMU_Spinal_Balanced.csv{RESET}\n")
        sys.exit(1)

    with open(POSTURE_MODEL,  'rb') as f: rf_pos = pickle.load(f)
    with open(ACTIVITY_MODEL, 'rb') as f: rf_act = pickle.load(f)
    with open(POSTURE_ENC,    'rb') as f: le_pos = pickle.load(f)
    with open(ACTIVITY_ENC,   'rb') as f: le_act = pickle.load(f)
    return rf_pos, rf_act, le_pos, le_act

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"\n{RED}✗ Dataset not found:{RESET} {csv_path}")
        print(f"  Make sure IMU_Spinal_Balanced.csv is in the same folder.\n")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df

def excel_row_to_df_index(excel_row):
    """Excel row 2 = DataFrame index 0 (row 1 is the header)."""
    return excel_row - 2

def predict_row(df, df_index, rf_pos, rf_act, le_pos, le_act):
    row   = df.iloc[df_index]
    X     = row[FEATURE_COLS].values.reshape(1, -1)

    pos_proba = rf_pos.predict_proba(X)[0]
    act_proba = rf_act.predict_proba(X)[0]

    posture  = le_pos.classes_[np.argmax(pos_proba)]
    activity = le_act.classes_[np.argmax(act_proba)]

    pos_dist = {cls: round(float(p)*100, 1) for cls, p in zip(le_pos.classes_, pos_proba)}
    act_dist = {cls: round(float(p)*100, 1) for cls, p in zip(le_act.classes_, act_proba)}

    # Actual labels if present
    actual_posture  = row.get('posture',  'N/A')
    actual_activity = row.get('activity', 'N/A')

    return {
        'excel_row':          df_index + 2,
        'timestamp':          row.get('timestamp', 'N/A'),
        'person_id':          row.get('person_id', 'N/A'),
        'posture':            posture,
        'posture_conf':       round(float(pos_proba.max() * 100), 1),
        'posture_probs':      pos_dist,
        'activity':           activity,
        'activity_conf':      round(float(act_proba.max() * 100), 1),
        'activity_probs':     act_dist,
        'actual_posture':     actual_posture,
        'actual_activity':    actual_activity,
        'features':           {c: round(float(row[c]), 5) for c in FEATURE_COLS},
    }

def print_result(result):
    pc = POSTURE_COLORS.get(result['posture'], CYAN)
    ac = ACTIVITY_COLORS.get(result['activity'], CYAN)
    cc_p = conf_color(result['posture_conf'])
    cc_a = conf_color(result['activity_conf'])

    is_correct_pos = str(result['actual_posture']).lower() == result['posture'].lower()
    is_correct_act = str(result['actual_activity']).lower() == result['activity'].lower()

    print()
    print(f"  {'─'*58}")
    print(f"  {BOLD}Row {result['excel_row']}{RESET}  "
          f"{DIM}│ Timestamp: {result['timestamp']}  "
          f"│ Person: {result['person_id']}{RESET}")
    print(f"  {'─'*58}")

    # Posture result
    print(f"\n  {BOLD}POSTURE DETECTION{RESET}")
    print(f"  Predicted  : {pc}{BOLD}{result['posture'].upper()}{RESET}")
    print(f"  Confidence : {cc_p}{result['posture_conf']}%  {bar(result['posture_conf'], 25)}{RESET}")
    if str(result['actual_posture']) != 'N/A':
        tick = f"{GREEN}✓ correct{RESET}" if is_correct_pos else f"{RED}✗ actual: {result['actual_posture']}{RESET}"
        print(f"  Actual     : {tick}")
    print(f"\n  {DIM}All class probabilities:{RESET}")
    for cls, pct in sorted(result['posture_probs'].items(), key=lambda x: -x[1]):
        b = bar(pct, 20)
        col = POSTURE_COLORS.get(cls, DIM)
        marker = ' ◀' if cls == result['posture'] else ''
        print(f"  {col}{cls:<22}{RESET}  {pct:>5.1f}%  {DIM}{b}{RESET}{marker}")

    # Activity result
    print(f"\n  {BOLD}ACTIVITY DETECTION{RESET}")
    print(f"  Predicted  : {ac}{BOLD}{result['activity'].upper()}{RESET}")
    print(f"  Confidence : {cc_a}{result['activity_conf']}%  {bar(result['activity_conf'], 25)}{RESET}")
    if str(result['actual_activity']) != 'N/A':
        tick = f"{GREEN}✓ correct{RESET}" if is_correct_act else f"{RED}✗ actual: {result['actual_activity']}{RESET}"
        print(f"  Actual     : {tick}")
    print(f"\n  {DIM}All class probabilities:{RESET}")
    for cls, pct in sorted(result['activity_probs'].items(), key=lambda x: -x[1]):
        b = bar(pct, 20)
        col = ACTIVITY_COLORS.get(cls, DIM)
        marker = ' ◀' if cls == result['activity'] else ''
        print(f"  {col}{cls:<22}{RESET}  {pct:>5.1f}%  {DIM}{b}{RESET}{marker}")

    # Feature values
    print(f"\n  {DIM}{'─'*58}{RESET}")
    print(f"  {DIM}Sensor values used:{RESET}")
    feats = result['features']
    cols_to_show = ['cervical_angle','thoracic_angle','lumbar_angle','scoliosis_angle',
                    'C1x','C7x','T12x','L5x']
    for c in cols_to_show:
        print(f"  {DIM}{c:<22}  {feats[c]:>10.5f}{RESET}")
    print()

def print_header(total_rows):
    print()
    print(f"  {BOLD}{'═'*58}{RESET}")
    print(f"  {BOLD}  SpinalSense — Row Predictor{RESET}")
    print(f"  {DIM}  Dual-model: Posture + Activity detection{RESET}")
    print(f"  {DIM}  Dataset rows: Excel 2 → {total_rows + 1}  ({total_rows:,} total){RESET}")
    print(f"  {BOLD}{'═'*58}{RESET}")

# ─── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict posture & activity for a specific Excel row')
    parser.add_argument('--row',  type=int, help='Excel row number (2 to 10227)')
    parser.add_argument('--data', type=str, default=DEFAULT_CSV, help='Path to CSV dataset')
    args = parser.parse_args()

    print(f"\n  {DIM}Loading models...{RESET}", end='', flush=True)
    rf_pos, rf_act, le_pos, le_act = load_models()
    print(f"\r  {GREEN}✓ Models loaded   {RESET}")

    print(f"  {DIM}Loading dataset...{RESET}", end='', flush=True)
    df = load_data(args.data)
    total_rows = len(df)
    min_excel  = 2
    max_excel  = total_rows + 1
    print(f"\r  {GREEN}✓ Dataset loaded  {RESET}  {DIM}({total_rows:,} rows){RESET}")

    print_header(total_rows)

    # ── Single row mode (--row flag) ──────────────────────────────────────────
    if args.row:
        excel_row = args.row
        if not (min_excel <= excel_row <= max_excel):
            print(f"\n  {RED}✗ Row {excel_row} is out of range. Valid: {min_excel}–{max_excel}{RESET}\n")
            sys.exit(1)
        result = predict_row(df, excel_row_to_df_index(excel_row), rf_pos, rf_act, le_pos, le_act)
        print_result(result)
        return

    # ── Interactive loop ──────────────────────────────────────────────────────
    print(f"\n  {DIM}Type a row number and press Enter. Type 'q' to quit.{RESET}\n")

    while True:
        try:
            user_input = input(f"  {BOLD}Enter Excel row ({min_excel}–{max_excel}) › {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {DIM}Goodbye.{RESET}\n")
            break

        if user_input.lower() in ('q', 'quit', 'exit'):
            print(f"\n  {DIM}Goodbye.{RESET}\n")
            break

        if not user_input:
            continue

        if not user_input.isdigit():
            print(f"  {YELLOW}  Please enter a number.{RESET}\n")
            continue

        excel_row = int(user_input)

        if not (min_excel <= excel_row <= max_excel):
            print(f"  {RED}  Row {excel_row} out of range. Enter {min_excel}–{max_excel}.{RESET}\n")
            continue

        result = predict_row(df, excel_row_to_df_index(excel_row), rf_pos, rf_act, le_pos, le_act)
        print_result(result)


if __name__ == '__main__':
    main()
