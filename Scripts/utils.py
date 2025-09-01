import datetime

def log_message(msg: str):
    """Log messages with timestamps."""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def feature_list(v_cols):
    """Return final feature list including engineered features."""
    return v_cols + ["Amount", "Amount_abs", "Amount_log1p", "txn_magnitude"]
