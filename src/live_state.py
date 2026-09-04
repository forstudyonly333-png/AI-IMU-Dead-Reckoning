import json
import os
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
STATE_FILE = os.path.join(RESULTS_DIR, "live_state.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

def save_live_state(state_dict):
    """
    Atomically writes state to prevent partial reads by the Streamlit dashboard.
    """
    try:
        # Write to temporary file in the same directory first
        with tempfile.NamedTemporaryFile("w", dir=RESULTS_DIR, delete=False) as tf:
            json.dump(state_dict, tf)
            temp_name = tf.name
        # Atomic replace
        os.replace(temp_name, STATE_FILE)
    except Exception as e:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def load_live_state():
    """Reads live navigation state."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None