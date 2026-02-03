import os
import pickle
import warnings


PLANNERS_DIR = os.path.dirname(__file__)


def load_saved_action_manager(filename):
    """Return a saved action manager if it exists; otherwise None."""
    path = os.path.join(PLANNERS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        warnings.warn(f"Failed to load action manager at {path}: {exc}")
        return None
