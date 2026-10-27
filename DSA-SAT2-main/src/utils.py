"""
Utils module for CH15 Road Safety Insight.
Provides helper functions for logging, path management, and formatting.
"""

import os
import json
import numpy as np
import pandas as pd


def get_project_root():
    """Returns the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def save_json(data, filepath):
    """Saves dictionary data to a formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    print(f"[INFO] Saved JSON to {filepath}")


def load_json(filepath):
    """Loads dictionary data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(value, decimals=1):
    """Formats float as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"
