"""
UrbanSAR - JSON Fallback Utility

Save and load prediction results as JSON for demo safety.
If the live model fails during presentation, the dashboard
seamlessly loads these cached results instead.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FALLBACK_DIR


def save_fallback(data: Any, phase_name: str, directory: Path = FALLBACK_DIR) -> Path:
    """
    Save prediction data as JSON fallback.

    Called at each phase boundary to ensure demo-safe outputs.

    Args:
        data: Data to save (must be JSON-serializable)
        phase_name: Phase identifier (e.g., 'phase1_foundation', 'phase2_core_ai')
        directory: Fallback data directory

    Returns:
        Path to saved file
    """
    directory.mkdir(parents=True, exist_ok=True)

    fallback = {
        "phase": phase_name,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }

    filepath = directory / f"{phase_name}.json"
    with open(filepath, "w") as f:
        json.dump(fallback, f, indent=2, default=str)

    print(f"[INFO] Fallback saved: {filepath}")
    return filepath


def load_fallback(phase_name: str, directory: Path = FALLBACK_DIR) -> Optional[Any]:
    """
    Load cached fallback data.

    Args:
        phase_name: Phase identifier
        directory: Fallback data directory

    Returns:
        The cached data, or None if not found
    """
    filepath = directory / f"{phase_name}.json"

    if not filepath.exists():
        print(f"[WARN] No fallback found: {filepath}")
        return None

    with open(filepath, "r") as f:
        fallback = json.load(f)

    print(f"[INFO] Loaded fallback from {fallback.get('timestamp', 'unknown')}")
    return fallback.get("data")


def get_latest_fallback(directory: Path = FALLBACK_DIR) -> Optional[Any]:
    """
    Load the most recent fallback file.

    Returns:
        The most recent cached data, or None if no fallbacks exist
    """
    json_files = sorted(directory.glob("*.json"), key=lambda f: f.stat().st_mtime)

    if not json_files:
        return None

    latest = json_files[-1]
    with open(latest, "r") as f:
        fallback = json.load(f)

    print(f"[INFO] Loaded latest fallback: {latest.name}")
    return fallback.get("data")


def list_fallbacks(directory: Path = FALLBACK_DIR) -> list:
    """List all available fallback files with metadata."""
    results = []
    for f in sorted(directory.glob("*.json")):
        with open(f, "r") as fp:
            data = json.load(fp)
        results.append({
            "file": f.name,
            "phase": data.get("phase", "unknown"),
            "timestamp": data.get("timestamp", "unknown"),
        })
    return results
