"""
Flow registry: tracks each execution of the training flow.

Each flow run records:
  - run_id     : a short content hash of the config (idempotent on re-run)
  - config     : the full flow configuration (cutoff, features, model_type...)
  - model_hash : id of the model the run produced (None if the run failed)
  - status     : "passed" or "failed"
  - timestamp  : when it ran

This serves two purposes for Task 3:

  1. Flow versioning - re-running with a different config produces a
     different run_id, and the config is preserved so we can reproduce
     the same training later.

  2. A/B test resolution - the A/B flow takes two run_ids as inputs and
     resolves each to its actual model_hash via this registry. That way
     the A/B flow refers to "flow versions", not specific model files.

Records live in flow_runs/<run_id>.json with an index at flow_runs/manifest.json.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RUNS_DIR = Path("flow_runs")
RUNS_MANIFEST = RUNS_DIR / "manifest.json"


def register_run(
    *,
    config: dict,
    model_hash: Optional[str],
    status: str,
    error: Optional[str] = None,
) -> str:
    """
    Persist a flow run. Returns the run id.

    Run id is a hash of the config, so re-running the same config gives
    the same run id (idempotent). Different configs always get different
    ids. To force a distinct id for an otherwise identical config, change
    run_name.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _hash_config(config)

    record = {
        "run_id":     run_id,
        "config":     config,
        "model_hash": model_hash,
        "status":     status,
        "error":      error,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))

    manifest = _load_manifest()
    manifest["runs"][run_id] = {
        "model_hash": model_hash,
        "status":     status,
        "config":     config,
        "timestamp":  record["timestamp"],
    }
    _save_manifest(manifest)
    return run_id


def get_run(run_id: str) -> dict:
    """Return the full record for a flow run."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise KeyError(f"No flow run with id '{run_id}'.")
    return json.loads(path.read_text())


def resolve_model_hash(run_id: str) -> str:
    """Return the model id produced by a flow run."""
    run = get_run(run_id)
    if not run.get("model_hash"):
        raise ValueError(
            f"Flow run '{run_id}' did not produce a model "
            f"(status: {run.get('status')})."
        )
    return run["model_hash"]


def list_runs() -> list:
    """Return all registered flow runs."""
    manifest = _load_manifest()
    return [{"run_id": rid, **meta} for rid, meta in manifest.get("runs", {}).items()]


# Internal helpers

def _hash_config(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _load_manifest() -> dict:
    if RUNS_MANIFEST.exists():
        return json.loads(RUNS_MANIFEST.read_text())
    return {"runs": {}}


def _save_manifest(manifest: dict) -> None:
    RUNS_MANIFEST.write_text(json.dumps(manifest, indent=2))


# Entry point: list runs for inspection

if __name__ == "__main__":
    import sys
    runs = list_runs()
    if not runs:
        print("No flow runs registered yet.")
        sys.exit(0)
    print(f"{'run_id':<14}  {'status':<8}  {'model_hash':<14}  run_name")
    print("-" * 70)
    for r in runs:
        cfg = r.get("config", {})
        print(f"{r['run_id']:<14}  {r['status']:<8}  "
              f"{(r.get('model_hash') or '-'):<14}  {cfg.get('run_name') or '-'}")
