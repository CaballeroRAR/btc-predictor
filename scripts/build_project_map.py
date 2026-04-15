"""
build_project_map.py
Scans src/, scripts/, tests/, infra/, and docker/ and generates
.agent/memory/project_map.json with the full project module graph.

Usage:
    python scripts/build_project_map.py
        Full regeneration. Preserves non-stable last_known_state values
        from the existing map so in-progress Work Order states are not lost.

    python scripts/build_project_map.py --set-state <state> --paths <path1> <path2> ...
        Targeted update only. Sets last_known_state for specified paths.
        No file scanning. Valid states: stable, modified, untested, broken.
        Example: python scripts/build_project_map.py --set-state modified --paths src/core/standardizer.py

Run from the project root.
"""
import argparse
import ast
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, ".agent", "memory", "project_map.json")

# -------------------------------------------------------------------
# Scan roots: (directory, file extensions to include, skip __init__)
# -------------------------------------------------------------------
SCAN_ROOTS = [
    ("src",     [".py"],                     True),
    ("scripts", [".py", ".ps1"],             False),
    ("tests",   [".py", ".ps1"],             False),
    ("infra",   [".yaml", ".yml"],           False),
    ("docker",  [".Dockerfile", ""],         False),   # Dockerfiles have no extension
]

# -------------------------------------------------------------------
# Agent assignment rules (path prefix, first match wins)
# -------------------------------------------------------------------
AGENT_RULES = [
    # Tier 1: Direct model interaction
    ("src/core/engine/lstm_strategy.py", "ds_expert",       1),
    ("src/core/engine/base.py",          "ds_expert",       1),
    ("src/model.py",                     "ds_expert",       1),
    ("src/main_trainer.py",              "mlops_expert",    1),
    ("src/vertex_trigger.py",            "gcp_expert",      1),
    # Tier 2: Data transformation
    ("src/data_loader.py",               "ds_expert",       2),
    ("src/core/standardizer.py",         "ds_expert",       2),
    ("src/core/schemas.py",              "ds_expert",       2),
    ("src/core/data_orchestrator.py",    "ds_expert",       2),
    ("src/adapters/market_adapter.py",   "ds_expert",       2),
    # Tier 3: Inference pipeline
    ("src/facades/forecasting.py",              "ds_expert",  3),
    ("src/facades/lifecycle_facade.py",         "ds_expert",  3),
    ("src/core/analysis.py",                    "ds_expert",  3),
    ("src/core/simulation.py",                  "ds_expert",  3),
    ("src/core/simulation_orchestrator.py",     "ds_expert",  3),
    ("src/repositories/prediction_repo.py",     "ds_expert",  3),
    ("src/repositories/calibration_repo.py",    "ds_expert",  3),
    # Tier 4: Supporting state
    ("src/repositories/asset_repo.py",      "ds_expert",       4),
    ("src/repositories/firestore_repo.py",  "gcp_expert",      4),
    ("src/repositories/base.py",            "patterns_expert", 4),
    ("src/repositories/investment_repo.py", "ds_expert",       4),
    ("src/core/config_service.py",          "gcp_expert",      4),
    ("src/cloud_config.py",                 "gcp_expert",      None),
    # UI
    ("src/main_dashboard.py",            "ui_expert",       None),
    ("src/ui_blocks.py",                 "ui_expert",       None),
    ("src/facades/simulation_facade.py", "ui_expert",       None),
    # Worker
    ("src/main_worker.py",               "mlops_expert",    None),
    # Utils
    ("src/utils/logger.py",              "patterns_expert", None),
    ("src/utils/style_utils.py",         "ui_expert",       None),
    # Scripts
    ("scripts/deploy_all.ps1",           "gcp_expert",      None),
    ("scripts/deploy_ui.ps1",            "gcp_expert",      None),
    ("scripts/redeploy_trainer.ps1",     "mlops_expert",    None),
    ("scripts/scheduler_config.ps1",     "gcp_expert",      None),
    ("scripts/worker_init.ps1",          "gcp_expert",      None),
    ("scripts/build_project_map.py",     "orchestrator",    None),
    # Tests
    ("tests/test_production_readiness.py",          "logic_expert",   None),
    ("tests/Run-ValidationTests.ps1",               "logic_expert",   None),
    ("tests/scripts/deploy_all.Tests.ps1",          "gcp_expert",     None),
    ("tests/scripts/deploy_ui.Tests.ps1",           "gcp_expert",     None),
    ("tests/scripts/scheduler_config.Tests.ps1",    "gcp_expert",     None),
    ("tests/scripts/worker_init.Tests.ps1",         "gcp_expert",     None),
    ("tests/bin/",                                  "gcp_expert",     None),
    # Infra
    ("infra/dashboard.yaml",             "gcp_expert",      None),
    ("infra/train.yaml",                 "mlops_expert",    None),
    ("infra/worker.yaml",                "gcp_expert",      None),
    # Docker
    ("docker/dashboard.Dockerfile",      "gcp_expert",      None),
    ("docker/train.Dockerfile",          "mlops_expert",    None),
    ("docker/worker.Dockerfile",         "gcp_expert",      None),
]

PATTERN_MAP = {
    "src/repositories":              "Repository Pattern",
    "src/facades":                   "Facade Pattern",
    "src/adapters":                  "Adapter Pattern",
    "src/core/engine":               "Strategy Pattern",
    "src/core/schemas":              "Data Schema",
    "src/core/standardizer":         "Data Transform",
    "src/core/analysis":             "Analysis Pipeline",
    "src/core/simulation":           "Simulation Strategy",
    "src/utils":                     "Utility",
    "src/main_":                     "Entry Point",
    "src/model.py":                  "Model Definition",
    "src/data_loader":               "Data Pipeline",
    "src/cloud_config":              "Infrastructure Config",
    "src/vertex_trigger":            "MLOps Trigger",
    "scripts/":                      "Deployment Script",
    "tests/scripts/":                "Script Test",
    "tests/bin/":                    "Test Tooling",
    "tests/":                        "Test Suite",
    "infra/":                        "Cloud Build Config",
    "docker/":                       "Container Definition",
}

# Maps a test file path to the src file it exercises (best-effort)
TEST_TARGETS = {
    "tests/scripts/deploy_all.Tests.ps1":       "scripts/deploy_all.ps1",
    "tests/scripts/deploy_ui.Tests.ps1":        "scripts/deploy_ui.ps1",
    "tests/scripts/scheduler_config.Tests.ps1": "scripts/scheduler_config.ps1",
    "tests/scripts/worker_init.Tests.ps1":      "scripts/worker_init.ps1",
}

# Maps a Docker/infra file to its related src entry point
INFRA_TARGETS = {
    "docker/dashboard.Dockerfile": "src/main_dashboard.py",
    "docker/train.Dockerfile":     "src/main_trainer.py",
    "docker/worker.Dockerfile":    "src/main_worker.py",
    "infra/dashboard.yaml":        "docker/dashboard.Dockerfile",
    "infra/train.yaml":            "docker/train.Dockerfile",
    "infra/worker.yaml":           "docker/worker.Dockerfile",
}


def normalize(path: str) -> str:
    rel = os.path.relpath(path, ROOT)
    return rel.replace("\\", "/")


def get_agent_and_tier(norm: str):
    for prefix, agent, tier in AGENT_RULES:
        if norm == prefix or norm.startswith(prefix):
            return agent, tier
    return "patterns_expert", None


def get_pattern(norm: str) -> str:
    for prefix, pattern in PATTERN_MAP.items():
        if norm.startswith(prefix) or norm == prefix:
            return pattern
    return "General"


def get_project_imports(filepath: str, all_norm_paths: set) -> list:
    """Parse Python files for project-internal imports only."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module.startswith("src.") or
                node.module.startswith("core.") or
                node.module.startswith("adapters.") or
                node.module.startswith("facades.") or
                node.module.startswith("repositories.") or
                node.module.startswith("utils.")
            ):
                parts = node.module.replace(".", "/")
                for candidate in [f"src/{parts}.py", f"{parts}.py"]:
                    if candidate in all_norm_paths:
                        imports.append(candidate)
                        break
    return sorted(set(imports))


def collect_files() -> list:
    """Collect all project files across all scan roots."""
    collected = []
    for folder, extensions, skip_init in SCAN_ROOTS:
        folder_abs = os.path.join(ROOT, folder)
        if not os.path.isdir(folder_abs):
            continue
        for dirpath, dirnames, filenames in os.walk(folder_abs):
            # Skip pycache and venv
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "venv", ".git", "node_modules")]
            for fn in filenames:
                if skip_init and fn == "__init__.py":
                    continue
                # Match by extension; empty string in list means extensionless files
                _, ext = os.path.splitext(fn)
                # Special case: Dockerfile (no extension, name contains 'Dockerfile')
                is_dockerfile = "Dockerfile" in fn and ext == ""
                if ext in extensions or (is_dockerfile and "" in extensions):
                    collected.append(os.path.join(dirpath, fn))
    return collected


VALID_STATES = {"stable", "modified", "untested", "broken"}
NON_STABLE_STATES = {"modified", "untested", "broken"}


def load_existing_states() -> dict:
    """Load last_known_state and notes from existing map. Returns {} if map does not exist."""
    if not os.path.exists(OUT_PATH):
        return {}
    try:
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {
            k: {
                "last_known_state": v.get("last_known_state", "stable"),
                "notes": v.get("notes", ""),
            }
            for k, v in existing.get("modules", {}).items()
            if v.get("last_known_state") in NON_STABLE_STATES
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def targeted_set_state(paths: list, state: str) -> None:
    """Update last_known_state for specific paths without a full rescan."""
    if state not in VALID_STATES:
        print(f"ERROR: '{state}' is not a valid state. Choose from: {', '.join(sorted(VALID_STATES))}")
        raise SystemExit(1)

    if not os.path.exists(OUT_PATH):
        print("ERROR: project_map.json does not exist. Run without --set-state to generate it first.")
        raise SystemExit(1)

    with open(OUT_PATH, "r", encoding="utf-8") as f:
        project_map = json.load(f)

    updated = []
    not_found = []
    for path in paths:
        norm = path.replace("\\", "/")
        if norm in project_map["modules"]:
            project_map["modules"][norm]["last_known_state"] = state
            updated.append(norm)
        else:
            not_found.append(norm)

    project_map["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(project_map, f, indent=2)

    for u in updated:
        print(f"  SET {u} -> {state}")
    for nf in not_found:
        print(f"  NOT FOUND in map: {nf}")
    print(f"Targeted update complete. {len(updated)} path(s) updated.")


def main():
    parser = argparse.ArgumentParser(description="Build or update project_map.json")
    parser.add_argument(
        "--set-state",
        metavar="STATE",
        help="Targeted update: set last_known_state for --paths. No scanning.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="File paths to update when using --set-state.",
    )
    args = parser.parse_args()

    # --- Targeted update mode ---
    if args.set_state:
        if not args.paths:
            print("ERROR: --set-state requires --paths.")
            raise SystemExit(1)
        targeted_set_state(args.paths, args.set_state)
        return

    # --- Full regeneration mode ---
    # Preserve non-stable states from the existing map before overwriting
    preserved_states = load_existing_states()

    all_files = collect_files()
    all_norm = {normalize(f) for f in all_files}

    modules = {}
    for filepath in sorted(all_files):
        norm = normalize(filepath)
        _, ext = os.path.splitext(filepath)
        agent, tier = get_agent_and_tier(norm)
        pattern = get_pattern(norm)
        stat = os.stat(filepath)

        project_imports = get_project_imports(filepath, all_norm) if ext == ".py" else []

        related_to = []
        if norm in TEST_TARGETS and TEST_TARGETS[norm] in all_norm:
            related_to.append(TEST_TARGETS[norm])
        if norm in INFRA_TARGETS and INFRA_TARGETS[norm] in all_norm:
            related_to.append(INFRA_TARGETS[norm])

        # Carry forward non-stable state if one was recorded; default to stable
        preserved = preserved_states.get(norm, {})
        modules[norm] = {
            "path": norm,
            "pattern": pattern,
            "responsible_agent": agent,
            "ml_tier": tier,
            "imports_from_project": project_imports,
            "related_to": related_to,
            "imported_by": [],
            "test_file": None,
            "has_tests": False,
            "last_known_state": preserved.get("last_known_state", "stable"),
            "size_bytes": stat.st_size,
            "notes": preserved.get("notes", ""),
        }

    for norm, entry in modules.items():
        for dep in entry["imports_from_project"]:
            if dep in modules and norm not in modules[dep]["imported_by"]:
                modules[dep]["imported_by"].append(norm)

    for test_norm, src_norm in TEST_TARGETS.items():
        if src_norm in modules:
            modules[src_norm]["test_file"] = test_norm
            modules[src_norm]["has_tests"] = True

    for norm in list(modules.keys()):
        if norm.startswith("tests/") and norm.endswith(".py"):
            basename = os.path.basename(norm)
            stem = basename.replace("test_", "").replace(".py", "")
            for candidate in all_norm:
                if candidate.startswith("src/") and candidate.endswith(f"{stem}.py"):
                    if candidate in modules:
                        modules[candidate]["test_file"] = norm
                        modules[candidate]["has_tests"] = True

    for entry in modules.values():
        entry["imported_by"] = sorted(entry["imported_by"])

    project_map = {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": ROOT.replace("\\", "/"),
        "total_modules": len(modules),
        "scan_roots": [s[0] for s in SCAN_ROOTS],
        "modules": modules,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(project_map, f, indent=2)

    counts = {}
    for norm in modules:
        root = norm.split("/")[0]
        counts[root] = counts.get(root, 0) + 1

    preserved_count = len(preserved_states)
    print(f"Project map written: {OUT_PATH} (schema v1.2)")
    print(f"Total modules indexed: {len(modules)}")
    if preserved_count:
        print(f"Non-stable states preserved: {preserved_count}")
    for root, count in sorted(counts.items()):
        print(f"  {root}/: {count} files")


if __name__ == "__main__":
    main()

