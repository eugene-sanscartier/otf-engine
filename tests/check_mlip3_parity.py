from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from otf_engine.io_cfg import read_cfg, write_cfg
from otf_engine.mtp_backend import calculate_grade, select_add


def _default_splits(count):
    train_counts = [1, 2, 3, 5, 8, 10, 12, 16, 20, 24, 28]
    splits = []
    for train_n in train_counts:
        if train_n >= count:
            break
        cand_n = min(8, count - train_n)
        if cand_n > 0:
            splits.append(f"{train_n}:{cand_n}")
    return splits


def _scrub(cfgs):
    cleaned = []
    for atoms in cfgs:
        cfg = atoms.copy()
        cfg.calc = atoms.calc
        if "nbh_grades" in cfg.arrays:
            del cfg.arrays["nbh_grades"]
        features = dict(cfg.info.get("features", {}))
        features.pop("MV_grade", None)
        features.pop("selected_eqn_inds", None)
        if features:
            cfg.info["features"] = features
        elif "features" in cfg.info:
            del cfg.info["features"]
        cleaned.append(cfg)
    return cleaned


def _fingerprint(atoms):
    return (
        tuple(np.asarray(atoms.numbers, dtype=int).tolist()),
        tuple(np.round(atoms.cell.array.reshape(-1), 12).tolist()),
        tuple(np.round(atoms.positions.reshape(-1), 12).tolist()),
    )


def _selected_indices(selected_cfgs, candidate_cfgs):
    by_fp = {_fingerprint(cfg): i for i, cfg in enumerate(candidate_cfgs)}
    return [by_fp[_fingerprint(cfg)] for cfg in selected_cfgs]


def _run_calculate_grade(mlp, potential, cfg_path, out_path, cwd):
    subprocess.run(
        [str(mlp), "calculate_grade", str(potential), str(cfg_path), str(out_path)],
        check=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_path.with_suffix(out_path.suffix + ".0")


def _run_select_add(mlp, potential, train_path, cand_path, out_path, cwd):
    subprocess.run(
        [str(mlp), "select_add", str(potential), str(train_path), str(cand_path), str(out_path)],
        check=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", default="/home/eugene/Doctorat/code_library/mlip-3/bin/mlp")
    parser.add_argument("--potential", default="examples/potential.almtp")
    parser.add_argument("--cfg", default="examples/configuration.cfg")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="train:candidate counts for select_add checks",
    )
    args = parser.parse_args()

    root = Path.cwd()
    mlp = Path(args.mlp)
    potential = (root / args.potential).resolve()
    cfg_path = (root / args.cfg).resolve()

    for label, path, flag in (("mlip-3 binary", mlp, "--mlp"), ("potential", potential, "--potential"), ("configurations", cfg_path, "--cfg")):
        if not path.exists():
            raise SystemExit(f"{label} not found at {path} — pass {flag}. These inputs are not in the repository: "
                             "the potential must carry a #MVS_v1.1 active set, and the configurations must be "
                             "extrapolative enough to cross the swap threshold. See AGENTS.md, Conventions.")

    with cfg_path.open() as f:
        all_cfgs = read_cfg(f)
    splits = args.splits if args.splits is not None else _default_splits(len(all_cfgs))

    results = {
        "match_criteria": {
            "calculate_grade": [
                "compare MV_grade per configuration by absolute difference",
                "compare nbh_grades per configuration by max absolute per-atom difference",
            ],
            "select_add": [
                "compare selected candidate identities by structure fingerprint",
                "compare selected candidate order",
                "report selected candidate indices relative to the candidate slice and to the full example set",
            ],
        }
    }
    with TemporaryDirectory(dir=root) as td:
        td = Path(td)

        all_clean = _scrub(all_cfgs)
        cand_in = td / "cand.cfg"
        with cand_in.open("w") as f:
            write_cfg(f, all_clean)

        graded_path = _run_calculate_grade(mlp, potential, cand_in, td / "graded.cfg", td)
        with graded_path.open() as f:
            mlp_graded = read_cfg(f)
        py_graded = calculate_grade(str(potential), _scrub(all_cfgs))

        mv_diffs = []
        nbh_diffs = []
        mv_values = []
        nbh_values = []
        for i, (mlp_cfg, py_cfg) in enumerate(zip(mlp_graded, py_graded, strict=True)):
            mlp_mv = float(mlp_cfg.info["features"]["MV_grade"])
            py_mv = float(py_cfg.info["features"]["MV_grade"])
            mlp_nbh = np.asarray(mlp_cfg.arrays["nbh_grades"])
            py_nbh = np.asarray(py_cfg.arrays["nbh_grades"])
            mv_diffs.append(abs(mlp_mv - py_mv))
            nbh_diffs.append(float(np.max(np.abs(mlp_nbh - py_nbh))))
            mv_values.append((i, mlp_mv, py_mv))
            nbh_values.append((i, float(mlp_nbh.max()), float(py_nbh.max())))

        worst_mv_i = int(np.argmax(mv_diffs))
        worst_nbh_i = int(np.argmax(nbh_diffs))

        results["calculate_grade"] = {
            "max_mv_diff": max(mv_diffs),
            "max_nbh_diff": max(nbh_diffs),
            "mean_mv_diff": float(np.mean(mv_diffs)),
            "mean_nbh_diff": float(np.mean(nbh_diffs)),
            "worst_mv_config": {
                "cfg_index": worst_mv_i,
                "mlp_mv_grade": mv_values[worst_mv_i][1],
                "py_mv_grade": mv_values[worst_mv_i][2],
                "abs_diff": mv_diffs[worst_mv_i],
            },
            "worst_nbh_config": {
                "cfg_index": worst_nbh_i,
                "mlp_max_nbh_grade": nbh_values[worst_nbh_i][1],
                "py_max_nbh_grade": nbh_values[worst_nbh_i][2],
                "max_abs_diff": nbh_diffs[worst_nbh_i],
            },
        }

        select_results = []
        for split in splits:
            train_n, cand_n = (int(x) for x in split.split(":", 1))
            train_cfgs = _scrub(all_cfgs[:train_n])
            cand_cfgs = _scrub(all_cfgs[train_n:train_n + cand_n])

            train_file = td / f"train_{train_n}_{cand_n}.cfg"
            cand_file = td / f"cand_{train_n}_{cand_n}.cfg"
            selected_file = td / f"selected_{train_n}_{cand_n}.cfg"
            with train_file.open("w") as f:
                write_cfg(f, train_cfgs)
            with cand_file.open("w") as f:
                write_cfg(f, cand_cfgs)

            mlp_selected_path = _run_select_add(mlp, potential, train_file, cand_file, selected_file, td)
            with mlp_selected_path.open() as f:
                mlp_selected = read_cfg(f)
            py_selected, _state = select_add(str(potential), _scrub(all_cfgs[:train_n]), _scrub(all_cfgs[train_n:train_n + cand_n]))

            mlp_fp = [_fingerprint(cfg) for cfg in mlp_selected]
            py_fp = [_fingerprint(cfg) for cfg in py_selected]
            mlp_local = _selected_indices(mlp_selected, cand_cfgs)
            py_local = _selected_indices(py_selected, cand_cfgs)
            select_results.append(
                {
                    "train_n": train_n,
                    "cand_n": cand_n,
                    "mlp_selected": len(mlp_selected),
                    "py_selected": len(py_selected),
                    "same_order": mlp_fp == py_fp,
                    "same_set": set(mlp_fp) == set(py_fp),
                    "mlp_candidate_local_indices": mlp_local,
                    "py_candidate_local_indices": py_local,
                    "mlp_candidate_global_indices": [train_n + i for i in mlp_local],
                    "py_candidate_global_indices": [train_n + i for i in py_local],
                }
            )

        results["select_add"] = select_results
        results["splits"] = splits

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
