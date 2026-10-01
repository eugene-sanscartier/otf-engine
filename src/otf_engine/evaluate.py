"""Evaluate one structure with an evaluator file: python -m otf_engine.evaluate EVALUATOR INPUT OUTPUT.

EVALUATOR is a Python file defining evaluator(structure), which labels the structure and returns it.
INPUT and OUTPUT are extxyz files holding the structure before and after.
"""
import argparse
import importlib.util
import os

import ase.io.extxyz


def command(binary):
    """The command that runs *binary* as the launcher sets it up for this evaluation.

    Prefixes binary with ``COMMAND_PREFIX``: ``"mpirun [runner_args]"`` (NestedLauncher),
    ``"srun [runner_args]"`` (SlurmLauncher, default runner_exec), ``""`` (ForkLauncher).
    """
    full_command = f"{os.environ.get('COMMAND_PREFIX', '')} {binary}".strip()
    return full_command


def load_evaluator(path):
    """Import the evaluator file at *path* as the module ``evaluator``."""
    spec = importlib.util.spec_from_file_location("evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(prog="python -m otf_engine.evaluate", description="Evaluate one structure with an evaluator file's evaluator(structure).")
    parser.add_argument("evaluator", help="Python file defining evaluator(structure)")
    parser.add_argument("input", help="extxyz file holding the structure")
    parser.add_argument("output", help="extxyz file the evaluated structure is written to")
    args = parser.parse_args()

    evaluator = load_evaluator(args.evaluator)
    with open(args.input) as f:
        structure = next(ase.io.extxyz.read_extxyz(f))
    ase.io.extxyz.write_extxyz(args.output, [evaluator.evaluator(structure)])


if __name__ == "__main__":
    main()
