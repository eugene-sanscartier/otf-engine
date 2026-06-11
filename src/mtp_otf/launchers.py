"""Execution backends for otf-mtp: mpirun (nested), fork (direct), and slurm (sbatch --wait)."""
from __future__ import annotations

import os
import sys
import shlex
import argparse
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------------

# Variables that prevent nested mpirun from working (parent's MPI job leaks in).
# These three are the set is known to work on a coumpute canada cluster with OpenMPI.
_MPIRUN_DELETE_VARS = frozenset({
    "OMPI_MCA_ess",
    "OMPI_MCA_ess_base_jobid",
    "OMPI_UNIVERSE_SIZE",
})

# Additional variables removed for fork mode (no mpirun wrapper).
# Job/session identity vars and parent rank vars are deleted so the child
# starts fresh without inheriting a stale rank or trying to rejoin the
# parent's MPI session.  Transport config vars (OMPI_MCA_btl, etc.) are
# intentionally kept.  This list is empirical — extend it when testing
# reveals additional problematic variables for your MPI implementation.
_FORK_REMOVE_VARS = frozenset({
    # Job / session identity
    "OMPI_MCA_ess",
    "OMPI_MCA_ess_base_jobid",
    "PMIX_SERVER_URI",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_STEPID",
    # Rank vars — child must not inherit parent's rank
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_NODE_RANK",
    "PMI_RANK",
    "PMIX_RANK",
})


def _env_for_nested(env: dict) -> dict:
    """Return env with nested-mpirun-breaking variables removed."""
    return {k: v for k, v in env.items() if k not in _MPIRUN_DELETE_VARS}


def _env_for_fork(size: int, env: dict) -> dict:
    """Return env for a forked process in a fresh MPI universe of *size* peers.

    Removes parent MPI context and rank assignment, then sets universe-size
    variables.  The MPI rank is NOT set — the library determines it via its
    rendezvous mechanism (shared memory files, PMI, etc.).
    """
    base = {k: v for k, v in env.items() if k not in _FORK_REMOVE_VARS}
    base["OMPI_UNIVERSE_SIZE"] = str(size)
    base["OMPI_COMM_WORLD_SIZE"] = str(size)
    base["PMI_SIZE"] = str(size)
    return base


def _physical_cpu_count() -> int:
    import psutil
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _default_n_procs() -> int:
    """Best-effort process count: Slurm allocation first, then physical CPU cores."""
    for var in ("SLURM_NTASKS", "SLURM_NPROCS"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return _physical_cpu_count()


def _join(parts: list[str]) -> str:
    return " ".join(parts)


# Used to strips task/node parallelism options from batch_args when parallel_eval=False for slurm sbatch.
_sbatch_parser = argparse.ArgumentParser(add_help=False)
_sbatch_parser.add_argument("--ntasks", "-n")
_sbatch_parser.add_argument("--ntasks-per-node")
_sbatch_parser.add_argument("--nodes", "-N")


def _run_cmd(cmd: str, log_file: str, env: dict) -> None:
    """Run a shell command string, appending combined output to log_file."""
    print(f"running: {cmd}")
    with open(log_file, "a") as f:
        subprocess.run(cmd, shell=True, text=True, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------------------------
# Launcher ABC
# ---------------------------------------------------------------------------


class Launcher(ABC):
    """Abstract execution backend.

    Subclasses implement how mlp binary calls are launched (mpirun, fork,
    sbatch, …) and optionally how the evaluator function is dispatched.
    """

    @abstractmethod
    def run(self, command: str, log_file: str, parallel_eval: bool = True) -> None:
        """Execute *command* (f-string of ``mlp binary + subcommand + args``,
        without any mpirun/srun prefix).

        Parameters
        ----------
        command:  Command string passed to the shell, e.g.
                  ``f"{mlp} calculate_grade {potential} ..."``
        log_file: Append combined stdout/stderr here.
        parallel_eval:  When True, use full parallelism; when False, restrict
                        to a single process (e.g. ``-n 1``).
        """

    def command_prefix(self, _parallel_eval: bool = True) -> str:
        """Full command prefix injected as ``COMMAND_PREFIX`` before calling the evaluator."""
        return ""

    def batch_prefix(self, _chdir: str, _log_file: str, _parallel_eval: bool = True) -> str:
        """Batch submission prefix: everything before ``--wrap="cmd"``."""
        return ""

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path):
        """Evaluate *structure* inside *eval_dir* and return the result."""
        eval_dir.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        os.chdir(eval_dir)
        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)

    @property
    def concurrent_eval(self) -> bool:
        """Whether eval_structures should submit evaluations concurrently via ThreadPoolExecutor."""
        return False


# ---------------------------------------------------------------------------
# NestedLauncher  (current behaviour, default)
# ---------------------------------------------------------------------------


class NestedLauncher(Launcher):
    """Wrap mlp calls with ``mpirun``; passes ``-n 1`` when ``parallel_eval=False``."""

    def __init__(self, runner_exec: str = "mpirun", runner_args: str = ""):
        if any(t in ("-n", "-np") for t in runner_args.split()):
            raise ValueError("runner_args must not contain '-n'/'-np'")
        self._runner_exec = runner_exec
        self.runner_args = runner_args

    def command_prefix(self, parallel_eval: bool = True) -> str:
        command_parts = [self._runner_exec]
        if not parallel_eval: command_parts += ["-n 1"]
        if self.runner_args: command_parts += [self.runner_args]
        command_prefix = _join(command_parts)
        return command_prefix

    def run(self, command: str, log_file: str, parallel_eval: bool = True) -> None:
        cmd = f"{self.command_prefix(parallel_eval)} {command}"
        _run_cmd(cmd, log_file, _env_for_nested(os.environ))

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path):
        eval_dir.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        os.chdir(eval_dir)
        saved = dict(os.environ)
        new_env = _env_for_nested(os.environ)
        new_env["COMMAND_PREFIX"] = self.command_prefix()
        os.environ.clear()
        os.environ.update(new_env)
        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)
            os.environ.clear()
            os.environ.update(saved)


# ---------------------------------------------------------------------------
# ForkLauncher  (direct exec, no mpirun wrapper)
# ---------------------------------------------------------------------------


class ForkLauncher(Launcher):
    """Run binaries directly without mpirun, constructing a fresh MPI universe
    via environment variables.

    The number of peer processes is taken from the current Slurm allocation
    when available, otherwise from ``os.cpu_count()``.
    """

    def __init__(self, parallel_eval: bool = True):
        self._parallel_eval = parallel_eval

    def run(self, command: str, log_file: str, parallel_eval: bool = True) -> None:
        n_procs = _default_n_procs() if parallel_eval else 1
        child_env = _env_for_fork(n_procs, os.environ)
        with open(log_file, "a") as log_f:
            procs = [subprocess.Popen(
                command,
                shell=True,
                env=child_env,
                text=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            ) for _ in range(n_procs)]
            for p in procs:
                ret = p.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, command)

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path):
        eval_dir.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        os.chdir(eval_dir)
        saved = dict(os.environ)
        new_env = _env_for_fork(_default_n_procs() if self._parallel_eval else 1, os.environ)
        new_env["COMMAND_PREFIX"] = self.command_prefix()
        os.environ.clear()
        os.environ.update(new_env)
        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)
            os.environ.clear()
            os.environ.update(saved)


# ---------------------------------------------------------------------------
# SlurmLauncher  (sbatch --wait with --wrap)
# ---------------------------------------------------------------------------


class SlurmLauncher(Launcher):
    """Submit mlp calls and evaluator jobs via ``sbatch --wait --wrap="cmd"``.

    Each ``launcher.run()`` call submits one Slurm batch job and blocks until
    it finishes (``--wait``).  Structure evaluations invoke ``evaluator.py``
    directly as a script (requires an ``if __name__ == "__main__":`` block
    with argparse in ``evaluator.py``).

    Structure evaluations are submitted concurrently by default
    (``concurrent_eval=True``): all sbatch jobs are submitted simultaneously
    from separate threads, each blocking on ``--wait``.  MPI parallelism is
    controlled by the job's resource allocation via ``batch_args``.

    ``COMMAND_PREFIX`` (``runner_exec``, default ``"srun"``) is set in the parent
    environment before submission and inherited by child jobs — ``evaluator.py``
    reads it via ``build_command()``.

    **Python environment requirement**: ``sys.executable`` must be on a shared
    filesystem accessible from all compute nodes (NFS/Lustre, /home, /project).

    Parameters
    ----------
    batch_exec:        Path to sbatch binary (default: ``"sbatch"``).
    batch_args:        Extra sbatch options, e.g. ``"--account=myaccount --partition=gpu --time=01:00:00"``.
    concurrent_eval:   Submit evaluations concurrently (default: True).
    runner_exec:       Command prefix for evaluator jobs, set as ``COMMAND_PREFIX`` (default: ``"srun"``).
    runner_args:       Extra arguments appended to ``runner_exec``, e.g. ``"--bind-to core"``.
    """

    def __init__(self, batch_exec: str = "sbatch", batch_args: str = "", concurrent_eval: bool = True, runner_exec: str = "srun", runner_args: str = ""):
        self.batch_exec = batch_exec
        self.batch_args = batch_args
        self._concurrent_eval = concurrent_eval
        self._runner_exec = runner_exec
        self.runner_args = runner_args

    @property
    def concurrent_eval(self) -> bool:
        return self._concurrent_eval

    def command_prefix(self, parallel_eval: bool = True) -> str:
        command_parts = [self._runner_exec]
        if not parallel_eval: command_parts += ["-n 1"]
        if self.runner_args: command_parts += [self.runner_args]
        command_prefix = _join(command_parts)
        return command_prefix

    def batch_prefix(self, chdir: str, log_file: str, parallel_eval: bool = True) -> str:
        batch_parts = [self.batch_exec, "--wait", f"--chdir={chdir}", f"--output={log_file}", "--quiet"]
        batch_args = shlex.split(self.batch_args)
        if not parallel_eval:
            _, batch_args = _sbatch_parser.parse_known_args(batch_args)
            batch_args += ["--ntasks=1", "--nodes=1"]
        batch_parts += batch_args
        batch_prefix = _join(batch_parts)
        return batch_prefix

    def run(self, command: str, log_file: str, parallel_eval: bool = True, chdir: str | None = None) -> None:
        cmd = f"{self.command_prefix(parallel_eval)} {command}"
        submit_cmd = f'{self.batch_prefix(chdir or os.getcwd(), log_file, parallel_eval)} --wrap="{cmd}"'
        print(f"running: {submit_cmd}")
        subprocess.run(submit_cmd, shell=True, env=os.environ, check=True)

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path):
        import ase.io.extxyz
        eval_dir.mkdir(parents=True, exist_ok=True)
        ase.io.extxyz.write_extxyz(os.path.join(eval_dir, "input_structure.extxyz"), [structure])
        os.environ["COMMAND_PREFIX"] = self.command_prefix()
        evaluator_py = os.path.relpath("evaluator.py", eval_dir)
        eval_cmd = _join([sys.executable, evaluator_py, "input_structure.extxyz", "output_structure.extxyz"])
        submit_cmd = f'{self.batch_prefix(eval_dir, "eval.log")} --wrap="{eval_cmd}"'
        print(f"running (eval): {submit_cmd}")
        subprocess.run(submit_cmd, shell=True, env=os.environ, check=True)
        with open(os.path.join(eval_dir, "output_structure.extxyz")) as f:
            return next(ase.io.extxyz.read_extxyz(f))
