"""Execution backends for otf-engine: mpirun (nested), fork (direct), and slurm (sbatch --wait)."""
from __future__ import annotations

import logging
import os
import sys
import shlex
import time
import argparse
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from wrapt import synchronized
import numpy

logger = logging.getLogger(__name__)


class JobTimedOut(subprocess.CalledProcessError):
    """Raised by SlurmLauncher when sbatch exits non-zero due to a Slurm TIME LIMIT."""

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


# Used to strip task/node parallelism options from batch_args when parallel_eval=False for slurm sbatch.
_sbatch_parser = argparse.ArgumentParser(add_help=False)
_sbatch_parser.add_argument("--ntasks", "-n")
_sbatch_parser.add_argument("--ntasks-per-node")
_sbatch_parser.add_argument("--nodes", "-N")

# ---------------------------------------------------------------------------
# Slurm time utilities
# ---------------------------------------------------------------------------

_stime_parser = argparse.ArgumentParser(add_help=False)
_stime_parser.add_argument("--time", "-t")


def _seconds_to_hms(s: float) -> str:
    s = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60

    return f"{h:02d}:{m:02d}:{sec:02d}"


def _hms_to_seconds(t: str) -> float | None:
    """Parse a Slurm time string (MM, MM:SS, HH:MM:SS, D-HH:MM:SS) into seconds."""
    try:
        t = t.strip()
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        parts = t.split(":")
        if len(parts) == 1:
            return float(days * 86400 + int(parts[0]) * 60)
        elif len(parts) == 2:
            return float(days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60)
        else:
            return float(days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    except (ValueError, TypeError):
        return None


def _parse_time_to_s(batch_args: str) -> float | None:
    parsed, _ = _stime_parser.parse_known_args(shlex.split(batch_args))

    return _hms_to_seconds(parsed.time) if parsed.time else None


def _is_slurm_timeout(log_path) -> bool:
    try:
        return any(all(cue in line for cue in ("CANCELLED AT", "DUE TO TIME LIMIT", "TIME_LIMIT")) for line in Path(log_path).read_text("utf-8", errors="ignore").splitlines())
    except OSError:
        return False


# ---------------------------------------------------------------------------
# TimingState
# ---------------------------------------------------------------------------


class TimingState:
    """Thread-safe timing observation store with adaptive time estimates."""

    window = 20
    safety = 1.25
    timeout_f = 2.0
    early_n = 3
    max_s = 7 * 24 * 3600
    ridge_alpha = 1e-3

    def __init__(self, data: dict, initial_time_s: float | None = None, on_record=None):
        self._d = data
        self._initial_time_s = initial_time_s
        self._on_record = on_record
        self._last_eval: dict = {}
        self._last_train: dict = {}
        self._refresh()

    @classmethod
    def load(cls, data: dict, initial_time_s: float | None = None, on_record=None) -> TimingState:
        return cls(dict(data), initial_time_s=initial_time_s, on_record=on_record)

    @synchronized
    def record_eval(self, elapsed_s: float, timed_out: bool, allocated_s: float | None):
        obs = self._d.setdefault("eval", {}).get("observations", [])
        obs = (obs + [{"elapsed": float(elapsed_s), "timed_out": bool(timed_out), "allocated": float(allocated_s) if allocated_s is not None else None}])[-self.window:]
        self._d["eval"]["observations"] = obs
        if elapsed_s >= (self._last_eval.get("eval_time_s") or 0.0): self._last_eval = {"eval_time_s": elapsed_s, "eval_time_alloc_s": allocated_s}
        self._refresh()
        if self._on_record:
            self._on_record(self.to_dict())

    @synchronized
    def record_train(self, elapsed_s: float, training_set_size: int | None, timed_out: bool, allocated_s: float | None):
        obs = self._d.setdefault("train", {}).get("observations", [])
        obs = (obs + [{"size": training_set_size, "elapsed": float(elapsed_s), "timed_out": bool(timed_out), "allocated": float(allocated_s) if allocated_s is not None else None}])[-self.window:]
        self._d["train"]["observations"] = obs
        self._last_train = {"train_time_s": elapsed_s, "train_time_alloc_s": allocated_s}
        self._refresh()
        if self._on_record:
            self._on_record(self.to_dict())

    @synchronized
    def estimate_eval(self) -> float | None:
        if self._eval_times is None:
            return self._initial_time_s
        return numpy.mean(self._eval_times) * self.safety

    @synchronized
    def estimate_train(self, next_size: int | None) -> float | None:
        if self._train_times is None:
            return self._initial_time_s
        base = max(self._train_times) if len(self._train_times) <= self.early_n else numpy.mean(self._train_times)
        fallback = min(base * self.safety, self.max_s)
        if self._train_sizes is None or next_size is None or numpy.unique(self._train_sizes).size < 2:
            return fallback
        if numpy.ptp(self._train_sizes) < 0.05 * numpy.mean(self._train_sizes):
            return fallback
        s_mean = numpy.mean(self._train_sizes)
        X = numpy.column_stack([self._train_sizes - s_mean, numpy.ones_like(self._train_sizes)])
        lam = self.ridge_alpha * numpy.mean((self._train_sizes - s_mean)**2)
        w = numpy.linalg.solve(X.T @ X + numpy.diag([lam, 0.0]), X.T @ self._train_t_vals)
        return min(max(w[0] * (next_size - s_mean) + w[1], 0.0) * self.safety, self.max_s)

    @synchronized
    def to_dict(self) -> dict:
        return dict(self._d)

    def _refresh(self):
        obs = self._d.get("eval", {}).get("observations", [])
        if obs:
            tf = self.timeout_f if any(o["timed_out"] for o in obs) else 1.0
            self._eval_times = [o["allocated"] * tf if o["timed_out"] else o["elapsed"] for o in obs]
        else:
            self._eval_times = None

        obs = self._d.get("train", {}).get("observations", [])
        if not obs:
            self._train_times = self._train_sizes = self._train_t_vals = None
            return

        tf = self.timeout_f if any(o["timed_out"] for o in obs) else 1.0
        self._train_times = [o["allocated"] * tf if o["timed_out"] else o["elapsed"] for o in obs]

        sized = [o for o in obs if o["size"] is not None]
        if len(sized) < 2:
            self._train_sizes = self._train_t_vals = None
            return

        self._train_sizes = numpy.array([o["size"] for o in sized], dtype=float)
        self._train_t_vals = numpy.array([o["allocated"] * tf if o["timed_out"] else o["elapsed"] for o in sized], dtype=float)



# ---------------------------------------------------------------------------
# Launcher ABC
# ---------------------------------------------------------------------------


class Launcher(ABC):
    """Abstract execution backend.

    Subclasses implement ``_run_impl()`` and optionally ``_call_evaluator_impl()``.
    The public ``run()`` and ``call_evaluator()`` are concrete template methods that
    handle timing instrumentation via an optional ``TimingState``.
    """

    timing: TimingState | None = None

    def configure_timing(self, state: dict, save_state_fn) -> None:
        """Load timing from *state* and wire auto-save so timeouts are always persisted."""

        def _on_record(timing_dict):
            state["timing"] = timing_dict
            save_state_fn(state)

        self.timing = TimingState.load(state.get("timing", {}), initial_time_s=getattr(self, "_initial_time_s", None), on_record=_on_record)

    def run(self, command: str, log_file: str, parallel_eval: bool = True, training_set_size: int | None = None) -> None:
        """Execute *command* with optional timing instrumentation.

        Parameters
        ----------
        command:            mlp binary + subcommand + args (no runner prefix).
        log_file:           Append combined stdout/stderr here.
        parallel_eval:      When True, use full parallelism; False restricts to one process.
        training_set_size:  Number of training structures — used for time estimation.
        """
        time_s = self.timing.estimate_train(training_set_size) if self.timing else None

        if time_s:
            logger.info(f"Training time estimate: {time_s:.0f}s for {training_set_size} structures")
        t0 = time.monotonic()
        exc = None

        try:
            self._run_impl(command, log_file, parallel_eval, time_limit_s=time_s)
        except Exception as e:
            exc = e
            raise
        finally:
            elapsed = time.monotonic() - t0
            if self.timing:
                timed_out = isinstance(exc, JobTimedOut)
                if timed_out:
                    logger.warning(f"Training timed out ({elapsed:.0f}s elapsed vs {time_s:.0f}s allocated)")
                if not exc or timed_out:
                    self.timing.record_train(elapsed, training_set_size, timed_out, time_s)

    @abstractmethod
    def _run_impl(self, command: str, log_file: str, parallel_eval: bool = True, time_limit_s: float | None = None) -> None:
        """Backend-specific command execution."""

    def command_prefix(self, _parallel_eval: bool = True) -> str:
        """Full command prefix injected as ``COMMAND_PREFIX`` before calling the evaluator."""
        return ""

    def batch_prefix(self, _chdir: str, _log_file: str, _parallel_eval: bool = True) -> str:
        """Batch submission prefix: everything before ``--wrap="cmd"``."""
        return ""

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path):
        """Evaluate *structure* inside *eval_dir* with optional timing instrumentation."""
        time_s = self.timing.estimate_eval() if self.timing else None

        if time_s:
            logger.info(f"Eval time estimate: {time_s:.0f}s per structure")
        t0 = time.monotonic()
        exc = None

        try:
            return self._call_evaluator_impl(evaluator_fn, structure, eval_dir, time_limit_s=time_s)
        except Exception as e:
            exc = e
            raise
        finally:
            elapsed = time.monotonic() - t0
            if self.timing:
                timed_out = isinstance(exc, JobTimedOut)
                if timed_out:
                    logger.warning(f"Eval timed out ({elapsed:.0f}s elapsed vs {time_s:.0f}s allocated)")
                if not exc or timed_out:
                    self.timing.record_eval(elapsed, timed_out, time_s)

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, time_limit_s: float | None = None):
        """Default evaluator dispatch: cd into eval_dir, call evaluator_fn directly."""
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

        return _join(command_parts)

    def _run_impl(self, command: str, log_file: str, parallel_eval: bool = True, time_limit_s: float | None = None) -> None:
        cmd = f"{self.command_prefix(parallel_eval)} {command}"
        logger.info(f"running: {cmd}")

        with open(log_file, "a") as file_obj:
            subprocess.run(cmd, shell=True, text=True, env=_env_for_nested(os.environ), stdout=file_obj, stderr=subprocess.STDOUT, check=True)

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, time_limit_s: float | None = None):
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

    def _run_impl(self, command: str, log_file: str, parallel_eval: bool = True, time_limit_s: float | None = None) -> None:
        n_procs = _default_n_procs() if parallel_eval else 1
        child_env = _env_for_fork(n_procs, os.environ)

        with open(log_file, "a") as log_f:
            procs = [subprocess.Popen(command, shell=True, env=child_env, text=True, stdout=log_f, stderr=subprocess.STDOUT) for _ in range(n_procs)]
            for p in procs:
                ret = p.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, command)

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, time_limit_s: float | None = None):
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
        self._initial_time_s = _parse_time_to_s(batch_args)

    @property
    def concurrent_eval(self) -> bool:
        return self._concurrent_eval

    def command_prefix(self, parallel_eval: bool = True) -> str:
        command_parts = [self._runner_exec]
        if not parallel_eval: command_parts += ["-n 1"]
        if self.runner_args: command_parts += [self.runner_args]
        return _join(command_parts)

    def batch_prefix(self, chdir: str, log_file: str, parallel_eval: bool = True, time_limit_s: float | None = None) -> str:
        batch_parts = [self.batch_exec, "--wait", f"--chdir={chdir}", f"--output={log_file}", "--quiet"]
        batch_args = shlex.split(self.batch_args)
        if not parallel_eval:
            _, batch_args = _sbatch_parser.parse_known_args(batch_args)
            batch_args += ["--ntasks=1", "--nodes=1"]

        if time_limit_s is not None:
            _, batch_args = _stime_parser.parse_known_args(batch_args)
            batch_args += [f"--time={_seconds_to_hms(time_limit_s)}"]

        batch_parts += batch_args
        return _join(batch_parts)

    def _run_impl(self, command: str, log_file: str, parallel_eval: bool = True, time_limit_s: float | None = None) -> None:
        cmd = f"{self.command_prefix(parallel_eval)} {command}"
        submit_cmd = f'{self.batch_prefix(os.getcwd(), log_file, parallel_eval, time_limit_s=time_limit_s)} --wrap="{cmd}"'
        logger.info(f"running: {submit_cmd}")
        try:
            subprocess.run(submit_cmd, shell=True, env=os.environ, check=True)
        except subprocess.CalledProcessError as exc:
            if _is_slurm_timeout(log_file): raise JobTimedOut(exc.returncode, exc.cmd) from exc
            raise

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, time_limit_s: float | None = None):
        import ase.io.extxyz
        eval_dir.mkdir(parents=True, exist_ok=True)
        ase.io.extxyz.write_extxyz(os.path.join(eval_dir, "input_structure.extxyz"), [structure])
        os.environ["COMMAND_PREFIX"] = self.command_prefix()
        evaluator_py = os.path.relpath("evaluator.py", eval_dir)
        eval_cmd = _join([sys.executable, evaluator_py, "input_structure.extxyz", "output_structure.extxyz"])
        submit_cmd = f'{self.batch_prefix(eval_dir, "eval.log", time_limit_s=time_limit_s)} --wrap="{eval_cmd}"'
        logger.info(f"running (eval): {submit_cmd}")
        try:
            subprocess.run(submit_cmd, shell=True, env=os.environ, check=True)
        except subprocess.CalledProcessError as exc:
            if _is_slurm_timeout(eval_dir / "eval.log"): raise JobTimedOut(exc.returncode, exc.cmd) from exc
            raise
        with open(os.path.join(eval_dir, "output_structure.extxyz")) as f:
            return next(ase.io.extxyz.read_extxyz(f))
