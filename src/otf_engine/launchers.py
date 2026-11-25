"""Execution backends of otf-engine: mpirun (nested), fork (direct) and Slurm (sbatch --wait)."""
from __future__ import annotations

import logging
import math
import os
import re
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
    """Error that SlurmLauncher raises when sbatch exits non-zero after a Slurm time limit.

    Attributes:
        time_limit_s: Limit that the job was submitted with, set by call_evaluator.
    """

    time_limit_s: float | None = None


class JobOutOfMemory(subprocess.CalledProcessError):
    """Error that SlurmLauncher raises when a job was OOM-killed and it can still request a larger ``--mem``."""

# ---------------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------------

# Variables of the parent MPI job that break a nested mpirun under OpenMPI.
_MPIRUN_DELETE_VARS = frozenset({
    "OMPI_MCA_ess",
    "OMPI_MCA_ess_base_jobid",
    "OMPI_UNIVERSE_SIZE",
})

# Variables removed in fork mode: the job, session and rank of the parent MPI process.
# The transport settings (OMPI_MCA_btl, ...) stay.
_FORK_REMOVE_VARS = frozenset({
    # Job / session identity
    "OMPI_MCA_ess",
    "OMPI_MCA_ess_base_jobid",
    "PMIX_SERVER_URI",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_STEPID",
    # Rank of the parent process
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_NODE_RANK",
    "PMI_RANK",
    "PMIX_RANK",
})


def _env_for_nested(env: dict) -> dict:
    """Return *env* without the variables that break a nested mpirun."""
    return {k: v for k, v in env.items() if k not in _MPIRUN_DELETE_VARS}


def _env_for_fork(size: int, env: dict) -> dict:
    """Return the environment of a forked process in a new MPI universe of *size* peers.

    The function removes the MPI context and rank of the parent, then sets the
    universe size. The MPI library assigns the rank at rendezvous (shared memory
    files, PMI, ...).
    """
    base = {k: v for k, v in env.items() if k not in _FORK_REMOVE_VARS}
    base["OMPI_UNIVERSE_SIZE"] = str(size)
    base["OMPI_COMM_WORLD_SIZE"] = str(size)
    base["PMI_SIZE"] = str(size)

    return base


# A new Slurm job drops the variables of one instance of the submitter (its job, srun
# step, mpirun launch, rank or process). It keeps those of the compute environment
# (cluster, Slurm config, modules, user settings).

_SBATCH_REMOVE_VAR_PREFIXES = (
    # Launch and rank scope: the PMIx and PMI server connection and process identity,
    # the Open MPI world layout and launch markers.
    "PMIX_", "PMI_", "OMPI_COMM_WORLD_", "OMPI_WORLD_",
    # Per-launch runtime state in OMPI_MCA_*: daemon URIs, session dirs, transport key,
    # jobid/vpid, size, cwd, binding state. The settings (btl, pml, plm_slurm_args, ...) stay.
    "OMPI_MCA_orte_", "OMPI_MCA_ess", "OMPI_MCA_prte_",
    # Job and step scope: the variables that Slurm sets for one allocation or srun step.
    "SLURM_", "SLURMD_", "SRUN_DEBUG",
)
_SBATCH_REMOVE_VARS = frozenset({
    "OMPI_UNIVERSE_SIZE", "OMPI_APP_CTX_NUM_PROCS", "OMPI_NUM_APP_CTX", "OMPI_FIRST_RANKS",
    "OMPI_ARGV", "OMPI_COMMAND", "OMPI_FILE_LOCATION", "OMPI_TOOL_NAME", "OMPI_LIBDIR_LOC",
    "OMPI_VERSION", "OPAL_USER_PARAMS_GIVEN", "ORTE_SCHIZO_DETECTION", "PRTE_LAUNCHED", "PRTE_SHARED_FS",
    "OMPI_MCA_pmix", "OMPI_MCA_num_procs", "OMPI_MCA_initial_wdir", "OMPI_MCA_mpi_oversubscribe",
    "OMPI_MCA_hwloc_base_binding_policy", "OMPI_MCA_cpu_type", "OMPI_MCA_shmem_RUNTIME_QUERY_hint",
    "OMPI_MCA_PREFIXES",
    # Process scope: defaults that MPI_Init sets for each process.
    "PSM2_DEVICES", "HFI_NO_BACKTRACE", "IPATH_NO_BACKTRACE", "ZES_ENABLE_SYSMAN", "HOSTNAME",
    "LOCAL_SCRATCH", "XDG_RUNTIME_DIR", "ENVIRONMENT",
})
# Cluster properties in the SLURM_ namespace, which stay.
_SLURM_ENVIRONMENT_NAMES = frozenset({"SLURM_MPI_TYPE", "SLURM_CONF"})


def _env_for_sbatch(env: dict) -> dict:
    """Return the environment of a new Slurm job: *env* without the job, step, launch, rank and process state of the submitter."""
    return {k: v for k, v in env.items() if k in _SLURM_ENVIRONMENT_NAMES or not (k.startswith(_SBATCH_REMOVE_VAR_PREFIXES) or k in _SBATCH_REMOVE_VARS)}


def _physical_cpu_count() -> int:
    import psutil
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _default_n_procs() -> int:
    """Return the task count of the Slurm allocation, else the physical core count."""
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


# Parses the task and node options to strip from batch_args for a single_rank Slurm job.
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
    """Parse a Slurm time string (M, M:S, H:M:S, D-H, D-H:M, D-H:M:S) into seconds, or None for any other form."""
    try:
        days, _, clock = t.strip().rpartition("-")
        parts = [int(p) for p in clock.split(":")]
        if days: parts += [0] * (3 - len(parts))
        elif len(parts) < 3: parts = [0] + parts + [0] * (2 - len(parts))
        h, m, s = parts
        return float(int(days or 0) * 86400 + h * 3600 + m * 60 + s)
    except (ValueError, TypeError):
        return None


def _parse_time_to_s(batch_args: str) -> float | None:
    parsed, _ = _stime_parser.parse_known_args(shlex.split(batch_args))

    return _hms_to_seconds(parsed.time) if parsed.time else None


# slurmstepd: error: *** STEP 17136522.0 ON c149 CANCELLED AT 2026-07-03T08:33:56 DUE TO TIME LIMIT ***
# slurmstepd: error: *** JOB 17136522 ON c149 CANCELLED AT 2026-07-03T08:33:56 DUE TO TIME LIMIT ***

_SLURM_TIMEOUT_RE = re.compile(r"JOB.*CANCELLED AT.*DUE TO TIME LIMIT")


def _is_slurm_timeout(log_path) -> bool:
    try:
        return bool(_SLURM_TIMEOUT_RE.search(Path(log_path).read_text("utf-8", errors="ignore")))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Slurm memory utilities
# ---------------------------------------------------------------------------

_MEM_UNITS = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0 ** 2}

# Parses --mem, and --mem-per-cpu to strip it where --mem is set.
_smem_parser = argparse.ArgumentParser(add_help=False)
_smem_parser.add_argument("--mem")
_smem_parser.add_argument("--mem-per-cpu")


def _mem_to_mb(s: str) -> float | None:
    """Parse a Slurm memory value (e.g. '58120K', '3075208K', '512M') into MB."""
    s = s.strip()
    if not s:
        return None
    unit = s[-1].upper()
    try:
        return float(s[:-1]) * _MEM_UNITS[unit] if unit in _MEM_UNITS else float(s)
    except ValueError:
        return None


def _parse_mem_to_mb(batch_args: str) -> float | None:
    """Parse the ``--mem`` ceiling out of sbatch arguments, in MB.

    ``--mem=0``, all the memory of the node, sets no ceiling, and neither does ``--mem-per-cpu``.
    """
    parsed, _ = _smem_parser.parse_known_args(shlex.split(batch_args))

    return (_mem_to_mb(parsed.mem) if parsed.mem else None) or None


# slurmstepd: error: Detected 1 oom_kill event in StepId=17136522.batch. Some of the job steps have been OOM Killed.
# slurmstepd: error: Exceeded job memory limit

_SLURM_OOM_RE = re.compile(r"oom[-_]kill|OOM Killed|Exceeded (?:job|step) memory limit", re.IGNORECASE)


def _is_slurm_oom(log_path) -> bool:
    try:
        return bool(_SLURM_OOM_RE.search(Path(log_path).read_text("utf-8", errors="ignore")))
    except OSError:
        return False


def _sacct_job(job_id: str) -> tuple[bool, float | None]:
    """Query sacct for *job_id*: whether a step was killed OUT_OF_MEMORY, and the peak memory in MB.

    The peak is TRESUsageInTot, the cgroup total of each step, which counts the
    child processes of an mpirun inside --wrap. --parsable keeps the fields
    untruncated. When the accounting lags behind ``sbatch --wait``, the state
    and the usage come back unknown.
    """
    try:
        out = subprocess.run(["sacct", "-j", job_id, "--format=State,TRESUsageInTot", "--noheader", "--parsable"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return False, None
    values = [v for line in out.splitlines() for field in line.split(",") if field.startswith("mem=") for v in [_mem_to_mb(field[len("mem="):])] if v is not None]

    return "OUT_OF_MEMORY" in out, (max(values) if values else None)


# ---------------------------------------------------------------------------
# TimingState
# ---------------------------------------------------------------------------


class TimingState:
    """Thread-safe timing observation store with adaptive time estimates."""

    window = 20
    safety = 1.25
    timeout_f = 2.0
    n_bootstrap = 3
    max_s = 7 * 24 * 3600
    ridge_alpha = 1e-3

    def __init__(self, data: dict, initial_time_s: float | None = None, on_record=None):
        self._d = data
        self._initial_time_s = initial_time_s
        self._on_record = on_record
        self._last_eval: dict = {}
        self._last_train: dict = {}
        self._compute_estimates()

    @classmethod
    def load(cls, data: dict, initial_time_s: float | None = None, on_record=None) -> TimingState:
        """Load the observations in *data*. cycle_times() covers only what the new instance records."""
        return cls({kind: data[kind] for kind in ("eval", "train") if kind in data}, initial_time_s=initial_time_s, on_record=on_record)

    @synchronized
    def record_eval(self, elapsed_s: float, timed_out: bool, allocated_s: float | None):
        obs = self._d.setdefault("eval", {}).get("observations", [])
        obs = (obs + [{"elapsed": float(elapsed_s), "timed_out": bool(timed_out), "allocated": float(allocated_s) if allocated_s is not None else None}])[-self.window:]
        self._d["eval"]["observations"] = obs
        if elapsed_s >= (self._last_eval.get("eval_time_s") or 0.0): self._last_eval = {"eval_time_s": elapsed_s, "eval_time_alloc_s": allocated_s}
        self._compute_estimates()
        if self._on_record:
            self._on_record(self.to_dict())

    @synchronized
    def record_train(self, elapsed_s: float, training_set_size: int | None, timed_out: bool, allocated_s: float | None):
        obs = self._d.setdefault("train", {}).get("observations", [])
        obs = (obs + [{"size": training_set_size, "elapsed": float(elapsed_s), "timed_out": bool(timed_out), "allocated": float(allocated_s) if allocated_s is not None else None}])[-self.window:]
        self._d["train"]["observations"] = obs
        self._last_train = {"train_time_s": elapsed_s, "train_time_alloc_s": allocated_s}
        self._compute_estimates()
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
        base = max(self._train_times) if len(self._train_times) <= self.n_bootstrap else numpy.mean(self._train_times)
        fallback = min(base * self.safety, self.max_s)
        if self._train_sizes is None or next_size is None or numpy.unique(self._train_sizes).size < 2:
            return fallback
        if numpy.ptp(self._train_sizes) < 0.05 * numpy.mean(self._train_sizes):
            return fallback
        s_mean = numpy.mean(self._train_sizes)
        X = numpy.column_stack([self._train_sizes - s_mean, numpy.ones_like(self._train_sizes)])
        lam = self.ridge_alpha * numpy.mean((self._train_sizes - s_mean)**2)
        w = numpy.linalg.solve(X.T @ X + numpy.diag([lam, 0.0]), X.T @ self._train_t_vals)
        regression_est = max(w[0] * (next_size - s_mean) + w[1], 0.0) * self.safety
        return min(max(regression_est, fallback), self.max_s)

    @synchronized
    def to_dict(self) -> dict:
        return dict(self._d)

    @synchronized
    def cycle_times(self) -> dict:
        """Return the longest evaluation and the last training recorded since loading, with the limits they were given."""
        return {"eval_time_s": None, "eval_time_alloc_s": None, "train_time_s": None, "train_time_alloc_s": None, **self._last_eval, **self._last_train}

    def _compute_estimates(self):
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
# MemoryState
# ---------------------------------------------------------------------------


class MemoryState:
    """Thread-safe store of the peak memory in MB seen so far per job *kind* ("eval", "train").

    The estimate is ``max_seen * safety``, capped at ``ceiling_mb``. Before any
    observation the estimate is ``ceiling_mb`` itself, so a job goes without
    ``--mem`` only when there is neither an observation nor a ceiling.
    SlurmLauncher uses the estimate to size ``--mem`` of the next jobs.
    """

    safety = 1.5

    def __init__(self, data: dict, ceiling_mb: float | None = None, on_record=None):
        self._d = data
        self._ceiling_mb = ceiling_mb
        self._on_record = on_record

    @classmethod
    def load(cls, data: dict, ceiling_mb: float | None = None, on_record=None) -> MemoryState:
        return cls(dict(data), ceiling_mb=ceiling_mb, on_record=on_record)

    @synchronized
    def record(self, kind: str, mem_mb: float):
        key = f"{kind}_max_mb"
        self._d[key] = max(self._d.get(key, 0.0), float(mem_mb))
        if self._on_record:
            self._on_record(self.to_dict())

    @synchronized
    def estimate(self, kind: str) -> float | None:
        mem_mb = self._d.get(f"{kind}_max_mb")
        if not mem_mb:
            return self._ceiling_mb
        request_mb = mem_mb * self.safety

        return min(request_mb, self._ceiling_mb) if self._ceiling_mb else request_mb

    @synchronized
    def to_dict(self) -> dict:
        return dict(self._d)


# ---------------------------------------------------------------------------
# Launcher ABC
# ---------------------------------------------------------------------------


class Launcher(ABC):
    """Abstract execution backend.

    Subclasses implement ``_run_impl()``, and optionally ``_child_env()`` and ``_run_evaluator()``.
    ``run()`` and ``call_evaluator()`` time the jobs through an optional ``TimingState``.
    """

    timing: TimingState | None = None
    memory: MemoryState | None = None
    max_retries: int = 0

    def configure_timing(self, state: dict, save_state_fn) -> None:
        """Load the timing from *state*, and save *state* through *save_state_fn* after each record."""

        def _on_record(timing_dict):
            state["timing"] = timing_dict
            save_state_fn(state)

        self.timing = TimingState.load(state.get("timing", {}), initial_time_s=getattr(self, "_initial_time_s", None), on_record=_on_record)

    def configure_memory(self, state: dict, save_state_fn) -> None:
        """Load the memory usage from *state*, and save *state* through *save_state_fn* after each record.

        Only SlurmLauncher uses the memory usage.
        """

        def _on_record(memory_dict):
            state["memory"] = memory_dict
            save_state_fn(state)

        self.memory = MemoryState.load(state.get("memory", {}), ceiling_mb=getattr(self, "_mem_ceiling_mb", None), on_record=_on_record)

    def run(self, command: str, log_file: str, single_rank: bool = False, training_set_size: int | None = None, _backoff: int | None = None) -> None:
        """Run *command* with optional timing, and retry it on a timeout or OOM.

        Args:
            command: mlp binary, subcommand and arguments, without the runner prefix.
            log_file: File that receives the combined stdout and stderr.
            single_rank: Run the command as one rank (one MPI process, one Slurm task) instead of the full parallelism of the launcher.
            training_set_size: Number of training structures, for the time estimate.
        """
        if _backoff is None: _backoff = self.max_retries
        time_s = self.timing.estimate_train(training_set_size) if self.timing else None
        if time_s:
            logger.info(f"Training time estimate: {_seconds_to_hms(time_s)} for {training_set_size} structures")
        t0 = time.monotonic()
        try:
            self._run_impl(command, log_file, single_rank, time_limit_s=time_s)
        except JobTimedOut:
            elapsed = time.monotonic() - t0
            # Without its own --time, the job died at the partition limit, and elapsed stands for that limit.
            limit_s = time_s or elapsed
            logger.warning(f"Training timed out ({_seconds_to_hms(elapsed)} elapsed vs {_seconds_to_hms(time_s) if time_s else 'the partition default'} allocated)")
            if self.timing:
                self.timing.record_train(elapsed, training_set_size, True, limit_s)
            if _backoff > 0:
                logger.info(f"Retrying training with new estimate ({_backoff} left)...")
                return self.run(command, log_file, single_rank, training_set_size, _backoff=_backoff - 1)
            raise
        except JobOutOfMemory:
            if _backoff > 0:
                logger.info(f"Retrying training with a larger memory request ({_backoff} left)...")
                return self.run(command, log_file, single_rank, training_set_size, _backoff=_backoff - 1)
            raise
        elapsed = time.monotonic() - t0
        if self.timing:
            self.timing.record_train(elapsed, training_set_size, False, time_s)

    @abstractmethod
    def _run_impl(self, command: str, log_file: str, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        """Run *command* on this backend."""

    def command_prefix(self, _single_rank: bool = False) -> str:
        """Return the command prefix that the evaluator gets as ``COMMAND_PREFIX``."""
        return ""

    def batch_prefix(self, _chdir: str, _log_file: str, _single_rank: bool = False, _time_limit_s: float | None = None, _mem_mb: float | None = None) -> str:
        """Return the batch submission prefix: everything before ``--wrap="cmd"``."""
        return ""

    def _child_env(self, _single_rank: bool = False) -> dict:
        """Return the environment of a process that this launcher starts."""
        return dict(os.environ)

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path, timed_out_at_s: float | None = None, single_rank: bool = False):
        """Evaluate *structure* inside *eval_dir* once, with optional timing.

        Args:
            evaluator_fn: Function called in-process. Without it, ./evaluator.py runs on the structure.
            timed_out_at_s: Limit at which this structure last timed out. The job gets at least ``timeout_f`` times that limit.
            single_rank: Run the program of the evaluator as one rank.

        Raises:
            JobTimedOut: The evaluation hit its time limit.
            JobOutOfMemory: The evaluation ran out of memory.
        """
        time_s = self.timing.estimate_eval() if self.timing else None
        if timed_out_at_s: time_s = max(time_s or 0.0, TimingState.timeout_f * timed_out_at_s)
        if time_s:
            logger.info(f"Eval time estimate: {_seconds_to_hms(time_s)} per structure")
        t0 = time.monotonic()
        try:
            result = self._call_evaluator_impl(evaluator_fn, structure, eval_dir, single_rank, time_limit_s=time_s)
        except JobTimedOut as e:
            elapsed = time.monotonic() - t0
            # Without its own --time, the job died at the partition limit, and elapsed stands for that limit.
            limit_s = time_s or elapsed
            logger.warning(f"Eval timed out ({_seconds_to_hms(elapsed)} elapsed vs {_seconds_to_hms(time_s) if time_s else 'the partition default'} allocated)")
            if self.timing:
                self.timing.record_eval(elapsed, True, limit_s)
            e.time_limit_s = limit_s
            raise
        elapsed = time.monotonic() - t0
        if self.timing:
            self.timing.record_eval(elapsed, False, time_s)
        return result

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, single_rank: bool = False, time_limit_s: float | None = None):
        """Evaluate *structure* inside *eval_dir* with ``COMMAND_PREFIX`` set.

        With *evaluator_fn*, call it in-process. Without it, run evaluator() of ./evaluator.py in a process of its own.
        """
        import ase.io.extxyz
        eval_dir.mkdir(parents=True, exist_ok=True)
        if evaluator_fn is None:
            ase.io.extxyz.write_extxyz(eval_dir / "input_structure.extxyz", [structure])
            self._run_evaluator(_join([sys.executable, "-m", "otf_engine.evaluate", os.path.relpath("evaluator.py", eval_dir), "input_structure.extxyz", "output_structure.extxyz"]), eval_dir, single_rank, time_limit_s)
            with open(eval_dir / "output_structure.extxyz") as f:
                return next(ase.io.extxyz.read_extxyz(f))

        prev = os.getcwd()
        os.chdir(eval_dir)
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self._child_env(single_rank), COMMAND_PREFIX=self.command_prefix(single_rank))

        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)
            os.environ.clear()
            os.environ.update(saved)

    def _run_evaluator(self, command: str, eval_dir: Path, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        """Run the evaluator *command* inside *eval_dir* and append its output to eval.log."""
        env = {**self._child_env(single_rank), "COMMAND_PREFIX": self.command_prefix(single_rank)}
        with open(eval_dir / "eval.log", "a") as log_f:
            subprocess.run(command, shell=True, cwd=eval_dir, env=env, text=True, stdout=log_f, stderr=subprocess.STDOUT, check=True)

    @property
    def concurrent_eval(self) -> bool:
        """Whether OTFCycle.evaluate runs the evaluations concurrently on a ThreadPoolExecutor."""
        return False


# ---------------------------------------------------------------------------
# NestedLauncher  (default)
# ---------------------------------------------------------------------------


class NestedLauncher(Launcher):
    """Launcher that wraps mlp calls with ``mpirun``, with ``-n 1`` for a ``single_rank`` call."""

    def __init__(self, runner_exec: str = "mpirun", runner_args: str = ""):
        if any(t in ("-n", "-np") for t in runner_args.split()):
            raise ValueError("runner_args must not contain '-n'/'-np'")

        self._runner_exec = runner_exec
        self.runner_args = runner_args

    def command_prefix(self, single_rank: bool = False) -> str:
        command_parts = [self._runner_exec]
        if single_rank: command_parts += ["-n 1"]
        if self.runner_args: command_parts += [self.runner_args]

        return _join(command_parts)

    def _child_env(self, single_rank: bool = False) -> dict:
        return _env_for_nested(os.environ)

    def _run_impl(self, command: str, log_file: str, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        cmd = f"{self.command_prefix(single_rank)} {command}"
        logger.info(f"running: {cmd}")

        with open(log_file, "a") as file_obj:
            subprocess.run(cmd, shell=True, text=True, env=self._child_env(single_rank), stdout=file_obj, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------------------------
# ForkLauncher  (direct exec without mpirun)
# ---------------------------------------------------------------------------


class ForkLauncher(Launcher):
    """Launcher that runs binaries without mpirun, in a new MPI universe set up by environment variables.

    The peer count is the task count of the Slurm allocation, else the physical core count.
    """

    def _child_env(self, single_rank: bool = False) -> dict:
        return _env_for_fork(1 if single_rank else _default_n_procs(), os.environ)

    def _run_impl(self, command: str, log_file: str, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        n_procs = 1 if single_rank else _default_n_procs()
        child_env = _env_for_fork(n_procs, os.environ)

        with open(log_file, "a") as log_f:
            procs = [subprocess.Popen(command, shell=True, env=child_env, text=True, stdout=log_f, stderr=subprocess.STDOUT) for _ in range(n_procs)]
            for p in procs:
                ret = p.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, command)


# ---------------------------------------------------------------------------
# SlurmLauncher  (sbatch --wait with --wrap)
# ---------------------------------------------------------------------------


class SlurmLauncher(Launcher):
    """Launcher that submits mlp calls and evaluator jobs with ``sbatch --wait --wrap="cmd"``.

    Each ``run()`` call submits one Slurm batch job and blocks until it finishes.
    A structure evaluation runs ``evaluator(structure)`` of ``evaluator.py`` through
    ``python -m otf_engine.evaluate``, and never calls an ``evaluator_fn``.

    With ``concurrent_eval``, the launcher submits all evaluation jobs at once from
    separate threads, each blocking on ``--wait``. ``batch_args`` sets the resources
    of each job, and with them its MPI parallelism.

    The command line of each evaluation job sets ``COMMAND_PREFIX`` to ``runner_exec``
    and ``runner_args``. ``evaluator.py`` reads it through ``otf_engine.evaluate.command()``.

    ``sys.executable`` must be on a filesystem that all compute nodes share (NFS/Lustre, /home, /project).

    Memory sizing: after each job, the launcher queries ``sacct`` for the peak memory
    and stores the running maximum per kind (eval, train) in ``MemoryState``. The next
    jobs request ``--mem`` at 1.5 times that maximum, capped at the ``--mem`` of
    ``batch_args``. Until a job has completed once, they request the ceiling itself.
    The launcher drops ``--mem-per-cpu`` from ``batch_args`` when it sets ``--mem``,
    and ``--mem-per-cpu`` sets no ceiling.

    Out-of-memory recovery: the launcher records the request at which a job was
    OOM-killed as a lower bound on its demand, and the next job asks for more. A
    training job is resubmitted at once, up to ``max_retries`` times. An evaluation
    raises ``JobOutOfMemory``, and its cycle defers it. Once the request reaches the
    ceiling, the job fails.

    Args:
        batch_exec: Path to the sbatch binary.
        batch_args: Extra sbatch options, e.g. ``"--account=myaccount --partition=gpu --time=01:00:00"``.
        concurrent_eval: Submit the evaluations concurrently.
        runner_exec: Command prefix of evaluator jobs, set as ``COMMAND_PREFIX``.
        runner_args: Extra arguments appended to ``runner_exec``, e.g. ``"--bind-to core"``.
    """

    def __init__(self, batch_exec: str = "sbatch", batch_args: str = "", concurrent_eval: bool = True, runner_exec: str = "srun", runner_args: str = "", max_retries: int = TimingState.n_bootstrap):
        self.batch_exec = batch_exec
        self.batch_args = batch_args
        self._concurrent_eval = concurrent_eval
        self._runner_exec = runner_exec
        self.runner_args = runner_args
        self._initial_time_s = _parse_time_to_s(batch_args)
        self._mem_ceiling_mb = _parse_mem_to_mb(batch_args)
        self.max_retries = max_retries

    @property
    def concurrent_eval(self) -> bool:
        return self._concurrent_eval

    def command_prefix(self, single_rank: bool = False) -> str:
        command_parts = [self._runner_exec]
        if single_rank: command_parts += ["-n 1"]
        if self.runner_args: command_parts += [self.runner_args]
        return _join(command_parts)

    def batch_prefix(self, chdir: str, log_file: str, single_rank: bool = False, time_limit_s: float | None = None, mem_mb: float | None = None) -> str:
        batch_parts = [self.batch_exec, "--wait", f"--chdir={chdir}", f"--output={log_file}"]
        batch_args = shlex.split(self.batch_args)
        if single_rank:
            _, batch_args = _sbatch_parser.parse_known_args(batch_args)
            batch_args += ["--ntasks=1", "--nodes=1"]

        if time_limit_s is not None:
            _, batch_args = _stime_parser.parse_known_args(batch_args)
            batch_args += [f"--time={_seconds_to_hms(time_limit_s)}"]

        if mem_mb is not None:
            _, batch_args = _smem_parser.parse_known_args(batch_args)
            batch_args += [f"--mem={max(1, math.ceil(mem_mb))}M"]

        batch_parts += batch_args
        return _join(batch_parts)

    def _submit_and_wait(self, submit_cmd: str) -> tuple[subprocess.CompletedProcess, str | None]:
        """Run an sbatch --wait command and return (proc, job_id), with job_id parsed from 'Submitted batch job N'."""
        proc = subprocess.run(submit_cmd, shell=True, env=_env_for_sbatch(os.environ), text=True, capture_output=True)
        job_id = next((line.rsplit(" ", 1)[-1] for line in proc.stdout.splitlines() if line.startswith("Submitted batch job")), None)
        if proc.returncode != 0 and proc.stderr: sys.stderr.write(proc.stderr)
        return proc, job_id

    def _escalate_mem(self, kind: str, request_mb: float | None, used_mb: float | None) -> bool:
        """Record the demand of an OOM-killed job and return whether the launcher can still request a larger ``--mem``.

        The demand is the larger of the request and the peak that sacct reported.
        """
        if self.memory is None: return False
        demand_mb = max(request_mb or 0.0, used_mb or 0.0)
        if demand_mb: self.memory.record(kind, demand_mb)
        next_mb = self.memory.estimate(kind)
        if next_mb is not None and (request_mb is None or next_mb > request_mb):
            at = f" at --mem={math.ceil(request_mb)}M" if request_mb else ""
            logger.warning(f"{kind} job ran out of memory{at}, retrying at --mem={math.ceil(next_mb)}M")
            return True

        if next_mb is None: logger.error(f"{kind} job ran out of memory and sacct reported no usage — cannot size a larger request")
        else: logger.error(f"{kind} job ran out of memory at --mem={math.ceil(next_mb)}M, the ceiling set by --mem in batch_args — raise it to continue")
        return False

    def _submit(self, kind: str, command: str, chdir, log_file: str, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        """Submit *command* as one job of *kind* ("train" or "eval"), wait for it and record its memory use.

        Raises:
            JobTimedOut: The job hit its time limit.
            JobOutOfMemory: The job ran out of memory, and a larger ``--mem`` remains possible.
            CalledProcessError: The job failed otherwise.
        """
        mem_mb = self.memory.estimate(kind) if self.memory else None
        submit_cmd = f'{self.batch_prefix(chdir, log_file, single_rank, time_limit_s=time_limit_s, mem_mb=mem_mb)} --wrap="{command}"'
        logger.info(f"running ({kind}): {submit_cmd}")
        proc, job_id = self._submit_and_wait(submit_cmd)

        oom, used_mb = _sacct_job(job_id) if job_id else (False, None)
        if self.memory and used_mb: self.memory.record(kind, used_mb)

        if proc.returncode != 0:
            exc = subprocess.CalledProcessError(proc.returncode, submit_cmd)
            log_path = Path(chdir) / log_file
            if _is_slurm_timeout(log_path): raise JobTimedOut(exc.returncode, exc.cmd) from exc
            if (oom or _is_slurm_oom(log_path)) and self._escalate_mem(kind, mem_mb, used_mb): raise JobOutOfMemory(exc.returncode, exc.cmd) from exc
            raise exc

    def _run_impl(self, command: str, log_file: str, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        self._submit("train", f"{self.command_prefix(single_rank)} {command}", os.getcwd(), log_file, single_rank, time_limit_s)

    def _call_evaluator_impl(self, evaluator_fn, structure, eval_dir: Path, single_rank: bool = False, time_limit_s: float | None = None):
        """Run ./evaluator.py as a job of its own. This launcher ignores *evaluator_fn*."""
        return super()._call_evaluator_impl(None, structure, eval_dir, single_rank, time_limit_s)

    def _run_evaluator(self, command: str, eval_dir: Path, single_rank: bool = False, time_limit_s: float | None = None) -> None:
        self._submit("eval", f"COMMAND_PREFIX={shlex.quote(self.command_prefix(single_rank))} {command}", eval_dir, "eval.log", single_rank, time_limit_s)
