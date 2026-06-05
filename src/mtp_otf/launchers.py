"""Execution backends for otf-mtp: mpirun (nested), fork (direct), and slurm (sbatch --wait)."""
from __future__ import annotations

import os
import sys
import subprocess
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------------

# Variables that prevent nested mpirun from working (parent's MPI job leaks in).
# These three are the set currently removed by __main__.py and are known to work.
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


def _env_for_mpirun(env: dict) -> dict:
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


def _default_n_procs() -> int:
    """Best-effort process count: Slurm allocation first, then cpu_count."""
    for var in ("SLURM_NTASKS", "SLURM_NPROCS"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return os.cpu_count() or 1


def _run_cmd(cmd: str, log_path: str, env: dict) -> None:
    """Run a shell command string, appending combined output to log_path."""
    print(f"running: {cmd}")
    with open(log_path, "a") as f:
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
    def run(self, command: str, log_path: str, env: dict, parallel_eval: bool = True) -> None:
        """Execute *command* (f-string of ``mlp binary + subcommand + args``,
        without any mpirun/srun prefix).

        Parameters
        ----------
        command:  Command string passed to the shell, e.g.
                  ``f"{mlp} calculate_grade {potential} ..."``
        log_path: Append combined stdout/stderr here.
        env:      Environment mapping for the subprocess.
        parallel_eval:  When true, use all available processes.  When false,
                        use a single process.
        """

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path, env: dict):
        """Evaluate *structure* inside *eval_dir* and return the result.

        Changes into *eval_dir*, calls ``evaluator_fn(structure)``, and restores
        cwd.  Safe because local launchers set concurrent_eval=False, so only
        one thread calls this at a time.
        """
        eval_dir.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        os.chdir(eval_dir)
        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)

    @property
    def parallel_eval(self) -> bool:
        """Whether run() should use all available MPI processes (True) or 1 (False)."""
        return True

    @property
    def concurrent_eval(self) -> bool:
        """Whether eval_structures should submit evaluations concurrently via ThreadPoolExecutor."""
        return False


# ---------------------------------------------------------------------------
# NestedMPILauncher  (current behaviour, default)
# ---------------------------------------------------------------------------


class NestedMPILauncher(Launcher):
    """Wrap mlp calls with ``mpirun -n <nprocs>``."""

    def __init__(self, mpirun_executable: str = "mpirun", mpirun_args: str = "", parallel_eval: bool = True):
        self.mpirun_executable = mpirun_executable
        self.mpirun_args = mpirun_args
        self._parallel_eval = parallel_eval

    @property
    def parallel_eval(self) -> bool:
        return self._parallel_eval

    def run(self, command: str, log_path: str, env: dict, parallel_eval: bool = True) -> None:
        n_procs = _default_n_procs() if parallel_eval else 1
        n_part = f"-n {n_procs} "
        extra = f"{self.mpirun_args} " if self.mpirun_args else ""
        cmd = f"{self.mpirun_executable} {n_part}{extra}{command}"
        _run_cmd(cmd, log_path, _env_for_mpirun(env))


# ---------------------------------------------------------------------------
# ForkLauncher  (direct exec, no mpirun wrapper)
# ---------------------------------------------------------------------------


class ForkLauncher(Launcher):
    """Run binaries directly without mpirun, constructing a fresh MPI universe
    via environment variables.

    Extra launcher arguments are accepted for interface consistency with the
    other launchers, but fork mode does not use them.

    For the evaluator, configure the DFT profile with just the binary path
    (e.g. ``command='pw.x'``) rather than ``'mpiexec -n 4 pw.x'``.
    The number of peer processes is taken from the current Slurm allocation
    when available, otherwise from ``os.cpu_count()``.
    """

    def __init__(self, fork_args: str = "", parallel_eval: bool = True):
        self.fork_args = fork_args
        self._parallel_eval = parallel_eval

    @property
    def parallel_eval(self) -> bool:
        return self._parallel_eval

    def run(self, command: str, log_path: str, env: dict, parallel_eval: bool = True) -> None:
        n_procs = _default_n_procs() if parallel_eval else 1
        child_env = _env_for_fork(n_procs, env)
        with open(log_path, "a") as log_f:
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

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path, env: dict):
        """Run evaluator in *eval_dir* with a fresh MPI universe environment.

        Serialised (concurrent_eval=False), so os.chdir + os.environ mutation is safe.
        """
        eval_dir.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        os.chdir(eval_dir)
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(_env_for_fork(_default_n_procs(), env))
        try:
            return evaluator_fn(structure)
        finally:
            os.chdir(prev)
            os.environ.clear()
            os.environ.update(saved)


# ---------------------------------------------------------------------------
# BatchSubmitLauncher ABC  (extensible job-manager base)
# ---------------------------------------------------------------------------


class BatchSubmitLauncher(Launcher, ABC):
    """Base for launchers that submit batch jobs and wait for completion.

    Subclasses implement ``_build_submit_cmd`` to produce the full shell
    command that submits *cmd* and blocks until the job finishes.

    To add a new job manager (PBS, LSF, …):
    1. Subclass ``BatchSubmitLauncher``.
    2. Override ``_build_submit_cmd`` to produce the scheduler-specific
       submission command (``qsub -W block=true``, ``jsrun``, …).
    3. Export and register in ``__main__.py``.
    """

    @abstractmethod
    def _build_submit_cmd(self, cmd: str, log_path: str) -> str:
        """Return the full shell command that submits *cmd* and waits."""

    def run(self, command: str, log_path: str, env: dict, _parallel_eval: bool = True) -> None:
        submit_cmd = self._build_submit_cmd(command, log_path)
        print(f"running: {submit_cmd}")
        subprocess.run(submit_cmd, shell=True, env=env, check=True)


# ---------------------------------------------------------------------------
# SlurmLauncher  (sbatch --wait with --wrap)
# ---------------------------------------------------------------------------


class SlurmLauncher(BatchSubmitLauncher):
    """Submit mlp calls and evaluator jobs via ``sbatch --wait --wrap="cmd"``.

    Each ``launcher.run()`` call submits one Slurm batch job and blocks until
    it finishes (``--wait``).  The command is passed inline via ``--wrap``
    (requires Slurm ≥ 2.6), avoiding temp script files.

    Structure evaluations are submitted concurrently by default
    (``concurrent_eval=True``): all sbatch jobs are submitted simultaneously
    from separate threads, each blocking on ``--wait``.  MPI parallelism is
    controlled by the job's resource allocation via ``sbatch_args``.

    **Python environment requirement**: the Python interpreter used to launch
    this process (``sys.executable``) is embedded directly in the evaluator
    batch script.  That interpreter — and every package it depends on (``ase``, etc.)
    — must reside on a filesystem that is accessible from
    all compute nodes (e.g. a shared NFS/Lustre mount or e.g. /home or /project).  A Python environment
    built on a local disk of the submission node will not be reachable by the
    batch job and will cause an import failure at runtime.

    Parameters
    ----------
    sbatch_executable: Path to sbatch binary (default: ``"sbatch"``).
    sbatch_args:       Extra sbatch options as a single shell string, e.g.
                       ``"--partition=gpu --time=01:00:00"``.
    concurrent_eval:   Submit structure evaluations concurrently (default: True).
    """

    def __init__(self, sbatch_executable: str = "sbatch", sbatch_args: str = "", concurrent_eval: bool = True):
        self.sbatch_executable = sbatch_executable
        self.sbatch_args = sbatch_args
        self._concurrent_eval = concurrent_eval

    @property
    def concurrent_eval(self) -> bool:
        return self._concurrent_eval

    def _build_submit_cmd(self, cmd: str, log_path: str) -> str:
        abs_log = os.path.abspath(log_path)
        cwd = os.getcwd()
        extra = self.sbatch_args
        extra_part = f"{extra} " if extra else ""
        return f"{self.sbatch_executable} --wait --chdir={cwd} --output={abs_log} --open-mode=append {extra_part}--wrap=\"{cmd}\""

    def call_evaluator(self, evaluator_fn, structure, eval_dir: Path, env: dict):
        """Submit the evaluator as a Slurm batch job and return the result.

        Writes the input structure to *eval_dir*, generates a Python wrapper
        that loads ``evaluator.py`` from the original working directory, submits
        it via ``sbatch --wait``, and reads the result back.  Always uses
        absolute paths — no os.chdir needed.
        """
        import ase.io

        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_dir_abs = eval_dir.resolve()
        orig_dir = os.getcwd()
        evaluator_py = os.path.join(orig_dir, "evaluator.py")

        input_path = str(eval_dir_abs / "input_structure.traj")
        output_path = str(eval_dir_abs / "output_structure.traj")
        eval_log = str(eval_dir_abs / "eval.log")
        wrapper_path = str(eval_dir_abs / "_run_eval.py")

        ase.io.write(input_path, structure)

        wrapper = (f"import sys; sys.path.insert(0, {repr(orig_dir)})\n"
                   f"import importlib.util, ase.io\n"
                   f"_spec = importlib.util.spec_from_file_location('evaluator', {repr(evaluator_py)})\n"
                   f"_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)\n"
                   f"_s = ase.io.read({repr(input_path)})\n"
                   f"_r = _mod.evaluator(_s)\n"
                   f"ase.io.write({repr(output_path)}, _r)\n")
        with open(wrapper_path, "w") as wf:
            wf.write(wrapper)

        extra_part = f"{self.sbatch_args} " if self.sbatch_args else ""
        sbatch_cmd = f"{self.sbatch_executable} --wait --chdir={eval_dir_abs} --output={eval_log} --open-mode=append {extra_part}--wrap=\"{sys.executable} {wrapper_path}\""
        print(f"running (eval): {sbatch_cmd}")
        subprocess.run(sbatch_cmd, shell=True, env=env, check=True)

        return ase.io.read(output_path)
