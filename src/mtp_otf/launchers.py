"""Execution backends for otf-mtp: mpirun (nested), fork (direct), and slurm (sbatch --wait)."""
from __future__ import annotations

import os
import subprocess
import threading
import concurrent.futures
from abc import ABC, abstractmethod

# os.chdir is process-wide; this lock serialises chdir+call+restore sequences
# so concurrent threads don't corrupt each other's working directory.
# SlurmLauncher overrides call_evaluator without os.chdir, so it is unaffected
# and truly parallel.
_CHDIR_LOCK = threading.Lock()

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
    def run(self, command: str, log_path: str, env: dict, n_procs: int | None = 1) -> None:
        """Execute *command* (f-string of ``mlp binary + subcommand + args``,
        without any mpirun/srun prefix).

        Parameters
        ----------
        command:  Command string passed to the shell, e.g.
                  ``f"{mlp} calculate_grade {potential} ..."``
        log_path: Append combined stdout/stderr here.
        env:      Environment mapping for the subprocess.
        n_procs:  Number of MPI tasks.  ``None`` means "use all available"
                  (launcher-dependent — mpirun omits ``-n``).
        """

    def call_evaluator(self, evaluator_fn, structure, eval_dir: str, env: dict):
        """Evaluate *structure* via *evaluator_fn* inside *eval_dir*.

        The default implementation changes into *eval_dir*, calls
        ``evaluator_fn(structure)``, and returns the result.  Override for
        env-cleaned (fork) or remote-dispatch (slurm) behaviour.

        Note: uses a process-wide lock around os.chdir to be thread-safe when
        parallel_eval=True is used.  Evaluations are serialised for this base
        implementation; SlurmLauncher overrides without the lock.
        """
        os.makedirs(eval_dir, exist_ok=True)
        with _CHDIR_LOCK:
            prev = os.getcwd()
            os.chdir(eval_dir)
            try:
                return evaluator_fn(structure)
            finally:
                os.chdir(prev)

    @property
    def parallel_eval(self) -> bool:
        """Whether eval_structures should submit evaluations in parallel."""
        return False


# ---------------------------------------------------------------------------
# NestedMPILauncher  (current behaviour, default)
# ---------------------------------------------------------------------------


class NestedMPILauncher(Launcher):
    """Wrap mlp calls with ``mpirun``.  Matches existing behaviour exactly."""

    def __init__(self, mpirun_executable: str = "mpirun", mpirun_args: str = ""):
        self.mpirun_executable = mpirun_executable
        self.mpirun_args = mpirun_args

    def run(self, command: str, log_path: str, env: dict, n_procs: int | None = 1) -> None:
        n_part = f"-n {n_procs} " if n_procs is not None else ""
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

    def __init__(self, fork_args: str = ""):
        self.fork_args = fork_args

    def run(self, command: str, log_path: str, env: dict, n_procs: int | None = 1) -> None:
        if n_procs is None:
            n_procs = _default_n_procs()
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

    def call_evaluator(self, evaluator_fn, structure, eval_dir: str, env: dict):
        """Run evaluator in *eval_dir* with MPI env rebuilt for a fresh universe.

        The evaluator's profile command should be set to just the binary
        (e.g. ``'pw.x'``), not ``'mpiexec -n N pw.x'``.  The evaluator uses
        all available processors in a fresh MPI universe.

        Note: uses a process-wide lock around os.chdir, so evaluations are
        serialised here.  Only SlurmLauncher achieves true parallelism.
        """
        os.makedirs(eval_dir, exist_ok=True)
        with _CHDIR_LOCK:
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
    def _build_submit_cmd(self, cmd: str, log_path: str, n_procs: int | None) -> str:
        """Return the full shell command that submits *cmd* and waits."""

    def run(self, command: str, log_path: str, env: dict, n_procs: int | None = 1) -> None:
        submit_cmd = self._build_submit_cmd(command, log_path, n_procs)
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

    Structure evaluations are submitted in parallel by default
    (``parallel_eval=True``): all sbatch jobs are submitted simultaneously
    from separate threads, each blocking on ``--wait``.

    Parameters
    ----------
    sbatch_executable: Path to sbatch binary (default: ``"sbatch"``).
    sbatch_args:       Extra sbatch options, e.g.
                       ``["--partition=gpu", "--time=01:00:00"]``.
    parallel_eval:     Submit structure evaluations concurrently (default: True).
    """

    def __init__(self, sbatch_executable: str = "sbatch", sbatch_args: str = "", parallel_eval: bool = True):
        self.sbatch_executable = sbatch_executable
        self.sbatch_args = sbatch_args
        self._parallel_eval = parallel_eval

    @property
    def parallel_eval(self) -> bool:
        return self._parallel_eval

    def _build_submit_cmd(self, cmd: str, log_path: str, _n_procs: int | None) -> str:
        abs_log = os.path.abspath(log_path)
        cwd = os.getcwd()
        extra = self.sbatch_args
        extra_part = f"{extra} " if extra else ""
        return f"{self.sbatch_executable} --wait --chdir={cwd} --output={abs_log} --open-mode=append {extra_part}--wrap=\"{cmd}\""

    def call_evaluator(self, evaluator_fn, structure, eval_dir: str, env: dict):
        """Submit the evaluator as a Slurm batch job.

        Writes the input structure to *eval_dir*, generates a small Python
        wrapper that imports ``evaluator.py`` from the original working
        directory, runs it, and writes the result.  Submits via
        ``sbatch --wait`` and reads the result back when done.

        Note: *evaluator_fn* is used only to check for custom imports; the
        actual code dispatched to the batch job always loads ``evaluator.py``
        from the calling working directory.  For programmatic use with a
        non-file evaluator, use ``NestedMPILauncher`` or ``ForkLauncher``.
        """
        import ase.io

        os.makedirs(eval_dir, exist_ok=True)
        eval_dir_abs = os.path.abspath(eval_dir)
        orig_dir = os.getcwd()
        evaluator_py = os.path.join(orig_dir, "evaluator.py")

        input_path = os.path.join(eval_dir_abs, "input_structure.traj")
        output_path = os.path.join(eval_dir_abs, "output_structure.traj")
        eval_log = os.path.join(eval_dir_abs, "eval.log")
        wrapper_path = os.path.join(eval_dir_abs, "_run_eval.py")

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

        extra = self.sbatch_args
        extra_part = f"{extra} " if extra else ""
        sbatch_cmd = f"{self.sbatch_executable} --wait --chdir={eval_dir_abs} --output={eval_log} --open-mode=append {extra_part}--wrap=\"python {wrapper_path}\""
        print(f"running (eval): {sbatch_cmd}")
        subprocess.run(sbatch_cmd, shell=True, env=env, check=True)

        return ase.io.read(output_path)
