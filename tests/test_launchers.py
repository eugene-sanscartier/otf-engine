"""SlurmLauncher's sizing of each job's --time and --mem, with sbatch and sacct stubbed, and the Slurm parsers."""

import re
import subprocess

import pytest

from otf_engine import launchers
from otf_engine.launchers import JobOutOfMemory, JobTimedOut, MemoryState, SlurmLauncher, TimingState


@pytest.mark.parametrize("text, seconds", [("30", 1800), ("30:15", 1815), ("2:30:15", 9015), ("1-2", 93600), ("1-2:30", 95400), ("1-2:30:15", 95415), ("soon", None)])
def test_hms_to_seconds(text, seconds):
    assert launchers._hms_to_seconds(text) == seconds


@pytest.mark.parametrize("text, mb", [("512M", 512), ("2G", 2048), ("1T", 1024 ** 2), ("58120K", 58120 / 1024), ("100", 100), ("", None), ("lots", None)])
def test_mem_to_mb(text, mb):
    assert launchers._mem_to_mb(text) == mb


@pytest.mark.parametrize("batch_args, ceiling", [("--mem=4G --time=1:00:00", 4096), ("--mem=0", None), ("--mem-per-cpu=2G", None), ("", None)])
def test_mem_ceiling(batch_args, ceiling):
    """--mem sets the ceiling; --mem=0 (the whole node) and --mem-per-cpu set none."""
    assert launchers._parse_mem_to_mb(batch_args) == ceiling


def test_batch_prefix():
    """The job's --time and --mem replace batch_args', --mem drops --mem-per-cpu, and a single-rank job drops the task and node counts."""
    launcher = SlurmLauncher(batch_args="--account=a --time=2:00:00 --mem-per-cpu=1G -n 8 --nodes=2")
    prefix = launcher.batch_prefix("/work", "train.log", single_rank=True, time_limit_s=3725, mem_mb=1500.2)
    assert prefix.split() == ["sbatch", "--wait", "--chdir=/work", "--output=train.log", "--account=a", "--ntasks=1", "--nodes=1", "--time=01:02:05", "--mem=1501M"]


def test_timing_estimates():
    """--time seeds the estimate and caps nothing; the first 3 trainings give their maximum, later ones their mean, which can
    fall; a timeout counts as twice the limit it died at."""
    timing = TimingState({}, initial_time_s=100.0)
    assert timing.estimate_train(10) == 100.0
    for elapsed in (200.0, 50.0):
        timing.record_train(elapsed, 10, False, 100.0)
    assert timing.estimate_train(10) == 200.0 * TimingState.safety
    for elapsed in (50.0, 50.0, 50.0):
        timing.record_train(elapsed, 10, False, 100.0)
    assert timing.estimate_train(10) == pytest.approx(80.0 * TimingState.safety)

    timing.record_eval(30.0, True, 40.0)
    assert timing.estimate_eval() == 80.0 * TimingState.safety


def test_timing_grows_with_size():
    """Trainings on growing sets give a larger estimate for a larger set."""
    timing = TimingState({})
    for size, elapsed in ((10, 100.0), (20, 200.0), (30, 300.0), (40, 400.0)):
        timing.record_train(elapsed, size, False, None)
    assert timing.estimate_train(60) > timing.estimate_train(40) > timing.estimate_train(20)


def test_memory_estimates():
    """The first job requests the ceiling; later ones 1.5 times the largest use, under the ceiling; the estimate never falls."""
    memory = MemoryState({}, ceiling_mb=4000.0)
    assert memory.estimate("train") == 4000.0
    memory.record("train", 1000.0)
    assert memory.estimate("train") == 1500.0
    memory.record("train", 500.0)
    assert memory.estimate("train") == 1500.0
    memory.record("train", 3000.0)
    assert memory.estimate("train") == 4000.0
    assert MemoryState({}).estimate("eval") is None


class StubSlurm:
    """sbatch and sacct stubbed: each submission takes the next (exit code, sacct OOM, sacct usage, log text) outcome."""

    def __init__(self, monkeypatch, launcher, outcomes):
        self.outcomes = list(outcomes)
        self.submitted = []
        self.log = ""
        monkeypatch.setattr(launcher, "_submit_and_wait", self.submit)
        monkeypatch.setattr(launchers, "_sacct_job", lambda job_id: self.sacct)
        monkeypatch.setattr(launchers, "_is_slurm_timeout", lambda path: "TIME LIMIT" in self.log)
        monkeypatch.setattr(launchers, "_is_slurm_oom", lambda path: "oom_kill" in self.log)

    def submit(self, submit_cmd):
        self.submitted += [submit_cmd]
        code, oom, used, self.log = self.outcomes.pop(0)
        self.sacct = (oom, used)
        return subprocess.CompletedProcess(submit_cmd, code, "Submitted batch job 1\n", ""), "1"

    def mem_requests(self):
        return [int(m.group(1)) if (m := re.search(r"--mem=(\d+)M", cmd)) else None for cmd in self.submitted]


def configured(batch_args, state=None):
    launcher = SlurmLauncher(batch_args=batch_args, max_retries=3)
    state = {} if state is None else state
    launcher.configure_timing(state, lambda s: None)
    launcher.configure_memory(state, lambda s: None)
    return launcher, state


def test_memory_sizing(monkeypatch, tmp_path):
    """The first training requests the ceiling, the next 1.5 times the usage sacct reported."""
    launcher, state = configured("--mem=8G")
    stub = StubSlurm(monkeypatch, launcher, [(0, False, 2000.0, ""), (0, False, 1000.0, "")])
    monkeypatch.chdir(tmp_path)
    launcher.run("mlp train", "train.log")
    launcher.run("mlp train", "train.log")
    assert stub.mem_requests() == [8192, 3000]
    assert state["memory"]["train_max_mb"] == 2000.0


def test_oom_retries_above_request(monkeypatch, tmp_path):
    """An OOM records the request the job died at as a floor, even when sacct reports less or nothing, retries above it,
    and writes no timing observation."""
    launcher, state = configured("--mem=8G", {"memory": {"train_max_mb": 1000.0}})
    stub = StubSlurm(monkeypatch, launcher, [(1, True, None, "oom_kill"), (1, False, 500.0, "oom_kill"), (0, False, 2000.0, "")])
    monkeypatch.chdir(tmp_path)
    launcher.run("mlp train", "train.log")
    assert stub.mem_requests() == [1500, 2250, 3375]
    assert len(state["timing"]["train"]["observations"]) == 1


def test_oom_at_ceiling_fails(monkeypatch, tmp_path):
    """At the ceiling an OOM fails the job instead of retrying at the same request."""
    launcher, _ = configured("--mem=2G", {"memory": {"train_max_mb": 2000.0}})
    stub = StubSlurm(monkeypatch, launcher, [(1, True, 2048.0, "oom_kill")])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        launcher.run("mlp train", "train.log")
    assert not isinstance(raised.value, JobOutOfMemory)
    assert stub.mem_requests() == [2048]


def test_empty_sacct_records_nothing(monkeypatch, tmp_path):
    """A job sacct has no usage for yet leaves the memory estimate as it was, never at zero."""
    launcher, state = configured("--mem=8G", {"memory": {"train_max_mb": 1000.0}})
    StubSlurm(monkeypatch, launcher, [(0, False, None, "")])
    monkeypatch.chdir(tmp_path)
    launcher.run("mlp train", "train.log")
    assert state["memory"]["train_max_mb"] == 1000.0


def test_timeout_retries_longer(monkeypatch, tmp_path):
    """A training that times out is recorded at twice its limit and resubmitted with a longer one."""
    launcher, state = configured("--time=1:00:00")
    stub = StubSlurm(monkeypatch, launcher, [(1, False, None, "CANCELLED DUE TO TIME LIMIT"), (0, False, None, "")])
    monkeypatch.chdir(tmp_path)
    launcher.run("mlp train", "train.log", training_set_size=10)
    times = [launchers._hms_to_seconds(re.search(r"--time=(\S+)", cmd).group(1)) for cmd in stub.submitted]
    assert times[0] == 3600 and times[1] == 7200 * TimingState.safety
    assert state["timing"]["train"]["observations"][0]["timed_out"]


def test_eval_timeout_floor(monkeypatch, tmp_path):
    """A structure that timed out at a limit is evaluated with at least twice that limit."""
    launcher, _ = configured("--time=0:10:00")
    seen = {}

    def evaluate(evaluator_fn, structure, eval_dir, single_rank=False, time_limit_s=None):
        seen["limit"] = time_limit_s
        return structure

    monkeypatch.setattr(launcher, "_call_evaluator_impl", evaluate)
    launcher.call_evaluator(None, "structure", tmp_path, timed_out_at_s=1000.0)
    assert seen["limit"] == 2000.0


def test_eval_timeout_raises(monkeypatch, tmp_path, structures):
    """An evaluation that times out raises JobTimedOut with the limit it died at, for its cycle to defer it."""
    launcher, state = configured("--time=0:10:00")
    StubSlurm(monkeypatch, launcher, [(1, False, None, "CANCELLED DUE TO TIME LIMIT")])
    with pytest.raises(JobTimedOut) as raised:
        launcher.call_evaluator(None, structures[1], tmp_path / "eval")
    assert raised.value.time_limit_s == 600.0
    assert state["timing"]["eval"]["observations"][0]["timed_out"]
