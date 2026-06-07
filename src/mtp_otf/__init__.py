from .otf_mtp import main
from .launchers import (
    Launcher,
    NestedMPILauncher,
    ForkLauncher,
    SlurmLauncher,
)

__all__ = [
    "main",
    "Launcher",
    "NestedMPILauncher",
    "ForkLauncher",
    "SlurmLauncher",
]

try:
    from .mtp_backend import (
        calculate_grade,
        select_add,
        train as train_mtp)
    __all__ += ["calculate_grade", "select_add", "train_mtp"]
except ImportError:
    pass
