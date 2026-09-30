"""Train an MTP as `mlp train` does.

    python -m otf_engine.train potential.almtp train.cfg [--save_to=FILE] [--ranks=N] [--al_mode=cfg|nbh] [mlp train options]

Takes `mlp train`'s options (--iteration_limit=300, --energy_weight=1, ...). --ranks sets the
thread count, by default the CPUs this process may run on. The trained potential is written to
--save_to, by default over the input, with an active set built on the training set. The
potential must already be trained, with coefficients for every species in the training set.
"""

import logging
import sys

from .io_cfg import read_cfg
from .mtp_backend import train


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    options = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true") for a in argv if a.startswith("--"))
    if len(args) != 2: raise SystemExit(__doc__)
    potential, training_set = args

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    ranks = int(options.pop("ranks")) if "ranks" in options else None
    al_mode = options.pop("al_mode", "cfg")
    save_to = options.pop("save_to", potential)

    with open(training_set) as f:
        structures = read_cfg(f)
    train(potential, structures, save_to, settings=options, ranks=ranks, al_mode=al_mode)
    logging.getLogger(__name__).info("training complete")


if __name__ == "__main__":
    main()
