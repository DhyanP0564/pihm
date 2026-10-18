"""Run a campaign, or one run of it, and collect its summary.

Usage, with pihm installed, from the repository root::

    python examples/run_campaign.py examples/campaigns/p2_reproduction.yaml --out results/p2

Each campaign file in ``examples/campaigns/`` is one set of runs, and ``examples/campaigns.yaml``
lists where each campaign's records go. Together they regenerate every number reported with the
method. Runs whose ``ok`` record already exists are skipped (``--rerun`` replaces them), so an
interrupted campaign resumes where it stopped. ``--prune`` first removes the records the campaign
no longer contains.

``--count`` prints the number of runs, ``--list`` numbers them, and ``--task i`` runs only the
i-th (from 0), so a cluster job array can run one task per run. ``--classes`` prints each
allocation the runs ask for (``memory`` and ``hours`` in the campaign's groups) with its task
numbers, one line each, ``<GB> <hours> <tasks>``. A task writes its own record and leaves the
summary to ``--collect``, which rebuilds it from every record present and then prints what
:func:`pihm.pipeline.audit.audit` finds: refusals whose timed step is suspect, tasks ended at
their time limit, out of memory or by a crash without an ``ok`` record, runs with no record, and
records made by a different version of the source.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pihm.pipeline.audit import audit
from pihm.pipeline.campaign import classes, load_campaign, prune, run_campaign
from pihm.pipeline.results import Results
from pihm.pipeline.spec import RunSpec
from pihm.pipeline.summary import collect

ROOT = Path(__file__).resolve().parents[1]


def _describe(spec: RunSpec) -> str:
    """Return a run as ``--list`` names it: problem, resolution, order, form, stage, tier, label."""
    n = "x".join(str(k) for k in spec.n)
    r = " r=" + ",".join(f"{v:g}" for v in spec.geometric_r)
    k = "" if spec.carleman is None else f" K={spec.carleman}"
    line = f"{spec.problem.key} n={n}{k} {spec.regime} {spec.stage} {spec.tier}{r}"
    return f"{line}  {spec.label}".rstrip()


def _ranges(tasks: list[int]) -> str:
    """Return task numbers as Slurm writes an array, runs of consecutive ones as ranges.

    >>> _ranges([0, 1, 2, 5, 7, 8])
    '0-2,5,7-8'
    """
    runs: list[list[int]] = []
    for task in tasks:
        if runs and task == runs[-1][1] + 1:
            runs[-1][1] = task
        else:
            runs.append([task, task])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


def main(argv: list[str] | None = None) -> int:
    """Run the campaign named on the command line; return the number of failed runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", type=Path, help="the campaign's YAML file")
    parser.add_argument("--out", type=Path, help="the results directory")
    parser.add_argument("--rerun", action="store_true", help="replace existing records")
    parser.add_argument(
        "--prune", action="store_true", help="remove records the campaign no longer contains"
    )
    parser.add_argument("--count", action="store_true", help="print the number of runs and stop")
    parser.add_argument("--list", action="store_true", help="number the runs and stop")
    parser.add_argument(
        "--classes", action="store_true", help="print each allocation's task numbers and stop"
    )
    parser.add_argument("--task", type=int, help="run only this run (0-based), without a summary")
    parser.add_argument(
        "--collect", action="store_true", help="only rebuild the summary from the records"
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logging.getLogger("pihm").setLevel(logging.INFO)
    campaign = load_campaign(arguments.campaign)
    if arguments.count:
        print(len(campaign.specs))
        return 0
    if arguments.classes:
        for allocation, tasks in classes(campaign).items():
            print(f"{allocation.memory} {allocation.hours} {_ranges(tasks)}")
        return 0
    if arguments.list:
        for index, spec in enumerate(campaign.specs):
            print(f"{index:3d}  {_describe(spec)}")
        return 0
    if arguments.out is None:
        parser.error("--out is required to run or collect a campaign")
    if arguments.collect:
        print(collect(arguments.out), flush=True)
        expected = [(spec.config_hash, _describe(spec)) for spec in campaign.specs]
        accepted = Results.at(ROOT).accepted_sources()
        for finding in audit(arguments.out, expected, accepted):
            print(finding, flush=True)
        return 0
    if arguments.task is not None:
        spec = campaign.specs[arguments.task]
        print(f"{campaign.name}: run {arguments.task} of {len(campaign.specs)}", flush=True)
        run_campaign([spec], arguments.out, rerun=arguments.rerun)
        return 0
    print(f"{campaign.name}: {len(campaign.specs)} runs -> {arguments.out}", flush=True)
    started = time.perf_counter()
    if arguments.prune:
        removed = prune(campaign.specs, arguments.out)
        print(f"pruned {len(removed)} stale records", flush=True)
    run_campaign(campaign.specs, arguments.out, rerun=arguments.rerun)
    table = collect(arguments.out)
    failed = sum(1 for line in table.read_text().splitlines()[1:] if ",failed," in line)
    print(f"done in {time.perf_counter() - started:.0f} s; {failed} failed; {table}", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
