# SPDX-License-Identifier: Apache-2.0
"""Resolve per-GPU CPU lists for the MI100 launcher's --numa-bind arguments."""

import os
import re
import shutil
import sys
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


def _cpu_range(spec: str) -> set[int]:
    cpus: set[int] = set()
    for entry in spec.split(","):
        bounds = entry.strip().split("-")
        cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return cpus


def _compress(cpus: set[int]) -> str:
    runs: list[str] = []
    start = prev = None
    for cpu in sorted(cpus):
        if prev is not None and cpu == prev + 1:
            prev = cpu
            continue
        if prev is not None:
            runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = cpu
    if prev is not None:
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(runs)


def parent_group(groups: list[str], count: int = 8, skip: int = 0) -> str:
    """The first ``count`` physical cores (with SMT siblings) past the groups.

    The API server and EngineCore run there once the workers are bound; the
    search starts after the worker groups to stay clear of kernel housekeeping
    on the lowest cores.
    """
    used: set[int] = set()
    for group in groups:
        used |= _cpu_range(group)
    allowed = sorted(os.sched_getaffinity(0))
    cores: list[set[int]] = []
    last_physical = -1
    for cpu in allowed:
        siblings = _cpu_range(
            (SYS_CPU / f"cpu{cpu}/topology/thread_siblings_list").read_text()
        )
        first = min(siblings)
        if first in used:
            last_physical = max(last_physical, first)
        if first != cpu or siblings & used:
            continue
        cores.append(siblings)
    cores.sort(key=lambda siblings: (min(siblings) <= last_physical, min(siblings)))
    chosen: set[int] = set().union(*cores[skip : skip + count]) if cores else set()
    return _compress(chosen)


def resolve(devices: list[str], groups: list[str]) -> list[str]:
    if len(groups) != len(devices):
        raise ValueError(f"expected one CPU list for each of {len(devices)} GPUs")
    if not shutil.which("numactl"):
        raise ValueError("install numactl to use CPU binding")
    allowed = os.sched_getaffinity(0)
    nodes = []
    for rank, (device, group) in enumerate(zip(devices, groups)):
        cpus = set()
        for entry in group.split(","):
            if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", entry):
                raise ValueError(f"invalid CPU list {group!r}")
            bounds = entry.split("-")
            first, last = int(bounds[0]), int(bounds[-1])
            if first > last or last > max(allowed):
                raise ValueError(f"CPU range {entry} is outside the allowed CPUs")
            cpus.update(range(first, last + 1))
        if not cpus <= allowed:
            raise ValueError(
                f"GPU {device} includes unavailable CPUs {sorted(cpus - allowed)}"
            )
        numa = set()
        caches = set()
        for cpu in cpus:
            path = SYS_CPU / f"cpu{cpu}"
            numa.update(node.name[4:] for node in path.glob("node[0-9]*"))
            for level in path.glob("cache/index*/level"):
                if level.read_text().strip() == "3":
                    caches.add((level.parent / "shared_cpu_list").read_text().strip())
        if len(numa) != 1:
            raise ValueError(f"CPU list {group} must resolve to one NUMA node")
        node = numa.pop()
        nodes.append(node)
        print(
            f"MI100 rank {rank} GPU {device}: CPUs {group}, NUMA {node}, "
            f"L3 shared CPU lists {' / '.join(sorted(caches)) or 'unknown'}",
            file=sys.stderr,
        )
    return ["--numa-bind", "--numa-bind-nodes", *nodes, "--numa-bind-cpus", *groups]


if __name__ == "__main__":
    try:
        if sys.argv[1:] and sys.argv[1] == "--parent":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
            skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            print(parent_group(os.environ["MI100_CPU_MAP"].split(), count, skip))
            raise SystemExit(0)
        devices = os.environ.get("MI100_DEVICES", "0,1,2,3").split(",")
        ranks = int(os.environ.get("TPS", "4")) * int(
            os.environ.get("MI100_PP_SIZE", "1")
        )
        if len(devices) < ranks:
            raise ValueError(f"expected {ranks} visible devices, found {len(devices)}")
        print(*resolve(devices[:ranks], os.environ["MI100_CPU_MAP"].split()))
    except (ValueError, OSError) as exc:
        sys.exit(f"MI100_CPU_MAP: {exc}")
