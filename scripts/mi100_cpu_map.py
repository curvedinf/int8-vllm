# SPDX-License-Identifier: Apache-2.0
"""Resolve per-GPU CPU lists for the MI100 launcher's --numa-bind arguments."""

import os
import re
import shutil
import sys
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


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
        devices = os.environ.get("MI100_DEVICES", "0,1,2,3").split(",")
        ranks = int(os.environ.get("TPS", "4")) * int(
            os.environ.get("MI100_PP_SIZE", "1")
        )
        if len(devices) < ranks:
            raise ValueError(f"expected {ranks} visible devices, found {len(devices)}")
        print(*resolve(devices[:ranks], os.environ["MI100_CPU_MAP"].split()))
    except (ValueError, OSError) as exc:
        sys.exit(f"MI100_CPU_MAP: {exc}")
