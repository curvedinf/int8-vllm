# SPDX-License-Identifier: Apache-2.0
import pytest

from scripts import mi100_cpu_map


@pytest.fixture
def cpu_topology(monkeypatch, tmp_path):
    monkeypatch.setattr(mi100_cpu_map, "SYS_CPU", tmp_path)
    monkeypatch.setattr(mi100_cpu_map.os, "sched_getaffinity", lambda _: {0, 4, 8, 12})
    monkeypatch.setattr(mi100_cpu_map.shutil, "which", lambda _: "/usr/bin/numactl")
    for node, cpus in enumerate(((0, 8), (4, 12))):
        for cpu in cpus:
            path = tmp_path / f"cpu{cpu}"
            (path / f"node{node}").mkdir(parents=True)
            cache = path / "cache/index3"
            cache.mkdir(parents=True)
            (cache / "level").write_text("3\n")
            (cache / "shared_cpu_list").write_text(",".join(map(str, cpus)))


def test_nodes_follow_visible_gpu_order(cpu_topology, capsys):
    assert mi100_cpu_map.resolve(["3", "1"], ["4,12", "0,8"]) == [
        "--numa-bind",
        "--numa-bind-nodes",
        "1",
        "0",
        "--numa-bind-cpus",
        "4,12",
        "0,8",
    ]
    assert "rank 0 GPU 3: CPUs 4,12, NUMA 1" in capsys.readouterr().err


@pytest.mark.parametrize("groups", [["0"], ["0,4", "8"], ["0-4", "8"], ["4-0", "8"]])
def test_invalid_groups(cpu_topology, groups):
    with pytest.raises(ValueError):
        mi100_cpu_map.resolve(["0", "1"], groups)
