# SPDX-License-Identifier: Apache-2.0
"""Exercise supervisor recognition with the restricted systemd PATH."""

import os
import shlex
import signal
import subprocess
import threading
from pathlib import Path

import pytest


@pytest.mark.parametrize("supervisor", [True, False])
def test_stop_matches_only_bolt_supervisor_without_ripgrep(tmp_path, supervisor):
    launcher = Path(__file__).resolve().parents[1] / "scripts/serve_recipe_bolt.sh"
    # Supply supervisor argv without loading a model. Reap in a separate
    # thread so the launcher's kill -0 loop sees the process actually exit.
    process = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            "sleep 60 & wait",
            str(launcher) if supervisor else "unrelated.sh",
            "supervise",
        ],
        start_new_session=True,
    )
    waiter = threading.Thread(target=process.wait, daemon=True)
    waiter.start()
    try:
        (tmp_path / "supervisor.pid").write_text(str(process.pid))
        result = subprocess.run(
            ["/bin/bash", str(launcher), "stop"],
            env={**os.environ, "PATH": "/usr/bin:/bin", "LOG_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "command not found" not in result.stderr
        if supervisor:
            waiter.join(timeout=5)
            assert process.returncode == -signal.SIGTERM
        else:
            assert process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        waiter.join(timeout=5)


def test_systemd_prestart_reclaims_stale_offload_without_removing_other_ipc(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/serve_recipe_bolt.sh"
    shm = tmp_path / "shm"
    shm.mkdir()
    stale = shm / "vllm_offload_abandoned.mmap"
    stale.write_bytes(b"old cache")
    unrelated = shm / "psm_unrelated"
    unrelated.write_bytes(b"another application")
    environment = {
        **os.environ,
        "PATH": "/usr/bin:/bin",
        "LOG_DIR": str(tmp_path / "logs"),
        "BOLT_SHM_DIR": str(shm),
    }
    # Execute the unit's actual pre-start sequence without starting a model.
    for line in (root / "scripts/vllm-bolt.service").read_text().splitlines():
        if line.startswith("ExecStartPre="):
            command = shlex.split(line.split("=", 1)[1])
            assert Path(command[-2]).name == launcher.name
            result = subprocess.run(
                ["/bin/bash", str(launcher), command[-1]],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, result.stderr
    assert not stale.exists()
    assert unrelated.read_bytes() == b"another application"


def test_cleanup_refuses_an_offload_file_still_in_use(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "scripts/serve_recipe_bolt.sh"
    active = tmp_path / "vllm_offload_active.mmap"
    active.write_bytes(b"live cache")
    with active.open("rb"):
        result = subprocess.run(
            ["/bin/bash", str(launcher), "cleanup"],
            env={
                **os.environ,
                "PATH": "/usr/bin:/bin",
                "LOG_DIR": str(tmp_path / "logs"),
                "BOLT_SHM_DIR": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode != 0
    assert "still in use" in result.stderr
    assert active.read_bytes() == b"live cache"
