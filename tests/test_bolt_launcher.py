# SPDX-License-Identifier: Apache-2.0
"""Exercise supervisor recognition with the restricted systemd PATH."""

import os
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
