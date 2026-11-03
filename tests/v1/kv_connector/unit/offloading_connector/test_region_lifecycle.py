# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-region startup handshake and abrupt process-exit regression tests."""

import mmap
import multiprocessing
import os
import signal
import socket
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _connector(engine_id, region=None):
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector._offloading_config = SimpleNamespace(
        engine_id=engine_id,
        parallel=SimpleNamespace(tp_size=2, pp_size=1, pcp_size=1),
    )
    connector.connector_scheduler = object() if region is None else None
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    worker._mmap_region = region
    connector.connector_worker = SimpleNamespace(worker=worker)
    return connector


def _region(engine_id, rank):
    return SharedOffloadRegion(
        engine_id=engine_id,
        num_blocks=2,
        rank=rank,
        kv_bytes_per_block=2 * mmap.PAGESIZE,
        cpu_page_size=mmap.PAGESIZE,
    )


def _creator_process(engine_id, connection):
    region = _region(engine_id, 0)
    region.lock_worker_pages()
    region.mmap_obj[0:1] = b"A"
    connection.send(_connector(engine_id, region).get_handshake_metadata())
    connection.recv()  # parent deliberately SIGKILLs us before any cleanup


def test_empty_handshake_for_non_mmap_backend():
    _connector(str(uuid.uuid4())).set_xfer_handshake_metadata_pp_aware({})


def test_handshake_unlink_survives_creator_sigkill():
    engine_id = str(uuid.uuid4())
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_creator_process, args=(engine_id, child))
    process.start()
    region = None
    try:
        assert parent.poll(60), "creator did not finish mapping"
        creator_metadata = parent.recv()
        region = _region(engine_id, 1)
        worker_metadata = _connector(engine_id, region).get_handshake_metadata()
        scheduler = _connector(engine_id)
        scheduler.set_xfer_handshake_metadata_pp_aware(
            {(0, 0): creator_metadata, (0, 1): worker_metadata}
        )
        assert not os.path.exists(region.mmap_path)
        assert os.fstat(region.fd).st_nlink == 0
        process.kill()
        process.join(timeout=10)
        assert process.exitcode == -signal.SIGKILL
        assert region.mmap_obj[0:1] == b"A"
        region.mmap_obj[mmap.PAGESIZE : mmap.PAGESIZE + 1] = b"B"
        assert region.mmap_obj[mmap.PAGESIZE : mmap.PAGESIZE + 1] == b"B"
        # No directory link can retain the RAM once this last mapping closes.
        region.cleanup()
        assert not os.path.exists(region.mmap_path)
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=10)
        if region is not None:
            region.cleanup()
        path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        if os.path.exists(path):
            os.unlink(path)
        parent.close()
        child.close()


@pytest.mark.parametrize("bad", ["missing", "inode", "path", "remote"])
def test_handshake_does_not_unlink_unconfirmed_region(bad):
    engine_id = str(uuid.uuid4())
    region = _region(engine_id, 0)
    try:
        metadata = _connector(engine_id, region).get_handshake_metadata()
        workers = {(0, 0): metadata, (0, 1): metadata}
        if bad == "missing":
            workers.pop((0, 1))
        elif bad == "inode":
            workers[(0, 1)] = replace(metadata, inode=metadata.inode + 1)
        elif bad == "path":
            workers = {key: replace(metadata, path="/wrong/path") for key in workers}
        else:
            workers[(0, 1)] = replace(
                metadata, hostname=socket.gethostname() + "-other"
            )
        scheduler = _connector(engine_id)
        if bad == "remote":
            scheduler.set_xfer_handshake_metadata_pp_aware(workers)
        else:
            with pytest.raises(RuntimeError):
                scheduler.set_xfer_handshake_metadata_pp_aware(workers)
        assert os.path.exists(region.mmap_path)
    finally:
        region.cleanup()
