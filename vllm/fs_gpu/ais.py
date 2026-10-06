# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings for libhipfile (AMD Infinity Storage, AIS).

Only the synchronous IO surface is bound here; the async stream API and the
batch API can be added when a consumer needs them (see csrc-era notes in
docs/recipes). The library is loaded lazily and exactly once per process via
:get_ais(). All calls release the GIL (ctypes default for CDLL), so sync reads
can be overlapped from Python threads.

Runtime coexistence: libhipfile.so links ``libamdhip64.so.7`` — the same
soname torch bundles — so in a process that imported torch first the loader
resolves AIS's HIP calls onto torch's (patched) runtime instead of
initializing a second one. :func:`verify_single_hip_runtime` asserts that.
"""

from __future__ import annotations

import ctypes
import os

HIPFILE_BASE_ERR = 5000

HIPFILE_HANDLE_TYPE_OPAQUE_FD = 1


class HipFileError_t(ctypes.Structure):
    """hipFileError_t { hipFileOpError_t err; hipError_t hip_drv_err; }"""

    _fields_ = [
        ("err", ctypes.c_int),
        ("hip_drv_err", ctypes.c_int),
    ]


class HipFileDescr_t(ctypes.Structure):
    """hipFileDescr_t for a POSIX fd (hipFileHandleTypeOpaqueFD)."""

    class _Handle(ctypes.Union):
        _fields_ = [
            ("fd", ctypes.c_int),
            ("hFile", ctypes.c_void_p),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _Handle),
        ("fs_ops", ctypes.c_void_p),
    ]

    @classmethod
    def from_fd(cls, fd: int) -> HipFileDescr_t:
        d = cls()
        d.type = HIPFILE_HANDLE_TYPE_OPAQUE_FD
        d.handle.fd = fd
        d.fs_ops = None
        return d


class FsGpuError(RuntimeError):
    """Raised when an AIS operation fails.

    ``op_error`` is the positive hipFileOpError_t value (0 for POSIX errors,
    in which case ``errno`` carries the detail).
    """

    def __init__(self, op: str, op_error: int, errno: int | None = None):
        self.op_error = op_error
        self.errno = errno
        if op_error == 0:
            detail = f"errno={errno}"
            if errno is not None:
                detail += f" ({os.strerror(errno)})"
        else:
            detail = _ERROR_NAMES.get(op_error, f"hipFile error {op_error}")
        super().__init__(f"AIS {op} failed: {detail}")


_ERROR_NAMES: dict[int, str] = {
    HIPFILE_BASE_ERR + v: n
    for v, n in enumerate(
        (
            "DriverNotInitialized",
            "DriverInvalidProps",
            "DriverUnsupportedLimit",
            "DriverVersionMismatch",
            "DriverVersionReadError",
            "DriverClosing",
            "PlatformNotSupported",
            "IONotSupported",
            "DeviceNotSupported",
            "DriverError",
            "HipDriverError",
            "HipPointerInvalid",
            "HipMemoryTypeInvalid",
            "HipPointerRangeError",
            "HipContextMismatch",
            "InvalidMappingSize",
            "InvalidMappingRange",
            "InvalidFileType",
            "InvalidFileOpenFlag",
            "DIONotSet",
            "Unused21",
            "InvalidValue",
            "MemoryAlreadyRegistered",
            "MemoryNotRegistered",
            "PermissionDenied",
            "DriverAlreadyOpen",
            "HandleNotRegistered",
            "HandleAlreadyRegistered",
            "DeviceNotFound",
            "InternalError",
            "GetNewFDFailed",
            "Unused32",
            "DriverSetupError",
            "IODisabled",
            "BatchSubmitFailed",
            "GPUMemoryPinningFailed",
            "BatchFull",
            "AsyncNotSupported",
            "IOMaxError",
        ),
        start=1,
    )
}





class AisLib:
    """Thin typed wrapper over the synchronous libhipfile C API."""

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("VLLM_AIS_LIB", "libhipfile.so.0")
        self.lib = ctypes.CDLL(self.path, mode=ctypes.RTLD_LOCAL)
        self._bind()
        self._driver_refs = 0
        self.version = self.get_version()

    # -- binding ----------------------------------------------------------

    def _bind(self) -> None:
        lib = self.lib
        off_t = ctypes.c_int64
        ssize_t = ctypes.c_int64

        lib.hipFileGetVersion.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.hipFileGetVersion.restype = HipFileError_t

        lib.hipFileGetOpErrorString.argtypes = [ctypes.c_int]
        lib.hipFileGetOpErrorString.restype = ctypes.c_char_p

        lib.hipFileDriverOpen.argtypes = []
        lib.hipFileDriverOpen.restype = HipFileError_t

        lib.hipFileDriverClose.argtypes = []
        lib.hipFileDriverClose.restype = HipFileError_t

        lib.hipFileHandleRegister.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(HipFileDescr_t),
        ]
        lib.hipFileHandleRegister.restype = HipFileError_t

        lib.hipFileHandleDeregister.argtypes = [ctypes.c_void_p]
        lib.hipFileHandleDeregister.restype = None

        lib.hipFileBufRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.hipFileBufRegister.restype = HipFileError_t

        lib.hipFileBufDeregister.argtypes = [ctypes.c_void_p]
        lib.hipFileBufDeregister.restype = HipFileError_t

        lib.hipFileRead.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            off_t,
            off_t,
        ]
        lib.hipFileRead.restype = ssize_t

        lib.hipFileWrite.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            off_t,
            off_t,
        ]
        lib.hipFileWrite.restype = ssize_t

    # -- driver lifecycle -------------------------------------------------

    def get_version(self) -> tuple[int, int, int]:
        major, minor, patch = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        err = self.lib.hipFileGetVersion(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        self._check_err(err, "hipFileGetVersion")
        return major.value, minor.value, patch.value

    def driver_open(self) -> None:
        err = self.lib.hipFileDriverOpen()
        self._check_err(err, "hipFileDriverOpen")
        self._driver_refs += 1

    def driver_close(self) -> None:
        err = self.lib.hipFileDriverClose()
        self._check_err(err, "hipFileDriverClose")
        self._driver_refs = max(0, self._driver_refs - 1)

    # -- handles & buffers -------------------------------------------------

    def handle_register(self, fd: int) -> int:
        descr = HipFileDescr_t.from_fd(fd)
        fh = ctypes.c_void_p()
        err = self.lib.hipFileHandleRegister(ctypes.byref(fh), ctypes.byref(descr))
        self._check_err(err, "hipFileHandleRegister")
        if not fh.value:
            raise FsGpuError("hipFileHandleRegister", HIPFILE_BASE_ERR + 30)
        return fh.value

    def handle_deregister(self, fh: int) -> None:
        self.lib.hipFileHandleDeregister(ctypes.c_void_p(fh))

    def buf_register(self, base: int, length: int, flags: int = 0) -> None:
        err = self.lib.hipFileBufRegister(
            ctypes.c_void_p(base), ctypes.c_size_t(length), ctypes.c_int(flags)
        )
        self._check_err(err, "hipFileBufRegister")

    def buf_deregister(self, base: int) -> None:
        err = self.lib.hipFileBufDeregister(ctypes.c_void_p(base))
        self._check_err(err, "hipFileBufDeregister")

    # -- synchronous IO ----------------------------------------------------

    def read(self, fh: int, buf_base: int, size: int, file_offset: int, buf_offset: int = 0) -> int:
        """hipFileRead; returns bytes read (retries short reads)."""
        done = 0
        while done < size:
            got = self.lib.hipFileRead(
                ctypes.c_void_p(fh),
                ctypes.c_void_p(buf_base),
                ctypes.c_size_t(size - done),
                ctypes.c_int64(file_offset + done),
                ctypes.c_int64(buf_offset + done),
            )
            if got == -1:
                raise FsGpuError("hipFileRead", 0, ctypes.get_errno())
            if got < 0:
                raise FsGpuError("hipFileRead", -got)
            if got == 0:
                raise FsGpuError("hipFileRead", 0, 0)  # unexpected EOF
            done += got
        return done

    def write(self, fh: int, buf_base: int, size: int, file_offset: int, buf_offset: int = 0) -> int:
        """hipFileWrite; returns bytes written (retries short writes)."""
        done = 0
        while done < size:
            put = self.lib.hipFileWrite(
                ctypes.c_void_p(fh),
                ctypes.c_void_p(buf_base),
                ctypes.c_size_t(size - done),
                ctypes.c_int64(file_offset + done),
                ctypes.c_int64(buf_offset + done),
            )
            if put == -1:
                raise FsGpuError("hipFileWrite", 0, ctypes.get_errno())
            if put < 0:
                raise FsGpuError("hipFileWrite", -put)
            if put == 0:
                raise FsGpuError("hipFileWrite", 0, 0)
            done += put
        return done

    def error_string(self, op_error: int) -> str:
        s = self.lib.hipFileGetOpErrorString(ctypes.c_int(op_error))
        return s.decode() if s else f"hipFile error {op_error}"

    # -- helpers -----------------------------------------------------------

    def _check_err(self, err: HipFileError_t, op: str) -> None:
        if err.err != 0:  # hipFileSuccess
            raise FsGpuError(op, err.err if err.err > 0 else -err.err)


def verify_single_hip_runtime() -> list[str]:
    """Return the set of libamdhip64 mappings in this process.

    One entry (torch's bundled runtime) is expected after AIS is loaded;
    two distinct paths mean a second HIP runtime got initialized and device
    state is unsafe — callers must abort in that case.
    """
    paths: set[str] = set()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libamdhip64" in line:
                paths.add(line.rstrip().split(maxsplit=5)[-1])
    return sorted(paths)


_ais: AisLib | None = None


def get_ais() -> AisLib:
    global _ais
    if _ais is None:
        _ais = AisLib()
    return _ais
