"""Pick the OpenGL the 3D viewer runs on: the graphics driver's, or software.

The VTK viewer needs OpenGL 3.2 or later. A virtual machine without a GPU
(Hyper-V, many VDI desktops) or a remote session often offers only Windows'
built-in OpenGL 1.1, and VTK then crashes natively while creating the
viewport: no Python exception, no error message, the window just never opens.

PySide6 ships Qt's software OpenGL fallback, `opengl32sw.dll` (Mesa llvmpipe,
OpenGL 3.3 core), so a working renderer is already installed. When the
driver's OpenGL is too old, that DLL is copied to `.venv/swgl/opengl32.dll`
and loaded before Qt or VTK. Windows then hands
it to every later request for "opengl32.dll" in this process, because a DLL
of that name is already loaded. Nothing is downloaded and nothing is written
outside the tool folder.

The driver's version is read in a short-lived child process (this file, run
as a script): loading the system opengl32.dll into the app's own process
would make the fallback impossible, and a broken driver may crash rather
than answer.

    python partestimator/opengl.py     prints e.g. "1.1|GDI Generic"
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_VERSION = (3, 2)
CHOICES = ("auto", "hardware", "software")
PROBE_TIMEOUT_S = 15.0


def _tool_dir() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# the probe (runs in the child process)
# ---------------------------------------------------------------------------


def _query_driver() -> tuple[str, str]:
    """(GL_VERSION, GL_RENDERER) from a legacy WGL context on a hidden window."""
    import ctypes
    from ctypes import wintypes as wt

    user32 = ctypes.WinDLL("user32")
    gdi32 = ctypes.WinDLL("gdi32")
    gl = ctypes.WinDLL("opengl32")

    class PFD(ctypes.Structure):
        _fields_ = [("nSize", wt.WORD), ("nVersion", wt.WORD), ("dwFlags", wt.DWORD)]
        _fields_ += [(n, ctypes.c_ubyte) for n in (
            "iPixelType", "cColorBits", "cRedBits", "cRedShift", "cGreenBits",
            "cGreenShift", "cBlueBits", "cBlueShift", "cAlphaBits", "cAlphaShift",
            "cAccumBits", "cAccumRedBits", "cAccumGreenBits", "cAccumBlueBits",
            "cAccumAlphaBits", "cDepthBits", "cStencilBits", "cAuxBuffers",
            "iLayerType", "bReserved")]
        _fields_ += [("dwLayerMask", wt.DWORD), ("dwVisibleMask", wt.DWORD),
                     ("dwDamageMask", wt.DWORD)]

    HANDLE = ctypes.c_void_p
    user32.CreateWindowExW.restype = HANDLE
    user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, HANDLE, HANDLE, HANDLE, HANDLE]
    user32.GetDC.restype = HANDLE
    user32.GetDC.argtypes = [HANDLE]
    user32.ReleaseDC.argtypes = [HANDLE, HANDLE]
    user32.DestroyWindow.argtypes = [HANDLE]
    gdi32.ChoosePixelFormat.argtypes = [HANDLE, ctypes.POINTER(PFD)]
    gdi32.SetPixelFormat.argtypes = [HANDLE, ctypes.c_int, ctypes.POINTER(PFD)]
    gl.wglCreateContext.restype = HANDLE
    gl.wglCreateContext.argtypes = [HANDLE]
    gl.wglMakeCurrent.argtypes = [HANDLE, HANDLE]
    gl.wglDeleteContext.argtypes = [HANDLE]
    gl.glGetString.restype = ctypes.c_char_p
    gl.glGetString.argtypes = [ctypes.c_uint]

    # "STATIC" is a predefined window class, so nothing has to be registered.
    hwnd = user32.CreateWindowExW(0, "STATIC", "gl-probe", 0, 0, 0, 1, 1,
                                  None, None, None, None)
    if not hwnd:
        raise OSError("could not create a probe window")
    hdc = user32.GetDC(hwnd)
    ctx = None
    try:
        pfd = PFD()
        pfd.nSize = ctypes.sizeof(PFD)
        pfd.nVersion = 1
        pfd.dwFlags = 0x04 | 0x20 | 0x01  # DRAW_TO_WINDOW | SUPPORT_OPENGL | DOUBLEBUFFER
        pfd.cColorBits = 32
        pfd.cDepthBits = 24
        fmt = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
        if not fmt or not gdi32.SetPixelFormat(hdc, fmt, ctypes.byref(pfd)):
            raise OSError("no OpenGL pixel format")
        ctx = gl.wglCreateContext(hdc)
        if not ctx or not gl.wglMakeCurrent(hdc, ctx):
            raise OSError("could not create an OpenGL context")
        version = (gl.glGetString(0x1F02) or b"").decode("ascii", "replace")
        renderer = (gl.glGetString(0x1F01) or b"").decode("ascii", "replace")
        return version, renderer
    finally:
        if ctx:
            gl.wglMakeCurrent(None, None)
            gl.wglDeleteContext(ctx)
        user32.ReleaseDC(hwnd, hdc)
        user32.DestroyWindow(hwnd)


# ---------------------------------------------------------------------------
# the decision (runs in the app)
# ---------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, int] | None:
    m = re.match(r"\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def probe_driver() -> tuple[tuple[int, int] | None, str]:
    """The driver's OpenGL version and a description, or (None, why not)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve())],
                           capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                           creationflags=flags)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"probe failed ({type(exc).__name__})"
    if r.returncode != 0:
        return None, f"probe failed (exit code {r.returncode})"
    version, _, renderer = r.stdout.strip().partition("|")
    return parse_version(version), f"{version} ({renderer})"


def software_dll() -> Path | None:
    """PySide6's bundled Mesa llvmpipe, found without importing Qt."""
    import importlib.util

    spec = importlib.util.find_spec("PySide6")
    for root in (spec.submodule_search_locations or []) if spec else []:
        p = Path(root) / "opengl32sw.dll"
        if p.is_file():
            return p
    return None


def _copy_dir() -> Path:
    """Beside the PySide6 it came from: inside .venv, so `pe.ps1 reset` removes
    it too. A Python outside a venv uses the tool's scratch folder instead."""
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix) / "swgl"
    return _tool_dir() / "scratch" / "swgl"


def load_software_gl() -> str:
    """Make the bundled software renderer this process's opengl32.dll."""
    import ctypes

    src = software_dll()
    if src is None:
        raise FileNotFoundError("PySide6's opengl32sw.dll is not installed")
    dst = _copy_dir() / "opengl32.dll"
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    if kernel32.GetModuleHandleW("opengl32.dll"):
        raise RuntimeError("opengl32.dll was already loaded; software OpenGL must be "
                           "chosen before Qt or VTK are imported")
    ctypes.WinDLL(str(dst))
    return "software OpenGL (Mesa llvmpipe bundled with PySide6)"


def choose(mode: str = "auto") -> str:
    """Apply the choice before Qt/VTK load. Returns a line for the log.

    Never raises: the worst case is the driver's OpenGL, which is what ran
    before this module existed.
    """
    if sys.platform != "win32" or mode == "hardware":
        return "driver OpenGL" + (" (forced)" if mode == "hardware" else "")
    try:
        if mode == "software":
            return load_software_gl() + " (forced)"
        version, detail = probe_driver()
        if version is not None and version >= MIN_VERSION:
            return f"driver OpenGL {detail}"
        return f"{load_software_gl()}; driver offers {detail}"
    except Exception as exc:  # noqa: BLE001 -- fall back to what always ran
        return f"driver OpenGL; software fallback unavailable ({type(exc).__name__}: {exc})"


if __name__ == "__main__":
    v, r = _query_driver()
    print(f"{v}|{r}")
