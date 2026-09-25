"""CLI entry for starting or restarting the Orion UI."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

from platformdirs import user_runtime_dir

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def pid_path() -> Path:
    """Runtime pid file for the running Orion UI process."""
    return Path(user_runtime_dir("orion", appauthor=False)) / "orion-ui.pid"


def write_pid(path: Path | None = None) -> Path:
    target = path or pid_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(os.getpid()), encoding="utf-8")
    return target


def read_pid(path: Path | None = None) -> int | None:
    target = path or pid_path()
    if not target.is_file():
        return None
    try:
        return int(target.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    if sys.platform == "win32":
        import subprocess

        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            check=False,
            capture_output=True,
            text=True,
        )
        return
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return
    for _ in range(20):
        if not _pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return


def pids_listening_on_port(port: int) -> list[int]:
    """Best-effort list of PIDs bound to TCP ``port`` (Windows/Unix)."""
    if sys.platform == "win32":
        import subprocess

        result = subprocess.run(["netstat", "-ano", "-p", "tcp"], check=False, capture_output=True, text=True)
        pids: list[int] = []
        needle = f":{port}"
        for line in result.stdout.splitlines():
            if "LISTENING" not in line.upper() and "LISTEN" not in line.upper():
                continue
            if needle not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid not in pids:
                pids.append(pid)
        return pids

    import subprocess

    result = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], check=False, capture_output=True, text=True)
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def stop_existing(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, path: Path | None = None) -> list[int]:
    """Stop a previous Orion UI (pidfile and/or listeners on ``port``)."""
    stopped: list[int] = []
    previous = read_pid(path)
    if previous is not None and _pid_alive(previous):
        _terminate_pid(previous)
        stopped.append(previous)
    for pid in pids_listening_on_port(port):
        if pid == os.getpid() or pid in stopped:
            continue
        _terminate_pid(pid)
        stopped.append(pid)
    deadline = time.time() + 5.0
    while time.time() < deadline and not port_is_free(host, port):
        time.sleep(0.1)
    target = path or pid_path()
    if target.is_file():
        target.unlink(missing_ok=True)
    return stopped


def main(argv: list[str] | None = None) -> None:
    """``uv run orion-ui`` — stop any prior UI on the port, then start."""
    parser = argparse.ArgumentParser(prog="orion-ui", description="Start or restart the Orion NiceGUI app.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable NiceGUI auto-reload")
    parser.add_argument("--no-restart", action="store_true", help="Do not kill an existing process on the port")
    args = parser.parse_args(argv)

    if not args.no_restart:
        stopped = stop_existing(host=args.host, port=args.port)
        if stopped:
            print(f"Stopped previous UI process(es): {', '.join(map(str, stopped))}")

    write_pid()
    from orion.ui.app import run

    run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
