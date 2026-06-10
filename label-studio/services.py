# services.py
# Handles waiting for ports and launching Label Studio / SAM2 backend processes.

import os
import socket
import subprocess
import shlex
import shutil
import time
from pathlib import Path
from typing import Sequence


def wait_for_port(host: str, port: int, timeout_s: int = 180) -> None:
    """Blocks until the given host:port is reachable, or raises TimeoutError."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"{host}:{port} not reachable after {timeout_s}s")


def _is_windows() -> bool:
    return os.name == "nt"


def _env_exec_path(conda_root: Path, conda_env: str, exe_name: str) -> Path:
    """
    Return the executable path inside a conda environment.

    Windows: <conda_root>/envs/<env>/Scripts/<exe_name>.exe
    POSIX:   <conda_root>/envs/<env>/bin/<exe_name>
    """
    env_dir = conda_root / "envs" / conda_env
    if _is_windows():
        return env_dir / "Scripts" / f"{exe_name}.exe"
    return env_dir / "bin" / exe_name


def _popen_kwargs() -> dict:
    """Cross-platform process launch options."""
    if _is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_CONSOLE", 0)}
    return {"start_new_session": True}


def _launch_process(
    cmd: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    kwargs = _popen_kwargs()
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if env is not None:
        kwargs["env"] = env
    return subprocess.Popen(list(cmd), **kwargs)


def start_label_studio(
    conda_root: Path,
    conda_env: str,
    images_root: Path,
    port: int = 8080,
) -> None:
    """Launches Label Studio in a new console window and waits for it to be ready."""
    ls_exe = _env_exec_path(conda_root, conda_env, "label-studio")
    if not ls_exe.exists():
        raise FileNotFoundError(f"Not found: {ls_exe}")

    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = images_root.resolve().as_posix()
    env["LABEL_STUDIO_CORS_ALLOWED_ORIGINS"] = "*"

    _launch_process([str(ls_exe), "start"], env=env)
    print(f"Waiting for Label Studio to open on port {port}...")
    wait_for_port("127.0.0.1", port)

    print("Port opened. Waiting for Django/database startup...")
    time.sleep(10)


def _spawn_in_terminal(
    cmd: list[str],
    cwd: Path,
    env: dict,
) -> subprocess.Popen | None:
    """ Open cmd in a new, visible terminal window. Returns the Popen handle
    when the process is the backend itself (Windows), else None. """

    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    # Linux – try common terminal emulators in preference order
    shell_cmd = "echo $PATH && ~/.pixi/bin/pixi run start_ml_backend"

    cd_and_run = (
        f"cd {shlex.quote(str(cwd))} && "
        f"{shell_cmd}; "
        f"status=$?; "
        f"echo; "
        f"echo '[process exited with code' $status ']'; "
        f"exec bash"
    )

    _TERMINALS = {
        "x-terminal-emulator": ["-e", "bash", "-c"],
        "gnome-terminal": ["--", "bash", "-c"],
        "konsole":        ["-e", "bash", "-c"],
        "xfce4-terminal": ["-e", "bash", "-c"],
        "xterm":          ["-e", "bash", "-c"],
    }
    for binary, flags in _TERMINALS.items():
        if shutil.which(binary):
            try:
                subprocess.Popen(
                    [binary, *flags, cd_and_run],
                    start_new_session=True,
                    env=env,
                )
            except Exception as e:
                print(f"Failed to launch {binary}: {e}")
                continue
            return None

    raise RuntimeError(
        "No supported terminal emulator found on PATH "
        f"(tried: {', '.join(_TERMINALS)})"
    )



def start_sam_backend(
    conda_root: Path,
    conda_env: str,
    backend_dir: Path,
    port: int = 9090,
    backend_module: str = "./segment_anything_2_image",
    label_studio_url: str = "http://127.0.0.1:8080",
    label_studio_api_key: str | None = None,
) -> None:
    """Start SAM2 ML backend cross-platform."""
    env_dir = conda_root / "envs" / conda_env
    backend_dir = backend_dir.expanduser().resolve() #this to fix: since resolve concats the current root when ~ is used, need to test with a full path
    print(f"Starting SAM2 backend dir: {backend_dir}")

    ml_exe = (
        env_dir / "Scripts" / "label-studio-ml.exe"
        if os.name == "nt"
        else env_dir / "bin" / "label-studio-ml"
    )
    print(f"Starting SAM2 backend with executable: {ml_exe}")
    if not ml_exe.exists():
        raise FileNotFoundError(f"Not found: {ml_exe}")

    if not label_studio_api_key:
        raise ValueError("LABEL_STUDIO_API_KEY is required for the SAM backend")

    env = os.environ.copy()
    env["LABEL_STUDIO_URL"] = label_studio_url
    env["LABEL_STUDIO_API_KEY"] = label_studio_api_key

    cmd = [str(ml_exe), "start", backend_module, "-p", str(port)]

    proc = _spawn_in_terminal(cmd, cwd=backend_dir, env=env)
    time.sleep(3)

    if proc is not None and proc.poll() is not None:
        raise RuntimeError(f"SAM backend exited immediately with code {proc.returncode}")

    wait_for_port("127.0.0.1", port, timeout_s=180)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Label Studio and/or SAM2 backend.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--conda-root", type=Path, default=Path.home() / "mambaforge")
    parser.add_argument("--ls-conda-env", default="ls-ui")
    parser.add_argument("--ls-url", default="http://localhost:8080")
    parser.add_argument("--ls-port", type=int, default=8080)

    parser.add_argument("--start-sam", action="store_true", help="Also start SAM2 backend")
    parser.add_argument("--sam-conda-env", default="ls-sam2")
    parser.add_argument(
        "--sam-backend-dir",
        type=Path,
        default=Path(r"C:\ITR\label-studio-ml-backend\label_studio_ml\examples")
        if _is_windows()
        else Path.home(),
    )
    parser.add_argument("--sam-port", type=int, default=9090)
    parser.add_argument(
        "--sam-module",
        default="./segment_anything_2_image",
        help="Backend module/path passed to label-studio-ml start",
    )

    args = parser.parse_args()

    print("Starting Label Studio...")
    start_label_studio(args.conda_root, args.ls_conda_env, args.images_root, args.ls_port)
    print("Label Studio started.")

    if args.start_sam:
        print("Starting SAM2 backend...")
        start_sam_backend(
            args.conda_root,
            args.sam_conda_env,
            args.sam_backend_dir,
            args.sam_port,
            args.sam_module,
            args.ls_url,
            args.api_key
        )
        print("SAM2 backend started.")
