# services.py
# Handles waiting for ports and launching Label Studio / SAM2 backend processes.

import os
import socket
import subprocess
import time
from pathlib import Path


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


def start_label_studio(
    conda_root: Path,
    conda_env: str,
    images_root: Path,
    port: int = 8080,
) -> None:
    """Launches Label Studio in a new console window and waits for it to be ready."""
    env_dir = conda_root / "envs" / conda_env
    ls_exe = env_dir / "Scripts" / "label-studio.exe"

    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(images_root.resolve())
    os.environ["LABEL_STUDIO_CORS_ALLOWED_ORIGINS"] = "*"

    subprocess.Popen([str(ls_exe), "start"], creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    print(f"Waiting for Label Studio to open on port {port}...")
    wait_for_port("127.0.0.1", port)

    print("Port opened. Wait some extra time for Django to start database...")
    time.sleep(10)

'''
def start_sam_backend(
    conda_root: Path,
    conda_env: str,
    backend_dir: Path,
    port: int = 9090,
) -> None:
    """Launches the SAM2 ML backend in a new console window and waits for it to be ready."""
    env_dir = conda_root / "envs" / conda_env
    ml_exe = env_dir / "Scripts" / "label-studio-ml.exe"

    if not ml_exe.exists():
        raise FileNotFoundError(f"Not found: {ml_exe}")

    print(f"Starting SAM2 Backend on port {port}...")
    subprocess.Popen(
        [str(ml_exe), "start", "./segment_anything_2_image", "-p", str(port)],
        cwd=str(backend_dir),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    wait_for_port("127.0.0.1", port, timeout_s=180)
'''

def start_sam_backend(bat_path: Path, port: int = 9090) -> None:
    if not bat_path.exists():
        raise FileNotFoundError(f"Bat not found: {bat_path}")

    subprocess.Popen(
        ["cmd.exe", "/c", str(bat_path)],
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    wait_for_port("127.0.0.1", port, timeout_s=180)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Label Studio and/or SAM2 backend.")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--conda-root", type=Path, default=Path.home() / "mambaforge")
    parser.add_argument("--ls-conda-env", default="ls-ui")
    parser.add_argument("--ls-port", type=int, default=8080)
    parser.add_argument("--start-sam", action="store_true", help="Also start SAM2 backend")
    # parser.add_argument("--sam-conda-env", default="ls-sam2")
    # parser.add_argument(
    #     "--sam-backend-dir",
    #     type=Path,
    #     default=Path(r"C:\\ITR\\label-studio-ml-backend\\label_studio_ml\\examples"),
    # )
    parser.add_argument("--sam-port", type=int, default=9090)
    parser.add_argument("--sam-bat", type=Path, default=Path(r"C:\\ITR\\vh-loop\\label-studio\\LS-ML.bat"))
    args = parser.parse_args()

    print("Starting Label Studio...")
    start_label_studio(args.conda_root, args.ls_conda_env, args.images_root, args.ls_port)
    print("Label Studio started.")

    if args.start_sam:
        print("Starting SAM2 backend...")
        # start_sam_backend(args.conda_root, args.sam_conda_env, args.sam_backend_dir, args.sam_port)
        start_sam_backend(args.sam_bat, args.sam_port)
        print("SAM2 backend started.")
