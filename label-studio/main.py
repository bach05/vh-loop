# main.py
# Orchestrates the full pipeline by calling services, connection, and importer.

import argparse
from pathlib import Path
from urllib.parse import urlparse

from services import wait_for_port, start_label_studio, start_sam_backend
from connection import make_client, load_label_config, get_or_create_project, connect_ml_backend, setup_local_storage
from importer import read_samples, build_tasks, import_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Studio full setup pipeline.")

    # Required
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-title", required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument(
        "--storage-path",
        type=Path,
        required=True,
        help="Absolute path to images folder. Used both for task URL building and local storage connection.",
    )

    # Optional / service flags
    parser.add_argument("--description", default="")
    parser.add_argument("--ls-url", default="http://localhost:8080")
    parser.add_argument("--ls-port", type=int, default=8080)
    parser.add_argument(
        "--start-services",
        action="store_true",
        help="Launch Label Studio (set if it's not already running)",
    )
    parser.add_argument("--conda-root", type=Path, default=Path.home() / "mambaforge")
    parser.add_argument("--ls-conda-env", default="ls-ui")

    # SAM2
    parser.add_argument(
        "--connect-sam-backend",
        action="store_true",
        help="Start and register the SAM2 ML backend",
    )
    parser.add_argument("--sam-port", type=int, default=9090)
    parser.add_argument(
        "--sam-bat",
        type=Path,
        default=Path(r"C:\ITR\vh-loop\label-studio\LS-ML.bat"),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # STEP 1 – Services
    # ------------------------------------------------------------------ #
    if args.start_services:
        print("STEP 1/5: starting Label Studio...")
        start_label_studio(args.conda_root, args.ls_conda_env, args.images_root, args.ls_port)
    else:
        print("STEP 1/5: check if Label Studio is reachable...")
        parsed = urlparse(args.ls_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        wait_for_port(host, port)
    print("STEP 1/5 completed.\n")

    # ------------------------------------------------------------------ #
    # STEP 2 – Project setup
    # ------------------------------------------------------------------ #
    print("STEP 2/5: creating the client and configuring the project...")
    client = make_client(args.api_key, args.ls_url)
    config_xml = load_label_config(args.label_config)
    project = get_or_create_project(
        client,
        title=args.project_title,
        description=args.description,
        label_config=config_xml,
    )
    print(f"STEP 2/5 completed: project_id={project.id}\n")

    # ------------------------------------------------------------------ #
    # STEP 3 – Local storage
    # ------------------------------------------------------------------ #
    print(f"STEP 3/5: configure local storage → {args.storage_path}...")
    setup_local_storage(client, project.id, args.storage_path)
    print("STEP 3/5 completed.\n")
    
    # ------------------------------------------------------------------ #
    # STEP 4 – Parse JSONL and import tasks #
    # ------------------------------------------------------------------ #
    print(f"STEP 4/5: reading JSONL from {args.jsonl}...")
    samples, dataset_id = read_samples(args.jsonl)
    print(f"Samples read: {len(samples)}  |  dataset_id: {dataset_id}")

    if not samples:
        print("Critical error: no samples extracted from the JSONL file. Exiting.")
        raise SystemExit(1)

    tasks = build_tasks(samples, args.images_root)
    print(f"Valid tasks generated: {len(tasks)}")

    if not tasks:
        print("No valid tasks generated (check the previous warning logs). Exiting.")
        raise SystemExit(1)

    import_tasks(client, project.id, tasks)
    print("STEP 4/5 completed.\n")

    # ------------------------------------------------------------------ #
    # STEP 5 – SAM2 backend (optional)
    # ------------------------------------------------------------------ #
    if args.connect_sam_backend:
        print("STEP 5/5: starting and connecting SAM2 backend...")
        start_sam_backend(args.sam_bat, args.sam_port)
        connect_ml_backend(
            client,
            project_id=project.id,
            backend_url=f"http://localhost:{args.sam_port}",
        )
        print("STEP 5/5 completed.\n")
    else:
        print("STEP 5/5: skipped (--connect-sam-backend not specified).\n")

    print(f"DONE! Open Label Studio and go to Project ID: {project.id}")


if __name__ == "__main__":
    main()