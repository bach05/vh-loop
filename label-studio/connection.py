# connection.py
# Handles the Label Studio client and project management: create/update projects, load XML config, connect ML backends.

import time
from pathlib import Path

from label_studio_sdk import LabelStudio


def make_client(api_key: str, base_url: str = "http://localhost:8080") -> LabelStudio:
    """Returns an authenticated Label Studio client."""
    return LabelStudio(base_url=base_url, api_key=api_key)


def load_label_config(path: Path) -> str:
    """Reads and returns the XML labelling configuration from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Labeling config not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_or_create_project(
    client: LabelStudio,
    title: str,
    description: str,
    label_config: str,
    retries: int = 10,
):
    """
    Returns an existing project matching *title* (updating its config),
    or creates a new one. Retries on Label Studio startup errors.
    """
    for i in range(retries):
        try:
            for p in client.projects.list():
                if p.title == title:
                    print(f"Existing project (ID={p.id}). Updating XML config...")
                    updated = client.projects.update(
                        id=p.id,
                        title=title,
                        description=description,
                        label_config=label_config,
                    )
                    return updated or p
            break
        except Exception as e:
            msg = str(e)
            if "Invalid page" in msg or "404" in msg or "NotFound" in msg:
                print(f"[Attempt {i+1}/{retries}] Label Studio not stable yet. Waiting 3s...")
                time.sleep(3)
            else:
                raise

    print(f"Creating new project: {title}")
    return client.projects.create(
        title=title,
        description=description,
        label_config=label_config,
    )


def setup_local_storage(
        client: LabelStudio,
        project_id: int,
        storage_path: Path,
        title: str = "Local Images",
) -> None:
    """
    Creates a local file source storage connection for the project.
    storage_path must be a subfolder of LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.
    """
    # LS on Windows needs single backslashes in the API path field
    path_str = str(storage_path.absolute())

    try:
        client.import_storage.local.create(
            project=project_id,
            path=path_str,
            use_blob_urls=True,
            title=title,
        )
        print(f"Local storage connected: '{path_str}' → project {project_id}.")
    except Exception as e:
        print(f"Could not create local storage (already exists?): {e}")


def connect_ml_backend(
    client: LabelStudio,
    project_id: int,
    backend_url: str,
    title: str = "SAM2 Backend",
) -> None:
    """Registers an ML backend with the given project."""
    try:
        client.ml.create(title=title, url=backend_url, project=project_id)
        print(f"ML backend '{title}' connected to project {project_id}.")
    except Exception as e:
        print(f"Could not connect ML backend (already registered?): {e}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from services import wait_for_port

    parser = argparse.ArgumentParser(description="Create or update a Label Studio project.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-title", required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--ls-url", default="http://localhost:8080")
    parser.add_argument(
        "--connect-sam",
        action="store_true",
        help="Connect SAM2 backend to the project (must already be running)",
    )
    parser.add_argument("--sam-url", default="http://localhost:9090")
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=None,
        help="Absolute path to the images folder for local storage (must be under DOCUMENT_ROOT)",
    )
    args = parser.parse_args()

    print("Checking Label Studio is reachable...")
    wait_for_port("127.0.0.1", 8080)

    client = make_client(args.api_key, args.ls_url)
    config_xml = load_label_config(args.label_config)
    project = get_or_create_project(client, args.project_title, args.description, config_xml)
    print(f"Project ready: ID={project.id}, title='{project.title}'")

    if args.connect_sam:
        connect_ml_backend(client, project.id, args.sam_url)

    if args.storage_path:
        setup_local_storage(client, project.id, args.storage_path)