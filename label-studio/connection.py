# connection.py
# Handles the Label Studio client and project management: create/update projects, load XML config, connect ML backends.

import json
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, Tuple
from label_studio_sdk import LabelStudio

DEFAULT_COLORS = [ "#5B9BFF", "#E8596A", "#7ED321", "#F5A623", "#BD10E0", "#50E3C2", "#F8E71C", "#B8E986", ]


def make_client(api_key: str, base_url: str = "http://localhost:8080") -> LabelStudio:
    """ Returns an authenticated Label Studio client. """
    return LabelStudio(base_url=base_url, api_key=api_key)


def read_dataset_info(jsonl_path: Path) -> Tuple[str, Dict[str, Any]]:
    """ Reads dataset_info row from JSONL and extracts dataset_id + label_info. """
    dataset_id = "unknown"
    label_info: Dict[str, Any] = {}

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            if data.get("record_type") != "dataset_info":
                continue

            info = data.get("info", {})
            dataset_id = info.get("dataset_id", dataset_id)
            label_info = info.get("label_info", {})
            break

    if not label_info:
        raise ValueError("No label_info found inside dataset_info")

    return dataset_id, label_info


def load_or_generate_config(config_path: Path, jsonl_path: Path | None = None) -> str:
    """
    Loads an existing XML config if present.
    Otherwise generates it from dataset_info inside the JSONL, saves it to config_path and returns it.
    """
    if config_path.exists():
        print(f"Loading existing config: {config_path}")
        return config_path.read_text(encoding="utf-8").strip()

    if jsonl_path is None:
        raise ValueError(
            "jsonl_path is required when config does not exist"
        )

    dataset_id, label_info = read_dataset_info(jsonl_path)

    print(f"Dataset ID: {dataset_id}")
    print(f"Found {len(label_info)} labels")

    config_xml = build_label_config(label_info)
    config_path.write_text(config_xml, encoding="utf-8")
    print(f"Generated config saved to: {config_path}")

    return config_xml


def build_label_config(label_info: Dict[str, Any]) -> str:
    """ Dynamically generates Label Studio XML config from label_info. """
    labels = sorted(
        label_info.values(),
        key=lambda x: x["label_id"]
    )

    lines = [
        "<View>",
        '  <Header value="Sample ID: $sample_id"/>',
        "",
        '  <Image',
        '    name="image"',
        '    value="$image"',
        '    zoom="true"',
        '    zoomControl="true"',
        '    brightnessControl="true"',
        '  />',
        "",
        '  <Header value="SAM2 Points"/>',
        '  <KeyPointLabels name="sam_points" toName="image" smart="true">',
    ]

    for i, label in enumerate(labels):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

        hotkey = ""
        if i < 9:
            hotkey = f' hotkey="{i + 1}"'

        lines.append(
            f'    <Label value="{label["label_name"]}" '
            f'background="{color}" '
            f'showInline="true"{hotkey}/>'
        )

    lines += [
        "  </KeyPointLabels>",
        "",
        '  <Header value="SAM2 Masks"/>',
        '  <BrushLabels name="mask" toName="image" smart="true" smartOnly="true">',
    ]

    for i, label in enumerate(labels):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

        lines.append(
            f'    <Label value="{label["label_name"]}" '
            f'background="{color}"/>'
        )

    lines += [
        "  </BrushLabels>",
        "</View>",
    ]

    return "\n".join(lines)


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
    path_str = str(storage_path.absolute())

    try:
        client.import_storage.local.create(
            project=project_id,
            path=path_str,
            use_blob_urls=True,
            title=title,
        )
        print(f"Local storage connected: '{path_str}' -> project {project_id}.")
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
        client.ml.create(title=title, url=backend_url, project=project_id, is_interactive=True)
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
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path of the XML config to load if present, otherwise generate from JSONL",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Dataset JSONL used only if config_path does not exist and config must be generated",
    )
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

    parsed = urlparse(args.ls_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    wait_for_port(host, port)
    print(f"Checking Label Studio is reachable at {host}:{port}...")
    wait_for_port(host, port)

    client = make_client(args.api_key, args.ls_url)
    config_xml = load_or_generate_config(config_path=args.config_path, jsonl_path=args.jsonl)
    project = get_or_create_project(client, args.project_title, args.description, config_xml)
    print(f"Project ready: ID={project.id}, title='{project.title}'")

    if args.connect_sam:
        connect_ml_backend(client, project.id, args.sam_url)

    if args.storage_path:
        setup_local_storage(client, project.id, args.storage_path)