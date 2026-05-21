from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError


def export_project_json(
    ls_url: str,
    api_key: str,
    project_id: int,
    output_dir: Path,
    title: str | None = None,
    poll_interval_s: int = 3,
    timeout_s: int = 600,
) -> Path:
    """ Export a Label Studio project to JSON using the official SDK snapshot API. """
    output_dir.mkdir(parents=True, exist_ok=True)

    client = LabelStudio(base_url=ls_url, api_key=api_key)

    # Ensure the project exists or is reachable
    _ = client.projects.get(id=project_id)

    snapshot_title = title or f"Export project {project_id}"
    export_job = client.projects.exports.create(
        id=project_id,
        title=snapshot_title,
    )

    export_pk = export_job.id
    deadline = time.time() + timeout_s

    while True:
        job = client.projects.exports.get(id=project_id, export_pk=export_pk)

        if job.status == "completed":
            break

        if job.status == "failed":
            raise RuntimeError(f"Export failed for project {project_id}, export_pk={export_pk}")

        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for export completion "
                f"(project_id={project_id}, export_pk={export_pk}, status={job.status})"
            )

        time.sleep(poll_interval_s)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"project_{project_id}_{timestamp}.json"

    with open(out_path, "wb") as f:
        for chunk in client.projects.exports.download(
            id=project_id,
            export_pk=export_pk,
            export_type="JSON",
            request_options={"chunk_size": 1024},
        ):
            f.write(chunk)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Label Studio project as JSON.")
    parser.add_argument("--ls-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default=None,
        help="Optional export snapshot title.",
    )
    args = parser.parse_args()

    print(f"Exporting project {args.project_id} from {args.ls_url} ...")
    out_path = export_project_json(
        ls_url=args.ls_url,
        api_key=args.api_key,
        project_id=args.project_id,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()