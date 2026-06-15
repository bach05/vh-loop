from __future__ import annotations

"""
compare_sample.py - Compare canonical_schema v2 prediction JSONL files.

This version is aligned with the new canonical schema:
- JSONL header: DatasetInfoRecord
- JSONL samples: DataRecord(sample=DataSample / SISimpleDataSample / future sample types)
- Annotations: InstanceAnnotation objects attached to sample assets

The script intentionally reuses schema models such as InstanceAnnotation,
BoundingBox, DataRecord and DatasetInfoRecord instead of defining duplicate
BoxItem-like containers.
"""

import logging
from pathlib import Path
import yaml

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.utils import (
    load_canonical_samples,
    resolve_prediction_files,
    write_csv,
    evaluate_prediction_file,
    plot_metric_by_threshold,
    plot_summary_bar,
    render_sample_grid
)

# Register custom Hydra resolvers only once --> TO DO: move it into a generic function/init shared between all scripts
if not OmegaConf.has_resolver("strip_null"):
    OmegaConf.register_new_resolver(
        "strip_null",
        lambda val: f"_{val}" if val is not None else "",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="compare_entrypoint")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    out_dir = Path(to_absolute_path(str(cfg.get("output_dir", "."))))
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(t) for t in cfg.get("thresholds", [0.1, 0.25, 0.5, 0.75, 0.9])]
    class_aware = bool(cfg.get("class_aware", True))

    # ------------------------------------------------------------------
    # Evaluate each prediction file
    # ------------------------------------------------------------------

    prediction_files = resolve_prediction_files(cfg)
    if not prediction_files:
        raise ValueError("No prediction JSONL files found. Use predictions or predictions_dir.")

    all_metric_rows = []
    summary_rows = []
    pred_samples_by_model = {}

    for model_name, pred_path in prediction_files:
        logging.info(f"Evaluating {model_name}: {pred_path}")

        pred_samples, _ = load_canonical_samples(pred_path)
        pred_samples_by_model[model_name] = pred_samples

        #Load gt samples from config folder
        logging.info(f"********* Looking for config file in {pred_path.parent / 'configs' / 'config.yaml'} to load GT samples for {model_name}")
        test_folder = pred_path.parent
        config_file = test_folder / "configs" / "config.yaml"
        if config_file.exists():
            try:
                cfg = OmegaConf.load(config_file)
                # Resolve ${oc.env:...}, ${...}, etc.
                test_cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception as e:
                logging.exception(f"Failed to load/resolve Hydra config at {config_file}: {e}")
                continue
        else:
            logging.warning(
                f"Config file not found at {config_file}, discarding evaluation of {model_name}"
            )
            continue

        merged_gt_samples = {}
        for test_dict in test_cfg['dataset']['testing']:
            testing_jsonl_file = test_dict['jsonl_path']

            gt_samples, gt_info = load_canonical_samples(testing_jsonl_file)
            logging.info(f"Loaded {len(gt_samples)} GT samples from {testing_jsonl_file}")
            logging.info(f"GT dataset_id: {gt_info.dataset_id}")
            merged_gt_samples.update(gt_samples)

        logging.info(f"Total merged GT samples for {model_name}: {len(merged_gt_samples)}")
        logging.info("--------------------------------------------------\n")

        metric_rows, summary = evaluate_prediction_file(
            model_name=model_name,
            gt_samples=merged_gt_samples,
            pred_samples=pred_samples,
            thresholds=thresholds,
            class_aware=class_aware,
        )

        all_metric_rows.extend(metric_rows)
        summary_rows.append(summary)

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    write_csv(out_dir / "metrics_by_threshold.csv", all_metric_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    # ------------------------------------------------------------------
    # Aggregate plots
    # ------------------------------------------------------------------
    if cfg.get("plots", {}).get("enabled", True):
        for metric in ["precision", "recall", "f1", "mean_iou_tp"]:
            plot_metric_by_threshold(
                all_metric_rows,
                metric=metric,
                out_path=out_dir / f"{metric}_by_threshold.png",
            )
        plot_summary_bar(
            summary_rows,
            metric="miou_gt_penalized",
            out_path=out_dir / "miou_gt_penalized.png",
        )

    # ------------------------------------------------------------------
    # Per-sample visualisations
    # ------------------------------------------------------------------
    if cfg.get("visualization", {}).get("enabled", False):
        vis_cfg = cfg.visualization
        threshold_mode = vis_cfg.get("threshold_mode", "best_f1")
        fixed_threshold = float(vis_cfg.get("fixed_threshold", 0.5))

        if threshold_mode == "best_f1":
            thresholds_by_model = {r["model"]: float(r["best_threshold"]) for r in summary_rows}
        elif threshold_mode == "fixed":
            thresholds_by_model = {r["model"]: fixed_threshold for r in summary_rows}
        else:
            raise ValueError(
                f"Unsupported visualization.threshold_mode={threshold_mode!r}. "
                "Use 'best_f1' or 'fixed'."
            )

        requested_ids = [str(s) for s in (vis_cfg.get("sample_ids", []) or [])]
        sample_ids = requested_ids or sorted(merged_gt_samples.keys())[: int(vis_cfg.get("max_samples", 25))]
        image_root = vis_cfg.get("image_root", None)

        for sid in tqdm(sample_ids, desc="Processing visualizations"):
            gt_sample = merged_gt_samples.get(sid)
            if gt_sample is None:
                logging.warning(f"Skipping visualization: missing GT sample_id={sid}")
                continue

            render_sample_grid(
                sample_id=sid,
                gt_sample=gt_sample,
                pred_samples_by_model=pred_samples_by_model,
                thresholds_by_model=thresholds_by_model,
                class_aware=class_aware,
                image_root=image_root,
                out_path=out_dir / "visualizations" / f"sample_{sid}.png",
            )

    logging.info(f"Comparison complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()