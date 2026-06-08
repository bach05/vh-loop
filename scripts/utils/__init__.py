""" scripts.utils - shared utility modules. """

from .bbox import (
    bbox_xyxy,
    annotation_label,
    bbox_iou,
    match_boxes,
)

from .io import (
    get_primary_image_asset,
    extract_bbox_annotations,
    resolve_image_path,
    load_canonical_samples,
    resolve_prediction_files,
    write_csv,
)

from .metrics import evaluate_prediction_file

from .visualization import (
    draw_box,
    render_sample_grid,
    plot_metric_by_threshold,
    plot_summary_bar,
)

