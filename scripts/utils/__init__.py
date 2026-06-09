""" scripts.utils - shared utility modules. """

from .bbox import (
    bbox_xyxy,
    annotation_label,
    bbox_iou,
    match_boxes,
)

from .io import (
    load_canonical_samples,
    resolve_prediction_files,
    load_and_stack_csvs,
    write_csv,
)

from .schema_helpers import (
    get_primary_image_asset,
    extract_bbox_annotations,
    resolve_image_path,
    get_dataset_info,
    label_name_to_id_map,
)

from .metrics import evaluate_prediction_file

from .visualization import (
    draw_box,
    render_sample_grid,
    plot_metric_by_threshold,
    plot_summary_bar,
)

