import re
import ast
import json
import logging

from typing import List, Any, Dict, Tuple, Optional

from data.schema.schema import Target, Annotation, BoundingBox, Point
from scripts.core.constants import NORM_SIZE


def _extract_after_last_channel(s: str) -> str:
    """
    Extract content after the last <channel|> marker.
    If a trailing <turn|> exists, cut before it.
    """
    matches = list(re.finditer(r"<channel\|>", s, flags=re.IGNORECASE))
    if not matches:
        return ""

    tail = s[matches[-1].end():].strip()

    turn_match = re.search(r"<turn\|>", tail, flags=re.IGNORECASE)
    if turn_match:
        tail = tail[:turn_match.start()].strip()

    return tail


def _extract_after_last_think(s: str) -> str:
    """
    Useful for models that output reasoning and then JSON after </think>.
    """
    matches = list(re.finditer(r"</think>", s, flags=re.IGNORECASE))
    if not matches:
        return ""

    return s[matches[-1].end():].strip()


def _unwrap_string_literal(s: str) -> str:
    s = s.strip()
    if not s:
        return s

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, str):
            return parsed.strip()
    except Exception:
        pass

    return s


def _strip_outer_code_fence(s: str) -> str:
    s = s.strip()

    m = re.match(
        r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if m:
        return m.group(1).strip()

    return s


def _extract_fenced_blocks(s: str) -> List[str]:
    matches = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return [m.strip() for m in matches if m.strip()]


def _extract_all_json_array_substrings(s: str) -> List[str]:
    """
    Extract all top-level [...] substrings while handling quoted strings.
    """
    results = []
    start = None
    depth = 0
    in_string = False
    string_char = ""
    escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            continue

        if ch == "[":
            if depth == 0:
                start = i
            depth += 1

        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    results.append(s[start:i + 1])
                    start = None

    return results


def _parse_json_list(s: str) -> list[Any]:
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None

    if parsed is None:
        try:
            parsed = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(f"Could not parse content as JSON/list: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"Parsed object is not a list: {type(parsed)}")

    return parsed


def _extract_json_candidates(block: str) -> List[str]:
    """
    From a raw text block, extract all plausible JSON array candidates.
    """
    s = block.strip()
    if not s:
        return []

    s = _unwrap_string_literal(s)
    s = _strip_outer_code_fence(s)

    candidates: list[str] = []

    if s.startswith("[") and s.endswith("]"):
        candidates.append(s)

    fenced_blocks = _extract_fenced_blocks(s)
    for fb in fenced_blocks:
        fb = _unwrap_string_literal(fb)
        fb = _strip_outer_code_fence(fb)
        candidates.extend(_extract_all_json_array_substrings(fb))

    candidates.extend(_extract_all_json_array_substrings(s))

    unique_candidates = []
    seen = set()

    for c in candidates:
        c_norm = c.strip()
        if c_norm not in seen:
            seen.add(c_norm)
            unique_candidates.append(c_norm)

    return unique_candidates


def _get_label_and_box(item: Dict[str, Any]) -> tuple[Any, Any]:
    label = None
    box = None

    for key in ("class", "label", "category", "category_name"):
        if key in item:
            label = item[key]
            break

    for key in ("box", "box_2d", "bbox_2d", "bbox"):
        if key in item:
            box = item[key]
            break

    return label, box


def _clip_pixel_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_shape: tuple[int, int]
) -> tuple[int, int, int, int]:

    img_w, img_h = img_shape

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    return x1, y1, x2, y2


def _convert_raw_box_to_pixel_bbox(
        box: list[Any] | tuple[Any, ...],
        img_shape: tuple[int,int] = None,
        coords_mode: str = "absolute",
        norm_factor: int = NORM_SIZE,
        bbox_format: str = "x1y1x2y2",
) -> tuple[int, int, int, int]:
    if bbox_format == "y1x1y2x2":
        y1_temp, x1_temp, y2_temp, x2_temp = map(float, box)
    elif bbox_format == "x1y1x2y2":
        x1_temp, y1_temp, x2_temp, y2_temp = map(float, box)
    else:
        logging.error(
            "Unsupported bbox_format=%s. Falling back to x1y1x2y2.",
            bbox_format,
        )
        x1_temp, y1_temp, x2_temp, y2_temp = map(float, box)

    if coords_mode == "normalized":
        # This matches your previous behavior:
        # VLM coordinates are interpreted as independently normalized
        # over width and height in a 0..norm_factor space.

        if img_shape is None:
            raise ValueError("img_shape must be provided for normalized coordinates mode")
        img_w, img_h = img_shape

        x1 = int(round(x1_temp / norm_factor * (img_w - 1)))
        y1 = int(round(y1_temp / norm_factor * (img_h - 1)))
        x2 = int(round(x2_temp / norm_factor * (img_w - 1)))
        y2 = int(round(y2_temp / norm_factor * (img_h - 1)))

        return _clip_pixel_bbox(x1, y1, x2, y2)

    elif coords_mode == "absolute":
        x1 = int(round(x1_temp))
        y1 = int(round(y1_temp))
        x2 = int(round(x2_temp))
        y2 = int(round(y2_temp))

        return x1, y1, x2, y2

    else:
        raise ValueError(f"Unsupported coords_mode: {coords_mode}")



def _annotation_to_text_item(
        label_str: str,
        bbox: BoundingBox,
) -> dict[str, Any]:
    """
    Serialize the annotation back to a canonical text target.

    Here the bbox is represented in the schema coordinate space:
    normalized 1000x1000 square space.
    """
    return {
        "class": label_str,
        "bbox": [
            bbox.tl.x,
            bbox.tl.y,
            bbox.br.x,
            bbox.br.y,
        ],
    }

def parse_out_text_json_objects_to_target(
    text: str,
    img_size: Tuple[int, int],  # (width, height)
    active_fallback: bool = True,
    category_to_id: Optional[Dict[str, int]] = None,
    coords_mode: str = "absolute",  # "absolute" or "normalized"
    norm_factor: int = NORM_SIZE,
    bbox_format: str = "x1y1x2y2",  # "x1y1x2y2" or "y1x1y2x2"
    unknown_label_id: int = -1,
    source: str = "vlm_json",
) -> tuple[Target, Dict[str, Any]]:
    """
    Parse VLM text output containing JSON detections and convert it to the
    canonical VLM dataset schema.

    Expected examples:

        <answer>
        [
            {"label": "stator", "bbox_2d": [442, 456, 505, 570]},
            {"label": "rotor", "bbox_2d": [512, 452, 558, 538]}
        ]
        </answer>

    or:

        ```json
        [
            {"class": "stator", "box": [442, 456, 505, 570]}
        ]
        ```

    The returned Target contains:
        target.text      -> cleaned JSON string
        target.instances -> list[Annotation]

    The returned debug dictionary contains parse diagnostics.
    """

    if text is None:
        text = ""

    if category_to_id is None:
        category_to_id = {}

    img_w, img_h = img_size

    target = Target()

    debug: Dict[str, Any] = {
        "raw_text": text,
        "answer_blocks": [],
        "normalized_blocks": [],
        "parse_errors": [],
        "valid_sample": True,
        "has_detection": False,
        "used_fallback_after_channel": False,
        "used_fallback_after_think": False,
        "active_fallback": active_fallback,
        "coords_mode": coords_mode,
        "norm_factor": norm_factor,
        "bbox_format": bbox_format,
        "img_size_wh": (img_w, img_h),
    }

    answer_blocks = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    debug["answer_blocks"] = answer_blocks

    candidate_blocks: list[str] = []

    if answer_blocks:
        candidate_blocks.extend(answer_blocks)
    else:
        if active_fallback:
            after_channel = _extract_after_last_channel(text)
            if after_channel:
                candidate_blocks.append(after_channel)
                debug["used_fallback_after_channel"] = True

            after_think = _extract_after_last_think(text)
            if after_think:
                candidate_blocks.append(after_think)
                debug["used_fallback_after_think"] = True

        if not candidate_blocks:
            candidate_blocks.append(text)

    saw_explicit_empty = False
    saw_empty_json_prediction = False
    canonical_text_items: list[dict[str, Any]] = []

    for i, block in enumerate(candidate_blocks):
        block_stripped = block.strip()

        if block_stripped == "":
            saw_explicit_empty = True
            continue

        json_candidates = _extract_json_candidates(block_stripped)
        debug["normalized_blocks"].append(json_candidates)

        if not json_candidates:
            debug["parse_errors"].append(
                f"Could not find a JSON array in block {i}: {block!r}"
            )
            continue

        parsed = None
        chosen_candidate = None

        for candidate in json_candidates:
            try:
                maybe_parsed = _parse_json_list(candidate)
                if isinstance(maybe_parsed, list):
                    parsed = maybe_parsed
                    chosen_candidate = candidate
                    break
            except Exception as e:
                debug["parse_errors"].append(
                    f"Candidate parse failed in block {i}: {candidate!r} ({e})"
                )

        if parsed is None:
            debug["parse_errors"].append(
                f"Could not parse any JSON array candidate in block {i}: "
                f"{json_candidates!r}"
            )
            continue

        if parsed == []:
            saw_empty_json_prediction = True
            break

        for j, item in enumerate(parsed):
            if not isinstance(item, dict):
                debug["parse_errors"].append(
                    f"Item {j} in block {i} is not a dict: {item!r}"
                )
                continue

            label, box = _get_label_and_box(item)

            if label is None:
                debug["parse_errors"].append(
                    f"Item {j} in block {i} has no class/label/category field: "
                    f"{item!r}"
                )
                continue

            if box is None:
                debug["parse_errors"].append(
                    f"Item {j} in block {i} has no box/bbox_2d/bbox field: "
                    f"{item!r}"
                )
                continue

            if not isinstance(box, (list, tuple)) or len(box) != 4:
                debug["parse_errors"].append(
                    f"Invalid box in item {j} of block {i}: {box!r}"
                )
                continue

            try:
                x1, y1, x2, y2 = _convert_raw_box_to_pixel_bbox(
                    box,
                    img_shape=(img_w, img_h),
                    coords_mode=coords_mode,
                    norm_factor=norm_factor,
                )

                if x1 >= x2 or y1 >= y2:
                    debug["parse_errors"].append(
                        f"Degenerate bbox after conversion in item {j} of block {i}: "
                        f"{(x1, y1, x2, y2)} from raw box {box!r}"
                    )
                    continue

                bbox = BoundingBox(
                    tl=Point(x=x1, y=y1),
                    br=Point(x=x2, y=y2),
                )

                label_str = str(label).strip()
                label_id = category_to_id.get(label_str, unknown_label_id)

                ann = Annotation(
                    label=label_id,
                    bbox=bbox,
                    point=None,
                    mask=None,
                    source=source,
                )

                target.instances.append(ann)
                canonical_text_items.append(
                    _annotation_to_text_item(label_str=label_str, bbox=bbox)
                )

                debug.setdefault("parsed_items", []).append(
                    {
                        "answer_index": i,
                        "bbox_index_in_answer": j,
                        "label": label_str,
                        "label_id": label_id,
                        "raw_item": item,
                        "raw_box": box,
                        "pixel_bbox_xyxy": [x1, y1, x2, y2],
                        "canonical_bbox_xyxy": [
                            bbox.tl.x,
                            bbox.tl.y,
                            bbox.br.x,
                            bbox.br.y,
                        ],
                        "raw_json_candidate": chosen_candidate,
                    }
                )

            except Exception as e:
                debug["parse_errors"].append(
                    f"Error while converting item {j} in block {i}: "
                    f"{item!r} ({e})"
                )

        if target.instances or saw_empty_json_prediction:
            break

    if saw_empty_json_prediction:
        target.text = "[]"
    else:
        target.text = json.dumps(
            canonical_text_items,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    debug["num_instances"] = len(target.instances)
    debug["explicit_empty_answer"] = saw_explicit_empty
    debug["explicit_empty_json_prediction"] = saw_empty_json_prediction
    debug["has_detection"] = len(target.instances) > 0
    debug["valid_sample"] = (
        debug["has_detection"]
        or saw_explicit_empty
        or saw_empty_json_prediction
    )

    return target, debug