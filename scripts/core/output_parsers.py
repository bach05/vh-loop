from __future__ import annotations

"""Small output parsers for canonical_schema v2 model answers.

The parser layer is intentionally simple:

- one public entry point: parse_model_output(...)
- one parser function per answer_format
- parser functions return InstanceAnnotation objects in pixel coordinates

Current supported formats
-------------------------

tag_bbox_list
    Compact bbox tags separated by semicolons:

        <class_name,x1,y1,x2,y2>;<class_name,x1,y1,x2,y2>

    The parser also accepts a pipe-separated variant:

        <class_name|x1|y1|x2|y2>

    Empty prediction:

        <none>

json_bbox_list
    Optional lightweight JSON support for future use. Accepts either:

        [{"class":"PET_bottle","box":[10,20,30,40]}]

    or:

        [["PET_bottle",10,20,30,40]]

Coordinates are interpreted as normalized by default, because the current
SISimpleDataSample emits coordinates normalized by MessageBuildInfo.normalization_factor.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from scripts.data.canonical_schema.annotations import InstanceAnnotation
from scripts.data.canonical_schema.geometry import BoundingBox, Point

AnswerFormat = Literal["tag_bbox_list", "json_bbox_list", "text"]
CoordsMode = Literal["normalized", "pixel"]


@dataclass
class ParseResult:
    """Result returned by parse_model_output."""

    annotations: list[InstanceAnnotation] = field(default_factory=list)
    cleaned_text: str = ""
    answer_format: str = ""
    valid: bool = False
    has_detection: bool = False
    is_empty: bool = False
    errors: list[str] = field(default_factory=list)
    parsed_items: list[dict[str, Any]] = field(default_factory=list)

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "cleaned_text": self.cleaned_text,
            "answer_format": self.answer_format,
            "valid": self.valid,
            "has_detection": self.has_detection,
            "is_empty": self.is_empty,
            "num_instances": len(self.annotations),
            "errors": self.errors,
            "parsed_items": self.parsed_items,
        }


def clean_model_text(text: str | None) -> str:
    """Remove common wrappers while keeping the function intentionally small."""

    if text is None:
        return ""

    s = str(text).strip()

    # Keep only content after final reasoning block if present.
    if "</think>" in s.lower():
        parts = re.split(r"</think>", s, flags=re.IGNORECASE)
        s = parts[-1].strip()

    # Keep only content after final channel marker if present.
    if "<channel|>" in s.lower():
        parts = re.split(r"<channel\|>", s, flags=re.IGNORECASE)
        s = parts[-1].strip()
        s = re.split(r"<turn\|>", s, flags=re.IGNORECASE)[0].strip()

    # Prefer explicit answer tags.
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
    if answer_blocks:
        s = answer_blocks[-1].strip()

    # Strip a single outer code fence.
    m = re.match(r"^```(?:json|text)?\s*\n?(.*?)\n?```$", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    # Unwrap quoted string literals, useful when the generation is itself serialized.
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, str):
            s = parsed.strip()
    except Exception:
        pass

    return s.strip()


def _lookup_label_id(
    label_name: str,
    label_name_to_id: Mapping[str, int] | None,
    *,
    unknown_label_id: int,
    strict_labels: bool,
) -> int:
    if label_name_to_id is None:
        return unknown_label_id

    if label_name in label_name_to_id:
        return int(label_name_to_id[label_name])

    if strict_labels:
        raise ValueError(f"Unknown label name: {label_name!r}")

    return unknown_label_id


def _clip_xyxy_exclusive(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    img_width: int,
    img_height: int,
    reorder: bool = True,
) -> tuple[int, int, int, int]:
    """Clip xyxy bbox to image bounds using exclusive bottom-right corner."""

    if reorder:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

    # tl must lie inside the image. br is exclusive and can be equal to width/height.
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(1, min(x2, img_width))
    y2 = max(1, min(y2, img_height))

    return x1, y1, x2, y2


def _coords_to_pixel_bbox(
    coords: Sequence[Any],
    *,
    img_size: tuple[int, int],
    coords_mode: CoordsMode,
    norm_factor: int,
) -> tuple[int, int, int, int]:
    if len(coords) != 4:
        raise ValueError(f"Expected 4 bbox coordinates, got {coords!r}")

    img_width, img_height = img_size
    x1, y1, x2, y2 = [float(v) for v in coords]

    if coords_mode == "normalized":
        # The training target is produced by x_pixel / image_width * norm_factor.
        # Invert with image_width/image_height, not width-1/height-1, because bbox.br
        # follows the exclusive [tl, br) convention.
        x1 = x1 / norm_factor * img_width
        y1 = y1 / norm_factor * img_height
        x2 = x2 / norm_factor * img_width
        y2 = y2 / norm_factor * img_height
    elif coords_mode != "pixel":
        raise ValueError(f"Unsupported coords_mode={coords_mode!r}")

    return _clip_xyxy_exclusive(
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        img_width=img_width,
        img_height=img_height,
    )


def _make_annotation(
    *,
    label_name: str,
    coords: Sequence[Any],
    img_size: tuple[int, int],
    label_name_to_id: Mapping[str, int] | None,
    unknown_label_id: int,
    strict_labels: bool,
    coords_mode: CoordsMode,
    norm_factor: int,
    instance_index: int,
) -> InstanceAnnotation:
    label_name = label_name.strip()
    if not label_name:
        raise ValueError("Empty label name")

    x1, y1, x2, y2 = _coords_to_pixel_bbox(
        coords,
        img_size=img_size,
        coords_mode=coords_mode,
        norm_factor=norm_factor,
    )

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Degenerate bbox after conversion: {(x1, y1, x2, y2)}")

    label_id = _lookup_label_id(
        label_name,
        label_name_to_id,
        unknown_label_id=unknown_label_id,
        strict_labels=strict_labels,
    )

    return InstanceAnnotation(
        instance_id=f"pred_{instance_index:04d}",
        label_id=label_id,
        label_name=label_name,
        bbox=BoundingBox(tl=Point(x=x1, y=y1), br=Point(x=x2, y=y2)),
        points=None,
        mask=None,
        caption=None,
        attributes={},
    )


# <class_name,x1,y1,x2,y2> or <class_name|x1|y1|x2|y2>
_TAG_BBOX_RE = re.compile(
    r"<\s*([A-Za-z0-9_.:-]+)\s*[,|]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,|]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,|]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,|]\s*"
    r"(-?\d+(?:\.\d+)?)\s*>",
)


def parse_tag_bbox_list(
    text: str,
    *,
    img_size: tuple[int, int],
    label_name_to_id: Mapping[str, int] | None = None,
    coords_mode: CoordsMode = "normalized",
    norm_factor: int = 1000,
    unknown_label_id: int = -1,
    strict_labels: bool = False,
) -> ParseResult:
    cleaned = clean_model_text(text)
    result = ParseResult(cleaned_text=cleaned, answer_format="tag_bbox_list")

    if cleaned == "" or cleaned.lower() in {"<none>", "none", "[]"}:
        result.valid = True
        result.is_empty = True
        return result

    matches = list(_TAG_BBOX_RE.finditer(cleaned))
    if not matches:
        result.errors.append("No tag bbox items found. Expected '<class_name,x1,y1,x2,y2>'.")
        return result

    for idx, match in enumerate(matches):
        label_name = match.group(1)
        raw_coords = match.groups()[1:]
        try:
            ann = _make_annotation(
                label_name=label_name,
                coords=raw_coords,
                img_size=img_size,
                label_name_to_id=label_name_to_id,
                unknown_label_id=unknown_label_id,
                strict_labels=strict_labels,
                coords_mode=coords_mode,
                norm_factor=norm_factor,
                instance_index=idx,
            )
            result.annotations.append(ann)
            result.parsed_items.append(
                {
                    "label_name": ann.label_name,
                    "label_id": ann.label_id,
                    "raw_coords": list(raw_coords),
                    "pixel_bbox_xyxy": [
                        ann.bbox.tl.x,
                        ann.bbox.tl.y,
                        ann.bbox.br.x,
                        ann.bbox.br.y,
                    ] if ann.bbox is not None else None,
                }
            )
        except Exception as exc:
            result.errors.append(f"Could not parse tag item {idx}: {match.group(0)!r} ({exc})")

    result.has_detection = len(result.annotations) > 0
    result.valid = result.has_detection or not result.errors
    return result


def _extract_json_array(text: str) -> str:
    cleaned = clean_model_text(text)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    # Small fallback: find the first [...] block. This is enough for noisy wrappers
    # without reintroducing the previous large extraction stack.
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON array found")
    return m.group(0)


def parse_json_bbox_list(
    text: str,
    *,
    img_size: tuple[int, int],
    label_name_to_id: Mapping[str, int] | None = None,
    coords_mode: CoordsMode = "normalized",
    norm_factor: int = 1000,
    unknown_label_id: int = -1,
    strict_labels: bool = False,
) -> ParseResult:
    cleaned = clean_model_text(text)
    result = ParseResult(cleaned_text=cleaned, answer_format="json_bbox_list")

    if cleaned == "" or cleaned.lower() in {"<none>", "none"}:
        result.valid = True
        result.is_empty = True
        return result

    try:
        raw = _extract_json_array(cleaned)
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = ast.literal_eval(_extract_json_array(cleaned))
        except Exception as exc:
            result.errors.append(f"Could not parse JSON bbox list: {exc}")
            return result

    if parsed == []:
        result.valid = True
        result.is_empty = True
        return result

    if not isinstance(parsed, list):
        result.errors.append(f"Expected list, got {type(parsed).__name__}")
        return result

    for idx, item in enumerate(parsed):
        try:
            if isinstance(item, dict):
                label = item.get("class") or item.get("label") or item.get("category")
                coords = item.get("box") or item.get("bbox") or item.get("bbox_2d")
            elif isinstance(item, (list, tuple)) and len(item) == 5:
                label, *coords = item
            else:
                raise ValueError(f"Unsupported item shape: {item!r}")

            if label is None or coords is None:
                raise ValueError(f"Missing label or bbox in item: {item!r}")

            ann = _make_annotation(
                label_name=str(label),
                coords=coords,
                img_size=img_size,
                label_name_to_id=label_name_to_id,
                unknown_label_id=unknown_label_id,
                strict_labels=strict_labels,
                coords_mode=coords_mode,
                norm_factor=norm_factor,
                instance_index=idx,
            )
            result.annotations.append(ann)
            result.parsed_items.append(
                {
                    "label_name": ann.label_name,
                    "label_id": ann.label_id,
                    "raw_item": item,
                    "pixel_bbox_xyxy": [
                        ann.bbox.tl.x,
                        ann.bbox.tl.y,
                        ann.bbox.br.x,
                        ann.bbox.br.y,
                    ] if ann.bbox is not None else None,
                }
            )
        except Exception as exc:
            result.errors.append(f"Could not parse JSON item {idx}: {exc}")

    result.has_detection = len(result.annotations) > 0
    result.valid = result.has_detection or not result.errors
    return result


def parse_text_output(text: str, **_: Any) -> ParseResult:
    """Generic text parser for pretraining-style answers with no geometry."""

    cleaned = clean_model_text(text)
    return ParseResult(
        annotations=[],
        cleaned_text=cleaned,
        answer_format="text",
        valid=bool(cleaned),
        has_detection=False,
        is_empty=not bool(cleaned),
    )


PARSERS: dict[str, Callable[..., ParseResult]] = {
    "tag_bbox_list": parse_tag_bbox_list,
    "json_bbox_list": parse_json_bbox_list,
    "text": parse_text_output,
}


def model_output_parsing(
    text: str,
    *,
    img_size: tuple[int, int],
    answer_format: AnswerFormat = "tag_bbox_list",
    label_name_to_id: Mapping[str, int] | None = None,
    coords_mode: CoordsMode = "normalized",
    norm_factor: int = 1000,
    unknown_label_id: int = -1,
    strict_labels: bool = False,
) -> ParseResult:
    """Parse a model answer according to answer_format.

    Parameters
    ----------
    text:
        Raw model output.
    img_size:
        Image size as (width, height).
    answer_format:
        Parser key. Current primary format is "tag_bbox_list".
    label_name_to_id:
        Optional mapping from canonical class token to label id.
    coords_mode:
        "normalized" for normalized VLM answers; "pixel" for raw pixel outputs.
    norm_factor:
        Coordinate normalization factor used during target generation.
    unknown_label_id:
        Label id to use for unknown labels when strict_labels=False.
    strict_labels:
        If True, unknown labels raise parsing errors. If False, they receive
        unknown_label_id.
    """

    parser = PARSERS.get(answer_format)
    if parser is None:
        raise ValueError(f"Unsupported answer_format={answer_format!r}. Available: {sorted(PARSERS)}")

    return parser(
        text,
        img_size=img_size,
        label_name_to_id=label_name_to_id,
        coords_mode=coords_mode,
        norm_factor=norm_factor,
        unknown_label_id=unknown_label_id,
        strict_labels=strict_labels,
    )

#
# # Backward-compatible convenience alias for the current format.
# def parse_tag_bbox_output(
#     text: str,
#     *,
#     img_size: tuple[int, int],
#     label_name_to_id: Mapping[str, int] | None = None,
#     norm_factor: int = 1000,
#     strict_labels: bool = False,
# ) -> ParseResult:
#     return parse_model_output(
#         text,
#         img_size=img_size,
#         answer_format="tag_bbox_list",
#         label_name_to_id=label_name_to_id,
#         coords_mode="normalized",
#         norm_factor=norm_factor,
#         strict_labels=strict_labels,
#     )