#!/usr/bin/env python3
"""
Collection of dataset grounders that allow to iterate among different dataset format for object detction.
"""

import json
import random
from abc import ABC, abstractmethod
import os
import xml.etree.ElementTree as ET

class BaseWasteDataset(ABC):
    """
    Abstract base class for all waste detection datasets.
    Defines the common interface for conversion to JSONL training format.
    """

    def __init__(self, classes_dict, detailed_ratio=0.2):
        """
        Args:
            classes_dict (dict): taxonomy {class_name: [descriptions]}
            detailed_ratio (float): % of samples with detailed prompts
        """
        self.classes_dict = classes_dict
        self.detailed_ratio = detailed_ratio

    @abstractmethod
    def __iter__(self):
        """
        Must yield dicts with at least:
        {
          "image": "path/to/image.jpg",
          "width": int,
          "height": int,
          "annotations": [
              {"class": str, "bbox": [x, y, w, h]}
          ]
        }
        """
        pass

    def build_prompt(self, detailed=False):
        """Generate canonical or detailed prompt text."""
        base = (
            "You are an object detection assistant for waste sorting. "
            "Detect all waste items in the image and return a JSON list "
            "where each element has the format {class, box:[x1,y1,x2,y2] normalized 0..1}. "
        )

        if not detailed:
            cls_list = ", ".join(self.classes_dict.keys())
            prompt = f"{base}Use only the following classes: {cls_list}. " \
                     "If none of these classes are present, output an empty list []."
        else:
            prompt = base + "Use only the following classes:\n\n"
            for cname, descs in self.classes_dict.items():
                prompt += f"# {cname}\n"
                for d in descs:
                    prompt += f"- {d}\n"
                prompt += "\n"
            prompt += "If none of these classes are present, output an empty list []."
        return prompt

    def normalize_bbox(self, bbox, width, height):
        """Convert COCO-style bbox [x,y,w,h] to normalized [x1,y1,x2,y2]."""
        x, y, w, h = bbox
        x1, y1, x2, y2 = x / width, y / height, (x + w) / width, (y + h) / height
        return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

    def to_jsonl(self, out_file):
        """Export dataset to JSONL with prompts + JSON-string targets.
        image field: already includes any per-entry folder_path (self.img_root concatenated).
        image_root field: should point to the GLOBAL ROOT (if available) so loader can join
                          global_root + image (which already has folder_path) to form absolute path.
        If no global_root is set, we fallback to using img_root for backward compatibility.
        """
        records = []
        for sample in self:
            dets = []
            for ann in sample["annotations"]:
                box = self.normalize_bbox(ann["bbox"], sample["width"], sample["height"])
                dets.append({"class": ann["class"], "box": box})

            target_str = json.dumps(dets, indent=2)
            detailed = random.random() < self.detailed_ratio
            prompt = self.build_prompt(detailed=detailed)

            record = {
                "image": sample["image"],  # already folder_path + filename if img_root was set
                "prompt": prompt,
                "target": target_str
            }
            # Prefer global_root for image_root (new convention); fallback to img_root if not present
            global_root = getattr(self, 'global_root', None)
            img_root = getattr(self, 'img_root', None)
            if global_root is not None:
                record['image_root'] = global_root
            elif img_root is not None:
                record['image_root'] = img_root
            records.append(record)

        with open(out_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"✅ Saved {len(records)} samples to {out_file}")

# -------------------- COCO Dataset -------------------- [EXAMPLE]
class CocoWasteDataset(BaseWasteDataset):
    def __init__(self, coco_dict, classes_dict, detailed_ratio=0.2, img_root=None):
        """
        Args:
            coco_dict: COCO-style dict with images, annotations, categories
            classes_dict: mapping of class name -> descriptions
            detailed_ratio: fraction of samples with detailed prompts
            img_root: optional folder path to prepend to image file names. If not None,
                      it will be concatenated with the image file name (no absolute-path checks).
        """
        super().__init__(classes_dict, detailed_ratio)
        self.coco = coco_dict
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.coco.get("categories", [])}
        self.img_root = img_root

    def _resolve_image_path(self, file_name):
        if not file_name:
            return file_name
        if self.img_root is not None:
            # per user's instruction: always concat img_root if provided, do not try to resolve absolute paths
            return os.path.join(self.img_root, file_name)
        return file_name

    def __iter__(self):
        for img in self.coco.get("images", []):
            anns = [a for a in self.coco.get("annotations", []) if a.get("image_id") == img.get("id")]
            image_path = self._resolve_image_path(img.get("file_name", img.get("filename", "")))
            sample = {
                "image": image_path,
                "width": img.get("width"),
                "height": img.get("height"),
                "annotations": [
                    {"class": self.id_to_name.get(a.get("category_id")), "bbox": a.get("bbox")}
                    for a in anns
                ]
            }
            yield sample



# -------------------- Warp Dataset --------------------
# in warp dataset labels are given in a txt file where each line is formatted as: class_id x_center y_center width height (normalized 0..1)
# example: 4 0.627083 0.634722 0.183333 0.399074
#class ids are in [0, num_classes-1] and map to class names via a list of class names, num_classes = 27
# there is one txt file per image, with the same name but .txt extension

class WarpDataset(BaseWasteDataset):
    def __init__(self, images_dir, labels_dir, classes_dict, detailed_ratio=0.2, img_root=None):
        """Dataset for Warp-format labels (one .txt per image, YOLO-style normalized boxes).

        Args:
            images_dir (str): folder containing image files
            labels_dir (str): folder containing .txt label files (one per image)
            classes_dict (dict or list): if dict, keys are class names (order -> id mapping); if list, index->name mapping
            detailed_ratio (float): fraction of samples with detailed prompts
            img_root (str): optional path to prepend to image filename when writing JSONL (kept out of internal image lookup)
        """
        super().__init__(classes_dict if isinstance(classes_dict, dict) else {k: [] for k in (classes_dict or [])}, detailed_ratio)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_root = img_root

        # build id->name mapping
        if isinstance(classes_dict, dict):
            # preserve YAML ordering of keys (Python 3.7+ preserves insertion order)
            self.id_to_name = list(classes_dict.keys())
        elif isinstance(classes_dict, list):
            self.id_to_name = list(classes_dict)
        else:
            # fallback: unknown mapping, will use string ids
            self.id_to_name = []

    def _resolve_image_path(self, file_name):
        """Resolve image path for JSONL output. If img_root is provided we join img_root + file_name.
        Otherwise join images_dir + file_name."""
        if not file_name:
            return file_name
        if self.img_root is not None:
            return os.path.join(self.img_root, file_name)
        return os.path.join(self.images_dir, file_name)

    def __iter__(self):
        """Yield samples with the same structure expected by BaseWasteDataset.to_jsonl

        For each image in images_dir, look for a .txt file with the same base name in labels_dir.
        Label format (per line): class_id x_center y_center width height  (normalized 0..1)
        We convert normalized center format to COCO-style bbox [x, y, w, h] in pixel coordinates.
        """
        try:
            from PIL import Image
        except Exception:
            Image = None

        # list image files in images_dir
        if not os.path.isdir(self.images_dir):
            return

        for fname in sorted(os.listdir(self.images_dir)):
            # skip non-image files
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            img_path_on_disk = os.path.join(self.images_dir, fname)
            # obtain image size
            width = None
            height = None
            if Image is not None:
                try:
                    with Image.open(img_path_on_disk) as im:
                        width, height = im.size
                except Exception:
                    # skip unreadable images
                    continue
            else:
                # unable to get size without PIL — skip
                continue

            # read corresponding label file
            base = os.path.splitext(fname)[0]
            label_file = os.path.join(self.labels_dir, base + '.txt')
            annotations = []
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                cls_id = int(parts[0])
                                xc = float(parts[1])
                                yc = float(parts[2])
                                w_n = float(parts[3])
                                h_n = float(parts[4])
                            except Exception:
                                continue

                            # convert normalized center to pixel COCO bbox
                            w_px = w_n * width
                            h_px = h_n * height
                            x_px = xc * width - (w_px / 2.0)
                            y_px = yc * height - (h_px / 2.0)

                            # clip to image bounds
                            x_px = max(0.0, x_px)
                            y_px = max(0.0, y_px)
                            w_px = max(0.0, min(w_px, width - x_px))
                            h_px = max(0.0, min(h_px, height - y_px))

                            # resolve class name
                            cls_name = str(cls_id)
                            if 0 <= cls_id < len(self.id_to_name):
                                cls_name = self.id_to_name[cls_id]

                            annotations.append({"class": cls_name, "bbox": [x_px, y_px, w_px, h_px]})
                except Exception:
                    # if label reading fails, continue with empty annotations
                    annotations = []

            sample = {
                "image": self._resolve_image_path(fname),
                "width": width,
                "height": height,
                "annotations": annotations
            }
            yield sample


# -------------------- CVAT Dataset --------------------
class CVATDataset(BaseWasteDataset):
    """Grounder for CVAT XML annotations.

    Expected CVAT mapping:
    - one <image> element -> one yielded sample;
    - instance masks are read from <mask> elements;
    - bbox is extracted from the decoded CVAT mask RLE when possible,
      otherwise from left/top/width/height;
    - instance caption is the mask attribute named "description";
    - asset-level caption is the attribute "description" of the tag
      <tag label="asset_description">.

    The yielded annotations remain backward-compatible with the existing
    converter: at minimum they contain {"class", "bbox"}. Additional keys
    {"points", "caption", "mask", "attributes"} can be consumed by an
    extended canonical converter.
    """

    def __init__(self, xml_path, classes_dict=None, detailed_ratio=0.2, img_root=None,
                 include_masks=True, include_mask_rle=True):
        super().__init__(classes_dict or {}, detailed_ratio)
        self.xml_path = str(xml_path)
        self.img_root = img_root
        self.include_masks = include_masks
        self.include_mask_rle = include_mask_rle

    def _resolve_image_path(self, file_name):
        if not file_name:
            return file_name
        if self.img_root is not None:
            return os.path.join(self.img_root, file_name)
        return file_name

    @staticmethod
    def _clean_text(value):
        if value is None:
            return None
        value = str(value).strip()
        return value if value else None

    @classmethod
    def _attributes_dict(cls, node):
        attrs = {}
        for attr in node.findall("attribute"):
            name = cls._clean_text(attr.get("name"))
            value = cls._clean_text(attr.text)
            if name is not None and value is not None:
                attrs[name] = value
        return attrs

    @classmethod
    def _attribute_text(cls, node, name):
        return cls._attributes_dict(node).get(name)

    @classmethod
    def _asset_description(cls, image_el):
        for tag in image_el.findall("tag"):
            if tag.get("label") != "asset_description":
                continue
            # In the provided XML this is <attribute name="description">...</attribute>.
            desc = cls._attribute_text(tag, "description")
            if desc:
                return desc
            # Fallback for non-standard exports where the text is directly in the tag.
            desc = cls._clean_text(tag.text)
            if desc:
                return desc
        return None

    @staticmethod
    def _parse_rle(rle_text):
        if not rle_text:
            return []
        # CVAT exports comma-separated counts; split is tolerant to extra spaces.
        return [int(v) for v in str(rle_text).replace(",", " ").split() if v]

    @staticmethod
    def _clip_bbox_xywh(x, y, w, h, img_w, img_h):
        x = max(0.0, min(float(x), float(img_w)))
        y = max(0.0, min(float(y), float(img_h)))
        w = max(0.0, min(float(w), float(img_w) - x))
        h = max(0.0, min(float(h), float(img_h) - y))
        return [x, y, w, h]

    @classmethod
    def _bbox_and_centroid_from_cvat_rle(cls, *, rle_text, left, top, width, height,
                                        img_width, img_height):
        """Return (bbox_xywh, centroid_xy, foreground_area).

        CVAT XML mask RLE is interpreted as alternating background/foreground
        runs over the cropped mask rectangle, starting with background. This
        function does not allocate the full mask; it accumulates foreground
        coordinates directly from runs.
        """
        counts = cls._parse_rle(rle_text)
        crop_w = int(round(float(width)))
        crop_h = int(round(float(height)))
        left_i = int(round(float(left)))
        top_i = int(round(float(top)))

        fallback_bbox = cls._clip_bbox_xywh(left_i, top_i, crop_w, crop_h, img_width, img_height)
        fallback_point = [round(left_i + crop_w / 2), round(top_i + crop_h / 2)]

        if crop_w <= 0 or crop_h <= 0 or not counts:
            return fallback_bbox, fallback_point, 0

        expected = crop_w * crop_h
        if sum(counts) != expected:
            # If the RLE is malformed, keep the conservative CVAT rectangle.
            return fallback_bbox, fallback_point, 0

        min_x = crop_w
        min_y = crop_h
        max_x = -1
        max_y = -1
        sum_x = 0.0
        sum_y = 0.0
        area = 0

        pos = 0
        foreground = False
        for run in counts:
            run = int(run)
            if run <= 0:
                foreground = not foreground
                continue

            if foreground:
                end = pos + run
                p = pos
                while p < end:
                    y = p // crop_w
                    x0 = p % crop_w
                    n = min(end - p, crop_w - x0)
                    x1 = x0 + n - 1

                    min_x = min(min_x, x0)
                    max_x = max(max_x, x1)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    sum_x += n * (x0 + x1) / 2.0
                    sum_y += n * y
                    area += n
                    p += n

            pos += run
            foreground = not foreground

        if area <= 0:
            return fallback_bbox, fallback_point, 0

        x = left_i + min_x
        y = top_i + min_y
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        bbox = cls._clip_bbox_xywh(x, y, w, h, img_width, img_height)
        centroid = [round(left_i + sum_x / area), round(top_i + sum_y / area)]
        return bbox, centroid, area

    @staticmethod
    def _bbox_and_centroid_from_box(box_el, img_width, img_height):
        xtl = float(box_el.get("xtl"))
        ytl = float(box_el.get("ytl"))
        xbr = float(box_el.get("xbr"))
        ybr = float(box_el.get("ybr"))
        x = min(xtl, xbr)
        y = min(ytl, ybr)
        w = abs(xbr - xtl)
        h = abs(ybr - ytl)
        x, y, w, h = CVATDataset._clip_bbox_xywh(x, y, w, h, img_width, img_height)
        return [x, y, w, h], [round(x + w / 2), round(y + h / 2)]

    @staticmethod
    def _parse_points(points_text):
        pts = []
        for token in str(points_text or "").split(";"):
            token = token.strip()
            if not token:
                continue
            x, y = token.split(",")[:2]
            pts.append((float(x), float(y)))
        return pts

    @classmethod
    def _bbox_and_centroid_from_points(cls, points_text, img_width, img_height):
        pts = cls._parse_points(points_text)
        if not pts:
            return None, None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        bbox = cls._clip_bbox_xywh(x, y, w, h, img_width, img_height)
        centroid = [round(sum(xs) / len(xs)), round(sum(ys) / len(ys))]
        return bbox, centroid

    def __iter__(self):
        root = ET.parse(self.xml_path).getroot()

        for image_el in root.findall("image"):
            image_name = image_el.get("name", "")
            width = int(float(image_el.get("width")))
            height = int(float(image_el.get("height")))
            asset_caption = self._asset_description(image_el)

            annotations = []

            # Main path for your dataset: instance masks.
            for mask_el in image_el.findall("mask"):
                label = self._clean_text(mask_el.get("label"))
                if label is None:
                    continue

                bbox, centroid, area = self._bbox_and_centroid_from_cvat_rle(
                    rle_text=mask_el.get("rle"),
                    left=mask_el.get("left", 0),
                    top=mask_el.get("top", 0),
                    width=mask_el.get("width", 0),
                    height=mask_el.get("height", 0),
                    img_width=width,
                    img_height=height,
                )
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                attrs = self._attributes_dict(mask_el)
                mask_payload = None
                if self.include_masks:
                    mask_payload = {
                        "format": "cvat_rle",
                        "left": int(round(float(mask_el.get("left", 0)))),
                        "top": int(round(float(mask_el.get("top", 0)))),
                        "width": int(round(float(mask_el.get("width", 0)))),
                        "height": int(round(float(mask_el.get("height", 0)))),
                    }
                    if self.include_mask_rle:
                        mask_payload["rle"] = mask_el.get("rle", "")

                annotations.append({
                    "class": label,
                    "bbox": bbox,
                    "points": [{"x": int(centroid[0]), "y": int(centroid[1]), "is_positive": True}],
                    "caption": attrs.get("description"),
                    "mask": mask_payload,
                    "attributes": {
                        **attrs,
                        "cvat_shape_type": "mask",
                        "cvat_source": mask_el.get("source", ""),
                        "cvat_occluded": mask_el.get("occluded", "0"),
                        "cvat_z_order": mask_el.get("z_order", "0"),
                        "mask_area_px": str(area),
                    },
                })

            # Optional fallback: CVAT box annotations, useful if a future export contains them.
            for box_el in image_el.findall("box"):
                label = self._clean_text(box_el.get("label"))
                if label is None:
                    continue
                bbox, centroid = self._bbox_and_centroid_from_box(box_el, width, height)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                attrs = self._attributes_dict(box_el)
                annotations.append({
                    "class": label,
                    "bbox": bbox,
                    "points": [{"x": int(centroid[0]), "y": int(centroid[1]), "is_positive": True}],
                    "caption": attrs.get("description"),
                    "mask": None,
                    "attributes": {**attrs, "cvat_shape_type": "box"},
                })

            # Optional fallback: polygons are converted to a bbox and vertex-average point.
            for poly_el in image_el.findall("polygon"):
                label = self._clean_text(poly_el.get("label"))
                if label is None:
                    continue
                bbox, centroid = self._bbox_and_centroid_from_points(poly_el.get("points"), width, height)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                attrs = self._attributes_dict(poly_el)
                annotations.append({
                    "class": label,
                    "bbox": bbox,
                    "points": [{"x": int(centroid[0]), "y": int(centroid[1]), "is_positive": True}],
                    "caption": attrs.get("description"),
                    "mask": None,
                    "attributes": {
                        **attrs,
                        "cvat_shape_type": "polygon",
                        "cvat_points": poly_el.get("points", ""),
                    },
                })

            sample = {
                "image": self._resolve_image_path(image_name),
                "width": width,
                "height": height,
                "annotations": annotations,
            }
            if asset_caption is not None:
                sample["caption"] = asset_caption
                sample["asset_caption"] = asset_caption

            yield sample


# -------------------- SegmentationMask Dataset -------------------- [TO DO]
# Convert segmentation masks to bounding boxes annotations
# Given the image and masks folders, it is possible to couple them by name (same name, different folder)
# A {color_id:class} mapping dict is also given to map mask ids to class names. Mask can be both color or grayscale.
# Bbox are extracted from masks using connected components.