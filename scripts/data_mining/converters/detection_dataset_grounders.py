#!/usr/bin/env python3
"""
Collection of dataset grounders that allow to iterate among different dataset format for object detction.
"""

import json
import random
from abc import ABC, abstractmethod
import os

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


# -------------------- SegmentationMask Dataset -------------------- [TO DO]
# Convert segmentation masks to bounding boxes annotations
# Given the image and masks folders, it is possible to couple them by name (same name, different folder)
# A {color_id:class} mapping dict is also given to map mask ids to class names. Mask can be both color or grayscale.
# Bbox are extracted from masks using connected components.