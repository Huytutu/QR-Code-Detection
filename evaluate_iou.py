"""
Evaluate output.csv against ground-truth CSV using the same greedy IoU logic
as described in qr/output_requirement.md.
"""

import argparse
import csv
import os
from pathlib import Path

from shapely.geometry import Polygon


def _normalize_polygon(points):
    """Create a valid polygon from 4-point quadrilateral coordinates."""
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if getattr(poly, "geom_type", "") == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area, default=None)
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return poly


def load_data(csv_path):
    """Load CSV and group polygons by image_id, preserving row order per image."""
    data = {}

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return {}

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_id",
            "qr_index",
            "x0",
            "y0",
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "content",
        }
        missing = required_columns.difference(set(reader.fieldnames or []))
        if missing:
            print(f"Error: missing required columns in {csv_path}: {sorted(missing)}")
            return {}

        for row in reader:
            img_id = (row.get("image_id") or "").strip()
            if not img_id:
                continue

            if img_id not in data:
                data[img_id] = []

            qr_index = (row.get("qr_index") or "").strip()
            if qr_index in ("", "-1"):
                continue

            try:
                points = [
                    (float(row["x0"].strip()), float(row["y0"].strip())),
                    (float(row["x1"].strip()), float(row["y1"].strip())),
                    (float(row["x2"].strip()), float(row["y2"].strip())),
                    (float(row["x3"].strip()), float(row["y3"].strip())),
                ]
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

            poly = _normalize_polygon(points)
            if poly is None:
                continue

            data[img_id].append(
                {
                    "polygon": poly,
                    "content": (row.get("content") or "").strip(),
                    "qr_index": qr_index,
                }
            )

    return data


def calculate_iou(poly1, poly2):
    """Compute IoU between two polygons."""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception:
        return 0.0


def evaluate(pred_file="output.csv", gt_file="../qr/output_valid.csv", iou_threshold=0.5):
    """Greedy IoU matching evaluation (per-image) with optional content accuracy."""
    print("Calculating metrics...")

    preds = load_data(pred_file)
    gts = load_data(gt_file)

    if not preds or not gts:
        print("No data to evaluate.")
        return None

    all_images = set(preds.keys()).union(set(gts.keys()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Optional content scoring over TP matches where GT content exists.
    content_compared = 0
    content_correct = 0

    for img_id in all_images:
        pred_items = preds.get(img_id, [])
        gt_items = gts.get(img_id, [])

        matched_gt_indices = set()

        # Keep prediction order from CSV (as in grader description).
        for pred_item in pred_items:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_item in enumerate(gt_items):
                if gt_idx in matched_gt_indices:
                    continue

                iou = calculate_iou(pred_item["polygon"], gt_item["polygon"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                matched_gt_indices.add(best_gt_idx)

                gt_content = (gt_items[best_gt_idx].get("content") or "").strip().lower()
                pred_content = (pred_item.get("content") or "").strip().lower()
                if gt_content:
                    content_compared += 1
                    if pred_content == gt_content:
                        content_correct += 1
            else:
                total_fp += 1

        total_fn += (len(gt_items) - len(matched_gt_indices))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Images evaluated        : {len(all_images)}")
    print(f"IoU threshold           : {iou_threshold}")
    print(f"TP (True Positives)     : {total_tp}")
    print(f"FP (False Positives)    : {total_fp}")
    print(f"FN (False Negatives)    : {total_fn}")
    print("-" * 60)
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1 Score                : {f1_score:.4f}")

    if content_compared > 0:
        content_accuracy = content_correct / content_compared
        print(f"Content Accuracy (opt.) : {content_accuracy:.4f} ({content_correct}/{content_compared})")
    else:
        print("Content Accuracy (opt.) : N/A (no GT content to compare)")

    print("=" * 60)

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "content_compared": content_compared,
        "content_correct": content_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate output.csv with greedy IoU matching")
    parser.add_argument("--pred", default="output.csv", help="Prediction CSV path")
    parser.add_argument("--gt", default="../qr/output_valid.csv", help="Ground-truth CSV path")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    args = parser.parse_args()

    if not Path(args.pred).exists():
        print(f"Error: prediction file not found: {args.pred}")
        return 1

    if not Path(args.gt).exists():
        print(f"Error: ground-truth file not found: {args.gt}")
        return 1

    result = evaluate(pred_file=args.pred, gt_file=args.gt, iou_threshold=args.iou)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
