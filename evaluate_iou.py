"""
Advanced evaluation with IoU matching against ground truth.
Optional: for evaluating against ground_truth.csv if available.
"""
import csv
import sys
from pathlib import Path
from shapely.geometry import Polygon


def read_csv_results(csv_path):
    """Read results from CSV file."""
    results = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"].strip()
            qr_idx = row.get("qr_index", "").strip()
            
            if img_id not in results:
                results[img_id] = []
            
            if qr_idx:  # Has QR detection
                try:
                    coords = [
                        float(row["x0"].strip()),
                        float(row["y0"].strip()),
                        float(row["x1"].strip()),
                        float(row["y1"].strip()),
                        float(row["x2"].strip()),
                        float(row["y2"].strip()),
                        float(row["x3"].strip()),
                        float(row["y3"].strip()),
                    ]
                    content = row.get("content", "").strip()
                    
                    results[img_id].append({
                        "index": int(qr_idx),
                        "coords": coords,
                        "content": content,
                        "polygon": Polygon([(coords[i], coords[i+1]) for i in range(0, 8, 2)])
                    })
                except (ValueError, IndexError):
                    pass
    
    return results


def compute_iou(poly1, poly2):
    """Compute IoU between two polygons."""
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0
        return intersection / union
    except:
        return 0


def match_detections(pred, ground_truth, iou_threshold=0.5):
    """
    Match predictions with ground truth using Greedy IoU Matching.
    Returns TP, FP, FN counts.
    """
    tp, fp, fn = 0, 0, 0
    
    for img_id in set(list(pred.keys()) + list(ground_truth.keys())):
        pred_qrs = pred.get(img_id, [])
        gt_qrs = ground_truth.get(img_id, [])
        
        # Track matched GT indices per image
        matched_gt = set()
        
        # For each prediction, find best GT match
        for pred_qr in pred_qrs:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_qr in enumerate(gt_qrs):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_qr["polygon"], gt_qr["polygon"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        # Unmatched GTs are FN
        fn += len(gt_qrs) - len(matched_gt)
    
    return tp, fp, fn


def evaluate_with_gt(pred_csv, gt_csv, iou_threshold=0.5):
    """Evaluate predictions against ground truth."""
    print(f"\n{'='*60}")
    print(f"Evaluating with Ground Truth")
    print(f"{'='*60}\n")
    
    if not Path(gt_csv).exists():
        print(f"⚠️  Ground truth file not found: {gt_csv}")
        print("Run evaluation without ground truth.\n")
        return None
    
    print(f"Loading predictions from: {pred_csv}")
    pred = read_csv_results(pred_csv)
    
    print(f"Loading ground truth from: {gt_csv}")
    gt = read_csv_results(gt_csv)
    
    print(f"\nPredictions: {len(pred)} images, {sum(len(qrs) for qrs in pred.values())} QRs")
    print(f"Ground truth: {len(gt)} images, {sum(len(qrs) for qrs in gt.values())} QRs\n")
    
    # Match detections
    tp, fp, fn = match_detections(pred, gt, iou_threshold)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"IoU Threshold: {iou_threshold}")
    print(f"\nDetection Results:")
    print(f"  TP (True Positives):  {tp}")
    print(f"  FP (False Positives): {fp}")
    print(f"  FN (False Negatives): {fn}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print("="*60 + "\n")
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    pred_csv = "output.csv"
    
    # Check if output.csv exists
    if not Path(pred_csv).exists():
        print(f"❌ {pred_csv} not found")
        sys.exit(1)
    
    # Look for ground truth
    gt_candidates = [
        "../qr/output_valid.csv",
    ]
    
    gt_csv = None
    for candidate in gt_candidates:
        if Path(candidate).exists():
            gt_csv = candidate
            break
    
    if gt_csv:
        evaluate_with_gt(pred_csv, gt_csv)
    else:
        print("\n⚠️  Ground truth not found. Skipping IoU evaluation.")
        print("To evaluate with IoU metrics, provide ground_truth.csv\n")


if __name__ == "__main__":
    main()
