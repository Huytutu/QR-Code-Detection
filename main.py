#!/usr/bin/env python
"""
Main QR Code Detection and Decoding Pipeline
Usage:
    python main.py --data public_train.csv
    python main.py --data public_valid.csv --decode=yes
    python main.py --data private_test.csv --decode=no
"""
import sys
import os
import argparse
import time
import cv2
import numpy as np
from pipeline import detect_qr_codes
from decoder import QRCodeDecoder
from utils import get_image_paths, write_output_csv


def _order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def decode_all_qrs(img_path, qr_boxes):
    """Decode QR content for each detected box using decoder.py."""
    if not qr_boxes:
        return qr_boxes

    img = cv2.imread(img_path)
    if img is None:
        return qr_boxes

    h, w = img.shape[:2]
    decoder = QRCodeDecoder(trace_mode=False)
    decoded_qrs = []
    full_image_content = None
    full_image_checked = False

    for qr_box in qr_boxes:
        qr_box_copy = qr_box.copy()
        qr_box_copy["content"] = ""

        try:
            pts_source = np.array([
                [np.clip(qr_box["x0"], 0, w - 1), np.clip(qr_box["y0"], 0, h - 1)],
                [np.clip(qr_box["x1"], 0, w - 1), np.clip(qr_box["y1"], 0, h - 1)],
                [np.clip(qr_box["x2"], 0, w - 1), np.clip(qr_box["y2"], 0, h - 1)],
                [np.clip(qr_box["x3"], 0, w - 1), np.clip(qr_box["y3"], 0, h - 1)],
            ], dtype=np.float32)
            pts_source = _order_points(pts_source)

            side_lengths = [
                np.linalg.norm(pts_source[1] - pts_source[0]),
                np.linalg.norm(pts_source[2] - pts_source[1]),
                np.linalg.norm(pts_source[3] - pts_source[2]),
                np.linalg.norm(pts_source[0] - pts_source[3]),
            ]
            max_side = int(max(side_lengths)) if side_lengths else 0
            if max_side < 20:
                decoded_qrs.append(qr_box_copy)
                continue

            # Make a high-resolution square warp so finder patterns are preserved for the decoder.
            warp_size = int(np.clip(max(max_side * 2, 128), 128, 1024))
            pts_dest = np.array([
                [0, 0],
                [warp_size - 1, 0],
                [warp_size - 1, warp_size - 1],
                [0, warp_size - 1],
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
            warped = cv2.warpPerspective(img, matrix, (warp_size, warp_size))
            if warped is None or warped.size == 0:
                decoded_qrs.append(qr_box_copy)
                continue

            content = decoder.decode(warped)
            if (not content) and (len(qr_boxes) == 1) and (not full_image_checked):
                full_image_checked = True
                try:
                    full_image_content = decoder.decode(img)
                except Exception:
                    full_image_content = ""
                content = full_image_content

            if content:
                qr_box_copy["content"] = str(content).strip()
        except Exception:
            pass

        decoded_qrs.append(qr_box_copy)

    return decoded_qrs


def main():
    parser = argparse.ArgumentParser(description="QR Code Detection and Decoding")
    parser.add_argument("--data", required=True, help="Path to CSV file with image list")
    parser.add_argument("--decode", default="no", choices=["yes", "no"],
                       help="Whether to decode QR content (default: no)")
    
    args = parser.parse_args()
    
    csv_path = args.data
    should_decode = args.decode.lower() == "yes"
    
    # Validate input CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading images from: {csv_path}")
    image_list = get_image_paths(csv_path)
    
    if not image_list:
        print("Error: No images found in CSV file")
        sys.exit(1)
    
    print(f"Found {len(image_list)} images to process")
    print(f"Decode: {'ON' if should_decode else 'OFF'}")
    print("-" * 60)
    
    # Process each image
    all_results = []
    start_time = time.time()
    error_count = 0
    
    for idx, item in enumerate(image_list):
        img_id = item["image_id"]
        img_path = item["path"]
        
        result_dict = {
            "image_id": img_id,
            "qrs": []
        }
        
        try:
            # Detect QR codes
            if os.path.exists(img_path):
                qrs = detect_qr_codes(img_path)
                
                # Decode if requested
                if should_decode and qrs:
                    qrs = decode_all_qrs(img_path, qrs)
                
                result_dict["qrs"] = qrs
                
                if qrs:
                    print(f"[{idx+1}/{len(image_list)}] {img_id}: Found {len(qrs)} QR(s)")
                else:
                    print(f"[{idx+1}/{len(image_list)}] {img_id}: No QR detected")
            else:
                print(f"[{idx+1}/{len(image_list)}] {img_id}: Image file not found - {img_path}")
                error_count += 1
        except Exception as e:
            print(f"[{idx+1}/{len(image_list)}] {img_id}: Error - {type(e).__name__}: {e}")
            error_count += 1
        
        all_results.append(result_dict)
    
    # Calculate processing time
    process_time = time.time() - start_time
    avg_time = process_time / len(image_list) if len(image_list) > 0 else 0
    
    print("-" * 60)
    print(f"Total processing time: {process_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds")
    if error_count > 0:
        print(f"Images with errors: {error_count}")
    
    # Write output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.csv")
    print(f"\nWriting results to: {output_path}")
    
    try:
        write_output_csv(all_results, output_path)
        print("Done!")
        return 0
    except Exception as e:
        print(f"Error: Failed to write output file - {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
