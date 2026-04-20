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
from pipeline import detect_qr_codes
from decoder_wrapper import decode_all_qrs
from utils import get_image_paths, write_output_csv


def main():
    parser = argparse.ArgumentParser(description="QR Code Detection and Decoding")
    parser.add_argument("--data", required=True, help="Path to CSV file with image list")
    parser.add_argument("--decode", default="yes", choices=["yes", "no"],
                       help="Whether to decode QR content (default: yes)")
    
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
    output_path = "output.csv"
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
