"""
QR Code Decoder Wrapper - Using decoder.py
"""
import numpy as np
import cv2
import sys

# Import từ decoder.py (import lại class QRDecode)
try:
    from decoder import QRDecode
except ImportError:
    # Fallback nếu không import được
    class QRDecode:
        def __init__(self, qr_code, debug=False, verbose=False):
            self.matrix = qr_code
            self.debug = debug
            self.verbose = verbose
        
        def decode(self):
            return ""  # Return empty string nếu lỗi


def get_nearest_qr_size(size):
    """Get nearest valid QR code size (21, 25, 29, 33, ..., 177)."""
    valid_sizes = [21 + 4*i for i in range(40)]  # QR versions 1-40
    # Find nearest size
    nearest = min(valid_sizes, key=lambda x: abs(x - size))
    return nearest


def decode_qr_content(img, qr_box):
    """
    Decode QR code content từ bounding box trên hình ảnh.
    Sử dụng QRDecode từ decoder.py.
    Returns decoded content hoặc empty string nếu fails.
    """
    try:
        # Validate coordinates are within image bounds
        h, w = img.shape[:2]
        x_coords = [qr_box["x0"], qr_box["x1"], qr_box["x2"], qr_box["x3"]]
        y_coords = [qr_box["y0"], qr_box["y1"], qr_box["y2"], qr_box["y3"]]
        
        # Check if all coordinates are valid (non-negative and within bounds)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        if min_x < 0 or max_x >= w or min_y < 0 or max_y >= h:
            # Invalid box coordinates - skip decoding
            return ""
        
        # Extract 4 corners
        pts_source = np.array([
            [qr_box["x0"], qr_box["y0"]],
            [qr_box["x1"], qr_box["y1"]],
            [qr_box["x2"], qr_box["y2"]],
            [qr_box["x3"], qr_box["y3"]],
        ], dtype=np.float32)
        
        # Estimate QR code size based on box corners
        # Use diagonal distance to estimate size
        diag1 = np.sqrt((qr_box["x2"] - qr_box["x0"])**2 + (qr_box["y2"] - qr_box["y0"])**2)
        diag2 = np.sqrt((qr_box["x3"] - qr_box["x1"])**2 + (qr_box["y3"] - qr_box["y1"])**2)
        estimated_size = int((diag1 + diag2) / 2 * 0.7)  # Rough estimate
        
        # Get nearest valid QR size
        target_size = get_nearest_qr_size(estimated_size)
        if target_size < 21:
            target_size = 21
        
        pts_dest = np.array([
            [0, 0], [target_size - 1, 0],
            [target_size - 1, target_size - 1], [0, target_size - 1],
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
        warped = cv2.warpPerspective(img, matrix, (target_size, target_size))
        
        if warped is None or warped.size == 0:
            return ""
        
        # Convert to grayscale
        if len(warped.shape) == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped.copy()
        
        # Threshold to binary (0 or 1)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert 0-255 to 0-1 binary matrix
        qr_matrix = (binary > 127).astype(np.uint8)
        
        # Attempt to decode using QRDecode
        try:
            decoder = QRDecode(qr_matrix, debug=False, verbose=False)
            content = decoder.decode()
            return content if content else ""
        except Exception as e:
            # Decoder failed, return empty - this is normal for non-standard QR codes
            return ""
    
    except Exception as e:
        return ""


def decode_all_qrs(img_path, qr_boxes):
    """
    Decode tất cả QR codes trong một ảnh.
    Returns list của QR boxes với 'content' field được populate.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return qr_boxes
        
        decoded_qrs = []
        for qr_box in qr_boxes:
            content = decode_qr_content(img, qr_box)
            qr_box_copy = qr_box.copy()
            qr_box_copy["content"] = content
            decoded_qrs.append(qr_box_copy)
        
        return decoded_qrs
    except Exception:
        return qr_boxes



