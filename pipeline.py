"""
QR Code Detection Pipeline with PCA Enhancement
Synchronized with: pipeline copy 4.ipynb + _insert_pca_cell.ps1
"""
import cv2
import numpy as np
import os
from itertools import combinations
from utils import is_valid_qr_triangle, get_image_paths, write_output_csv


# ============================================================================
# PCA MODEL LOADING & DISTANCE CALCULATION
# ============================================================================

_PCA_MODEL_CACHE = None


def _load_pca_model():
    """Load PCA model from cache or disk."""
    global _PCA_MODEL_CACHE
    if _PCA_MODEL_CACHE is not None:
        return _PCA_MODEL_CACHE
    
    model_path = "pca_model.npz"
    if not os.path.exists(model_path):
        return None
    
    try:
        data = np.load(model_path, allow_pickle=False)
        if not all(k in data.files for k in ["mean", "eigenvectors", "ideal_coeffs", "threshold"]):
            return None
        
        _PCA_MODEL_CACHE = {
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "eigenvectors": np.asarray(data["eigenvectors"], dtype=np.float32),
            "ideal_coeffs": np.asarray(data["ideal_coeffs"], dtype=np.float32).reshape(1, -1),
            "threshold": float(np.asarray(data["threshold"]).reshape(-1)[0]),
        }
        return _PCA_MODEL_CACHE
    except Exception:
        return None


def _pca_distance_from_box(source_img, qr_box, size=20, model=None):
    """Compute PCA distance for a QR candidate box."""
    if model is None:
        model = _load_pca_model()
    if model is None:
        return None
    
    try:
        pts_source = np.array([
            [qr_box["x0"], qr_box["y0"]],
            [qr_box["x1"], qr_box["y1"]],
            [qr_box["x2"], qr_box["y2"]],
            [qr_box["x3"], qr_box["y3"]],
        ], dtype=np.float32)
        
        pts_dest = np.array([
            [0, 0], [size - 1, 0],
            [size - 1, size - 1], [0, size - 1],
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
        warped = cv2.warpPerspective(source_img, matrix, (size, size))
        
        if warped is None or warped.size == 0:
            return None
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
        _, roi_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        vec = roi_bin.astype(np.float32).reshape(1, -1)
        coeffs = cv2.PCAProject(vec, model["mean"], model["eigenvectors"])
        
        return float(np.linalg.norm(coeffs - model["ideal_coeffs"]))
    except Exception:
        return None


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(img):
    """Preprocess image for QR detection using Scharr gradient approach."""
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scharr-based gradient (not simple binary threshold)
    gradX = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gradY = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradX, gradY)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fine mask
    kernel_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_fine = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_fine)
    mask_fine = cv2.erode(mask_fine, None, iterations=2)
    mask_fine = cv2.dilate(mask_fine, None, iterations=2)
    
    # Coarse mask
    kernel_coarse = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    mask_coarse = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_coarse)
    
    return mask_fine, mask_coarse, thresh


def preprocessing_for_FP(image):
    """Preprocess image for finder pattern detection with adaptive thresholding."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    median_blurred = cv2.medianBlur(gray_img, 3)
    
    # Adaptive threshold with blockSize=61, C=1 (from notebook)
    binary_img = cv2.adaptiveThreshold(
        median_blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=61,
        C=1,
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    return morph_img


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def order_points(pts):
    """Order 4 corner points in standard format (TL, TR, BR, BL)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ============================================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================================

def apply_nms(qrs_list, distance_threshold=20):
    """Remove duplicate QR detections by NMS, keeping highest solidity."""
    if len(qrs_list) == 0:
        return []
    
    keep = []
    boxes_info = []
    
    for qr in qrs_list:
        cx = (qr['x0'] + qr['x2']) / 2
        cy = (qr['y0'] + qr['y2']) / 2
        solidity = qr.get('solidity', 0.0)
        boxes_info.append({'qr': qr, 'cx': cx, 'cy': cy, 'solidity': solidity})
    
    boxes_info = sorted(boxes_info, key=lambda k: k['solidity'], reverse=True)
    
    for i in range(len(boxes_info)):
        box1 = boxes_info[i]
        should_keep = True
        
        for kept_box in keep:
            dist = np.sqrt((box1['cx'] - kept_box['cx']) ** 2 + (box1['cy'] - kept_box['cy']) ** 2)
            if dist < distance_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(box1)
    
    return [item['qr'] for item in keep]


# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_qr_soft(img, qr_box, mode="default", pca_model=None):
    """Verify if a box is a valid QR code using visual characteristics + PCA."""
    try:
        # Extract 4 corners
        pts_source = np.array([
            [qr_box['x0'], qr_box['y0']],
            [qr_box['x1'], qr_box['y1']],
            [qr_box['x2'], qr_box['y2']],
            [qr_box['x3'], qr_box['y3']]
        ], dtype="float32")
        
        src_area = abs(cv2.contourArea(pts_source))
        
        # Normalize to 100x100
        size = 100
        pts_dest = np.array([
            [0, 0], [size - 1, 0],
            [size - 1, size - 1], [0, size - 1]
        ], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
        warped = cv2.warpPerspective(img, matrix, (size, size))
        
        # Convert to binary
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Check 1: White ratio balance
        white_pixels = cv2.countNonZero(binary)
        total_pixels = size * size
        white_ratio = white_pixels / total_pixels
        
        low_white, high_white = 0.18, 0.82
        if src_area < 500:
            low_white, high_white = 0.16, 0.86
        
        if white_ratio < low_white or white_ratio > high_white:
            return False
        
        # Check 2: Transitions (changes in binary pattern)
        mid_row = binary[size // 2, :]
        mid_col = binary[:, size // 2]
        
        row_transitions = np.sum(np.abs(np.diff(mid_row)) > 0)
        col_transitions = np.sum(np.abs(np.diff(mid_col)) > 0)
        
        transition_min = 7
        if src_area < 800:
            transition_min = 5
        if src_area < 350:
            transition_min = 4
        
        if row_transitions < transition_min or col_transitions < transition_min:
            if mode == "dense_small" and src_area < 900:
                relaxed_ok = (row_transitions + col_transitions >= 11) and (min(row_transitions, col_transitions) >= 2)
                if not relaxed_ok:
                    return False
            else:
                return False
        
        # Check 3: PCA-based gating
        model = pca_model if pca_model is not None else _load_pca_model()
        if model is not None:
            d = _pca_distance_from_box(img, qr_box, model=model)
            if d is not None:
                limit = model["threshold"] * (1.10 if mode == "dense_small" else 0.95)
                if d > limit:
                    return False
        
        return True
    except Exception:
        return False


# ============================================================================
# BOUNDING BOX EXTRACTION
# ============================================================================

def get_small_bounding_boxes(mask, min_solidity=0.9, min_area=350, aspect_ratio_threshold=2.2):
    """Extract small QR bounding boxes from mask."""
    raw_qrs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue
        
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w < 18 or h < 18:
            continue
        
        aspect_ratio = max(w, h) / min(w, h)
        
        is_normal_qr = (aspect_ratio <= aspect_ratio_threshold) and (solidity >= min_solidity)
        is_skewed_qr = (aspect_ratio_threshold < aspect_ratio <= 4.9) and (solidity >= 0.9)
        
        if is_normal_qr or is_skewed_qr:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            ordered_box = order_points(box)
            
            pad_w = int(w * 0.061)
            pad_h = int(h * 0.061)
            pad = max(pad_w, pad_h)
            
            raw_qrs.append({
                "x0": float(ordered_box[0][0] - pad), "y0": float(ordered_box[0][1] - pad),
                "x1": float(ordered_box[1][0] + pad), "y1": float(ordered_box[1][1] - pad),
                "x2": float(ordered_box[2][0] + pad), "y2": float(ordered_box[2][1] + pad),
                "x3": float(ordered_box[3][0] - pad), "y3": float(ordered_box[3][1] + pad),
                "content": "",
                "solidity": solidity
            })
    
    final_qrs = apply_nms(raw_qrs, distance_threshold=18)
    return final_qrs


def get_big_bounding_boxes(mask, min_solidity=0.81, min_area=9250, aspect_ratio_threshold=2.9):
    """Extract large QR bounding boxes from mask."""
    raw_qrs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue
        
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w < 18 or h < 18:
            continue
        
        aspect_ratio = max(w, h) / min(w, h)
        
        is_normal_qr = (aspect_ratio <= aspect_ratio_threshold) and (solidity >= min_solidity)
        is_skewed_qr = (aspect_ratio_threshold < aspect_ratio <= 4.9) and (solidity >= 0.9)
        
        if is_normal_qr or is_skewed_qr:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            ordered_box = order_points(box)
            
            pad_w = int(w * 0.061)
            pad_h = int(h * 0.061)
            pad = max(pad_w, pad_h)
            
            raw_qrs.append({
                "x0": float(ordered_box[0][0] - pad), "y0": float(ordered_box[0][1] - pad),
                "x1": float(ordered_box[1][0] + pad), "y1": float(ordered_box[1][1] - pad),
                "x2": float(ordered_box[2][0] + pad), "y2": float(ordered_box[2][1] + pad),
                "x3": float(ordered_box[3][0] - pad), "y3": float(ordered_box[3][1] + pad),
                "content": "",
                "solidity": solidity
            })
    
    final_qrs = apply_nms(raw_qrs, distance_threshold=18)
    return final_qrs


# ============================================================================
# FINDER PATTERN DETECTION (Complex, with multiple fallback modes)
# ============================================================================

def get_finder_patterns(work_mask, min_area=240):
    """Detect QR codes from finder patterns (3 corner squares + fallbacks)."""
    final_qrs = []
    
    def _rect_iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0
    
    def _quad_to_rect(q):
        xs = [q["x0"], q["x1"], q["x2"], q["x3"]]
        ys = [q["y0"], q["y1"], q["y2"], q["y3"]]
        min_x, max_x = float(min(xs)), float(max(xs))
        min_y, max_y = float(min(ys)), float(max(ys))
        return (min_x, min_y, max(1.0, max_x - min_x), max(1.0, max_y - min_y))
    
    def _child_depth(hierarchy_arr, child_idx):
        depth = 0
        cur = child_idx
        while cur != -1 and depth < 5:
            depth += 1
            cur = hierarchy_arr[cur][2]
        return depth
    
    def _is_qr_triangle_loose(c1, c2, c3):
        d12 = float(np.linalg.norm(c1 - c2))
        d13 = float(np.linalg.norm(c1 - c3))
        d23 = float(np.linalg.norm(c2 - c3))
        ds = sorted([d12, d13, d23])
        a, b, c = ds[0], ds[1], ds[2]
        if a <= 1e-6:
            return False
        
        pythag_err = abs(c * c - (a * a + b * b)) / max(c * c, 1.0)
        if pythag_err > 0.20:
            return False
        
        leg_ratio = b / a
        diag_ratio = c / a
        if leg_ratio > 1.9:
            return False
        if diag_ratio < 1.15 or diag_ratio > 2.4:
            return False
        
        return True
    
    def _count_descendants(hierarchy_arr, idx):
        count = 0
        child = hierarchy_arr[idx][2]
        while child != -1:
            count += 1
            count += _count_descendants(hierarchy_arr, child)
            child = hierarchy_arr[child][0]
        return count
    
    def _extract_candidates(bin_mask, tag):
        candidates = []
        contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return candidates
        hierarchy = hierarchy[0]
        
        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            if w < 14 or h < 14:
                continue
            
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.65 or aspect_ratio > 1.55:
                continue
            
            child = hierarchy[idx][2]
            if child == -1:
                continue
            
            depth = _child_depth(hierarchy, child)
            if depth < 2:
                continue
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            
            solidity = float(area) / hull_area
            extent = float(area) / float(w * h)
            if solidity < 0.35 or extent < 0.22:
                continue
            
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
            candidates.append({
                "idx": idx,
                "rect": (x, y, w, h),
                "center": center,
                "box": box,
                "area": float(area),
                "depth": depth,
                "solidity": solidity,
                "tag": tag
            })
        
        return candidates
    
    def _extract_square_candidates(bin_mask, tag):
        candidates = []
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return candidates
        
        img_area = float(img_h * img_w)
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(95.0, min_area * 0.40):
                continue
            if area > img_area * 0.09:
                continue
            
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            
            x, y, w, h = cv2.boundingRect(approx)
            if w < 10 or h < 10:
                continue
            ar = w / float(h)
            if ar < 0.60 or ar > 1.65:
                continue
            
            extent = float(area) / float(w * h)
            if extent < 0.26:
                continue
            
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect).astype(np.float32)
            center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
            candidates.append({
                "rect": (x, y, w, h),
                "center": center,
                "box": box,
                "area": float(area),
                "depth": 1,
                "solidity": extent,
                "tag": tag
            })
        
        return candidates
    
    if work_mask is None:
        return final_qrs
    
    if len(work_mask.shape) == 3:
        gray = cv2.cvtColor(work_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = work_mask.copy()
    
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    img_h, img_w = gray.shape[:2]
    
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_img = cv2.bitwise_not(bin_img)
    
    # Extract all candidates
    finder_candidates = []
    finder_candidates.extend(_extract_candidates(bin_img, tag="bin"))
    finder_candidates.extend(_extract_candidates(inv_img, tag="inv"))
    
    finder_candidates.sort(key=lambda t: (t["depth"], t["area"], t["solidity"]), reverse=True)
    
    # Deduplication
    unique_finders = []
    for cand in finder_candidates:
        duplicated = False
        for kept in unique_finders:
            cx1, cy1 = cand["center"]
            cx2, cy2 = kept["center"]
            dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
            min_side = float(min(cand["rect"][2], cand["rect"][3],
                                 kept["rect"][2], kept["rect"][3]))
            if dist < (0.42 * min_side) or _rect_iou(cand["rect"], kept["rect"]) > 0.35:
                duplicated = True
                break
        if not duplicated:
            unique_finders.append(cand)
    
    # MODE 1: Try 3-finder combinations (standard QR)
    if len(unique_finders) >= 3:
        for combo in combinations(unique_finders, 3):
            c1, c2, c3 = combo[0]["center"], combo[1]["center"], combo[2]["center"]
            
            finder_areas = np.array([combo[0]["area"], combo[1]["area"], combo[2]["area"]], dtype=np.float32)
            if float(np.min(finder_areas)) <= 1e-6:
                continue
            if float(np.max(finder_areas) / np.min(finder_areas)) > 3.1:
                continue
            
            if not is_valid_qr_triangle(c1, c2, c3, tolerance=0.42):
                if not _is_qr_triangle_loose(c1, c2, c3):
                    continue
            
            sides = np.array([
                np.linalg.norm(c1 - c2),
                np.linalg.norm(c1 - c3),
                np.linalg.norm(c2 - c3)
            ], dtype=np.float32)
            if float(np.min(sides)) <= 1e-6:
                continue
            if float(np.max(sides) / np.min(sides)) > 3.0:
                continue
            
            merged_pts = np.vstack([f["box"] for f in combo]).astype(np.float32)
            merged_pts = merged_pts.reshape(-1, 1, 2)
            
            hull = cv2.convexHull(merged_pts)
            qr_rect = cv2.minAreaRect(hull)
            side = max(qr_rect[1][0], qr_rect[1][1])
            if side < 24:
                continue
            
            rw, rh = qr_rect[1]
            if rw <= 1e-6 or rh <= 1e-6:
                continue
            rect_ar = max(rw, rh) / min(rw, rh)
            if rect_ar > 1.85:
                continue
            
            qr_box = cv2.boxPoints(qr_rect)
            ordered_box = order_points(qr_box)
            
            corners = ordered_box.astype(np.float32)
            centers = [c1, c2, c3]
            nearest_corner_ids = []
            nearest_corner_dists = []
            for cc in centers:
                d = np.linalg.norm(corners - cc.reshape(1, 2), axis=1)
                cid = int(np.argmin(d))
                nearest_corner_ids.append(cid)
                nearest_corner_dists.append(float(d[cid]))
            
            if len(set(nearest_corner_ids)) < 3:
                continue
            if max(nearest_corner_dists) > side * 0.52:
                continue
            
            pad = max(4.0, side * 0.12)
            box_center = ordered_box.mean(axis=0)
            padded_box = []
            for pt in ordered_box:
                direction = pt - box_center
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                padded_box.append(pt + direction * pad)
            padded_box = np.array(padded_box, dtype=np.float32)
            padded_box[:, 0] = np.clip(padded_box[:, 0], 0, img_w - 1)
            padded_box[:, 1] = np.clip(padded_box[:, 1], 0, img_h - 1)
            
            final_qrs.append({
                "x0": float(padded_box[0][0]), "y0": float(padded_box[0][1]),
                "x1": float(padded_box[1][0]), "y1": float(padded_box[1][1]),
                "x2": float(padded_box[2][0]), "y2": float(padded_box[2][1]),
                "x3": float(padded_box[3][0]), "y3": float(padded_box[3][1]),
                "content": ""
            })
    
    # Clean up: NMS + area filtering
    final_qrs = apply_nms(final_qrs, distance_threshold=24)
    
    filtered_qrs = []
    for q in final_qrs:
        pts_q = np.array([
            [q["x0"], q["y0"]],
            [q["x1"], q["y1"]],
            [q["x2"], q["y2"]],
            [q["x3"], q["y3"]],
        ], dtype=np.float32)
        area_ratio = float(cv2.contourArea(pts_q) / max(1.0, float(img_w * img_h)))
        if area_ratio > 0.78:
            continue
        filtered_qrs.append(q)
    final_qrs = filtered_qrs
    
    # PCA-based ranking
    model = _load_pca_model()
    work_mask_bgr = work_mask if len(work_mask.shape) == 3 else cv2.cvtColor(work_mask, cv2.COLOR_GRAY2BGR)
    scored_qrs = []
    for q in final_qrs:
        if model is not None:
            d = _pca_distance_from_box(work_mask_bgr, q, model=model)
            if d is None or d <= model["threshold"] * 1.25:
                scored_qrs.append((1e9 if d is None else d, q))
        else:
            scored_qrs.append((0, q))
    
    if scored_qrs:
        scored_qrs.sort(key=lambda t: t[0])
        final_qrs = [q for _, q in scored_qrs]
    
    return final_qrs


# ============================================================================
# MAIN DETECTION PIPELINE
# ============================================================================

def detect_qr_codes(img_path):
    """
    Main detection pipeline for a single image.
    Returns list of QR bounding boxes.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    # Preprocessing
    mask_fine, mask_coarse, _ = preprocess_image(img)
    
    # Get candidate QR boxes from multiple sources
    qrs_fine = get_small_bounding_boxes(mask_fine, min_area=300)
    qrs_coarse_1 = get_big_bounding_boxes(mask_fine)
    qrs_coarse_2 = get_big_bounding_boxes(mask_coarse)
    qrs_finder_patterns_fine = get_finder_patterns(mask_fine)

    # Lazy-load threshold-based finder branch only when needed.
    qrs_finder_patterns_thresh = None

    def _get_qrs_finder_patterns_thresh():
        nonlocal qrs_finder_patterns_thresh
        if qrs_finder_patterns_thresh is None:
            morph_img = preprocessing_for_FP(img)
            qrs_finder_patterns_thresh = get_finder_patterns(morph_img)
        return qrs_finder_patterns_thresh

    dense_small_mode = len(qrs_fine) >= 20

    pca_model = _load_pca_model()

    def _verify_branch(branch_qrs):
        verify_mode = "dense_small" if (branch_qrs is qrs_fine and dense_small_mode) else "default"
        return [
            qr for qr in branch_qrs
            if verify_qr_soft(img, qr, mode=verify_mode, pca_model=pca_model)
        ]
    
    # Selection strategy
    if qrs_finder_patterns_fine:
        qrs = qrs_finder_patterns_fine
    elif qrs_fine:
        qrs = qrs_fine
    else:
        coarse_count = len(qrs_coarse_1) + len(qrs_coarse_2)
        if coarse_count < 2:
            thresh_branch = _get_qrs_finder_patterns_thresh()
            qrs = thresh_branch if thresh_branch else (qrs_coarse_1 if qrs_coarse_1 else qrs_coarse_2)
        else:
            qrs = qrs_coarse_1 if qrs_coarse_1 else qrs_coarse_2

    # Verification
    verified_qrs = _verify_branch(qrs)

    # Fallback chain
    if not verified_qrs:
        fallback_order = [
            _get_qrs_finder_patterns_thresh(),
            qrs_coarse_2,
            qrs_coarse_1,
            qrs_fine,
        ]
        for branch_qrs in fallback_order:
            branch_verified = _verify_branch(branch_qrs)
            if branch_verified:
                verified_qrs = branch_verified
                break

    return verified_qrs
