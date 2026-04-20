"""
QR Code Decoder Wrapper - custom decoder only.
No built-in QR detect/decode APIs are used.
"""

import cv2
import numpy as np


try:
    from decoder import QRDecode
except ImportError:
    class QRDecode:
        def __init__(self, qr_code, debug=False, verbose=False):
            self.matrix = qr_code
            self.debug = debug
            self.verbose = verbose

        def decode(self):
            return ""


def _order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _get_candidate_sizes(side):
    """Pick plausible QR module sizes for current ROI side length."""
    valid_sizes = [21 + 4 * i for i in range(40)]
    chosen = set()

    # Typical module density observed in cropped QR warps.
    for ppm in [1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        estimate = int(round(side / ppm))
        nearest = min(valid_sizes, key=lambda v: abs(v - estimate))
        chosen.add(nearest)
        if nearest - 4 >= 21:
            chosen.add(nearest - 4)
        if nearest + 4 <= 177:
            chosen.add(nearest + 4)

    # Always keep lower versions because they are common.
    for s in [21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]:
        chosen.add(s)

    out = []
    for s in sorted(chosen):
        ppm = side / float(s)
        if 1.25 <= ppm <= 18.0:
            out.append(s)
    return out


def _nearest_valid_qr_size(estimate):
    valid_sizes = [21 + 4 * i for i in range(40)]
    return min(valid_sizes, key=lambda v: abs(v - int(round(estimate))))


def _shift_image(gray, dx, dy):
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _scale_quad(pts, scale):
    center = np.mean(pts, axis=0, keepdims=True)
    return (pts - center) * float(scale) + center


def _finder_corner_score(matrix, corner):
    """Return how closely a 7x7 region matches a QR finder pattern (0..1)."""
    n = matrix.shape[0]
    if n < 21:
        return 0.0

    if corner == "tl":
        patch = matrix[0:7, 0:7]
    elif corner == "tr":
        patch = matrix[0:7, n - 7:n]
    elif corner == "bl":
        patch = matrix[n - 7:n, 0:7]
    else:
        return 0.0

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[0, :] = 1
    expected[6, :] = 1
    expected[:, 0] = 1
    expected[:, 6] = 1
    expected[2:5, 2:5] = 1

    return float(np.mean((patch == expected).astype(np.float32)))


def _looks_like_qr_matrix(matrix):
    """Quick heuristic to reject matrices that cannot be valid QR symbols."""
    if matrix is None or matrix.size == 0:
        return False

    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if n < 21 or ((n - 21) % 4 != 0):
        return False

    black_ratio = float(np.mean(matrix))
    if black_ratio < 0.08 or black_ratio > 0.92:
        return False

    s1 = _finder_corner_score(matrix, "tl")
    s2 = _finder_corner_score(matrix, "tr")
    s3 = _finder_corner_score(matrix, "bl")
    scores = sorted([s1, s2, s3])
    return scores[0] >= 0.40 and scores[1] >= 0.46


def _decode_with_custom(gray_warp):
    """Try many matrix hypotheses and decode with QRDecode only."""
    try:
        if gray_warp is None or gray_warp.size == 0:
            return ""

        h, w = gray_warp.shape[:2]
        min_side = min(h, w)
        if min_side < 24:
            return ""

        trim_ratios = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

        for trim_ratio in trim_ratios:
            trim = int(min_side * trim_ratio)
            if trim * 2 >= min_side - 4:
                continue

            roi = gray_warp[trim:h - trim, trim:w - trim]
            if roi.size == 0:
                continue

            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            roi_eq = cv2.equalizeHist(roi_blur)
            rois = [roi_blur, roi_eq]

            target_sizes = _get_candidate_sizes(min(roi.shape[:2]))
            for target_size in target_sizes:
                module_px = max(1.0, min(roi.shape[:2]) / float(target_size))
                shift_fracs = [0.0, 0.33, 0.66]

                for base_roi in rois:
                    for sx in shift_fracs:
                        for sy in shift_fracs:
                            shifted = _shift_image(base_roi, sx * module_px, sy * module_px)

                            sampled_area = cv2.resize(
                                shifted,
                                (target_size, target_size),
                                interpolation=cv2.INTER_AREA,
                            )
                            sampled_near = cv2.resize(
                                shifted,
                                (target_size, target_size),
                                interpolation=cv2.INTER_NEAREST,
                            )

                            for sampled in [sampled_area, sampled_near]:
                                _, binary_otsu = cv2.threshold(
                                    sampled,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                binary_adp = cv2.adaptiveThreshold(
                                    sampled,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    11,
                                    2,
                                )

                                base_matrices = [
                                    (binary_otsu < 128).astype(np.uint8),
                                    (binary_otsu > 127).astype(np.uint8),
                                    (binary_adp < 128).astype(np.uint8),
                                    (binary_adp > 127).astype(np.uint8),
                                ]

                                for base in base_matrices:
                                    for rotation in range(4):
                                        matrix = np.rot90(base, k=rotation)
                                        if not _looks_like_qr_matrix(matrix):
                                            continue
                                        try:
                                            decoder = QRDecode(matrix, debug=False, verbose=False)
                                            content = decoder.decode()
                                            if content:
                                                content = str(content).strip()
                                                if content:
                                                    return content
                                        except Exception:
                                            continue
        return ""
    except Exception:
        return ""


def _refine_warp_candidates(gray_warp):
    """Generate refined square candidates from contour geometry only."""
    candidates = [gray_warp]
    try:
        if gray_warp is None or gray_warp.size == 0:
            return candidates

        h, w = gray_warp.shape[:2]
        min_side = min(h, w)
        if min_side < 40:
            return candidates

        prep = cv2.GaussianBlur(gray_warp, (3, 3), 0)
        _, binary = cv2.threshold(prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        probes = [binary, cv2.bitwise_not(binary)]

        for probe in probes:
            contours, _ = cv2.findContours(probe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 0.08 * (h * w):
                    continue

                peri = cv2.arcLength(contour, True)
                if peri <= 0:
                    continue

                approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
                if len(approx) < 4:
                    continue

                rect = cv2.minAreaRect(contour)
                rw, rh = rect[1]
                if rw < 20 or rh < 20:
                    continue

                ar = max(rw, rh) / max(min(rw, rh), 1e-6)
                if ar > 1.6:
                    continue

                src = _order_points(cv2.boxPoints(rect).astype(np.float32))
                dst = np.array(
                    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float32,
                )
                mat = cv2.getPerspectiveTransform(src, dst)
                refined = cv2.warpPerspective(gray_warp, mat, (w, h))
                if refined is not None and refined.size > 0:
                    candidates.append(refined)

            if len(candidates) > 1:
                break
    except Exception:
        return candidates

    return candidates


def decode_qr_content(img, qr_box):
    """
    Decode QR content from one bounding box using custom QRDecode only.
    """
    try:
        h, w = img.shape[:2]

        base_pts = np.array([
            [np.clip(qr_box["x0"], 0, w - 1), np.clip(qr_box["y0"], 0, h - 1)],
            [np.clip(qr_box["x1"], 0, w - 1), np.clip(qr_box["y1"], 0, h - 1)],
            [np.clip(qr_box["x2"], 0, w - 1), np.clip(qr_box["y2"], 0, h - 1)],
            [np.clip(qr_box["x3"], 0, w - 1), np.clip(qr_box["y3"], 0, h - 1)],
        ], dtype=np.float32)

        quad_scales = [0.72, 0.80, 0.88, 0.96, 1.00, 1.06, 1.12]
        for scale in quad_scales:
            pts_source = _order_points(_scale_quad(base_pts, scale).astype(np.float32))
            pts_source[:, 0] = np.clip(pts_source[:, 0], 0, w - 1)
            pts_source[:, 1] = np.clip(pts_source[:, 1], 0, h - 1)

            side_lengths = [
                np.linalg.norm(pts_source[1] - pts_source[0]),
                np.linalg.norm(pts_source[2] - pts_source[1]),
                np.linalg.norm(pts_source[3] - pts_source[2]),
                np.linalg.norm(pts_source[0] - pts_source[3]),
            ]
            max_side = int(max(side_lengths)) if side_lengths else 0
            if max_side < 20:
                continue

            est_modules = _nearest_valid_qr_size(max_side / 3.0)
            min_warp = int(est_modules * 8)
            warp_size = int(np.clip(max(max_side * 2, min_warp), 128, 1024))
            pts_dest = np.array([
                [0, 0], [warp_size - 1, 0],
                [warp_size - 1, warp_size - 1], [0, warp_size - 1],
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts_source, pts_dest)
            warped = cv2.warpPerspective(img, matrix, (warp_size, warp_size))
            if warped is None or warped.size == 0:
                continue

            if len(warped.shape) == 3:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                gray = warped.copy()

            for candidate in _refine_warp_candidates(gray):
                content = _decode_with_custom(candidate)
                if content:
                    return content

        return ""
    except Exception:
        return ""


def decode_all_qrs(img_path, qr_boxes):
    """
    Decode all QR boxes in one image using custom QRDecode only.
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
