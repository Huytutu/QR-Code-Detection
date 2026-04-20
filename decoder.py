import numpy as np
import cv2

# EC codewords table: {version: {ec_level: total_ec_codewords}}
EC_CODEWORDS = {
    1:  {"L": 7,   "M": 10,  "Q": 13,  "H": 17},
    2:  {"L": 10,  "M": 16,  "Q": 22,  "H": 28},
    3:  {"L": 15,  "M": 26,  "Q": 36,  "H": 44},
    4:  {"L": 20,  "M": 36,  "Q": 52,  "H": 64},
    5:  {"L": 26,  "M": 48,  "Q": 72,  "H": 88},
    6:  {"L": 36,  "M": 64,  "Q": 96,  "H": 112},
    7:  {"L": 40,  "M": 72,  "Q": 108, "H": 130},
    8:  {"L": 48,  "M": 88,  "Q": 132, "H": 156},
    9:  {"L": 60,  "M": 110, "Q": 160, "H": 192},
    10: {"L": 72,  "M": 130, "Q": 192, "H": 224},
}

# ── GF(256) ──────────────────────────────────────────────────────────────────
class GF256:
    def __init__(self, prim=0x11D):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def mul(self, a, b):
        return 0 if (a == 0 or b == 0) else self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError("GF256 division by zero")
        return self.exp[(self.log[a] - self.log[b]) % 255]

    def poly_eval(self, poly, x):
        result = 0
        for coef in poly:
            result = self.mul(result, x) ^ coef
        return result


# ── Reed-Solomon ──────────────────────────────────────────────────────────────

class ReedSolomonDecoder:
    def __init__(self, gf):
        self.gf = gf

    def correct_errors(self, msg, nsym):
        syndromes = [self.gf.poly_eval(msg, self.gf.exp[i]) for i in range(nsym)]
        if max(syndromes) == 0:
            return msg

        err_loc, old_loc = [1], [1]
        for i in range(nsym):
            delta = syndromes[i]
            for j in range(1, len(err_loc)):
                delta ^= self.gf.mul(err_loc[-(j + 1)], syndromes[i - j])
            old_loc.append(0)
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = list(old_loc)
                    ev = self.gf.poly_eval(old_loc, 0)
                    if ev == 0:
                        return msg
                    scale = self.gf.div(delta, ev)
                    old_loc = [self.gf.mul(c, scale) for c in err_loc]
                    err_loc = new_loc
                err_loc = [c ^ self.gf.mul(delta, t)
                           for c, t in zip(err_loc + [0], old_loc)]

        err_pos = [i for i in range(len(msg))
                   if self.gf.poly_eval(err_loc, self.gf.exp[255 - i]) == 0]
        if len(err_pos) != len(err_loc) - 1:
            raise ValueError("Too many errors to correct")

        for pos in err_pos:
            x = self.gf.exp[255 - pos]
            y = self.gf.poly_eval(syndromes[::-1], x)
            denom = 1
            for i in range(len(err_loc)):
                if i != pos:
                    denom = self.gf.mul(denom, x ^ self.gf.exp[255 - i])
            if denom == 0:
                return msg
            msg[pos] ^= self.gf.div(y, denom)
        return msg


# ── Main Decoder ──────────────────────────────────────────────────────────────

class QRCodeDecoder:
    def __init__(self, trace_mode=False):
        self.gf  = GF256()
        self.rs  = ReedSolomonDecoder(self.gf)
        self.trace_mode = trace_mode

    # ── helpers ──

    def _trace(self, lbl, data=None):
        if self.trace_mode:
            print(f"[TRACE] {lbl}:", data)

    def _binarize(self, image_input):
        # Accept either image path or image ndarray.
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = image_input.copy()
            else:
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("image_input must be a file path or numpy array")

        if img is None or img.size == 0:
            raise ValueError("Cannot load image for decoding")

        # Tiền xử lý nâng cao: Adaptive Threshold (Khắc phục ảnh bị đổ bóng, chênh sáng)
        # Block size = 21, C = 5 là các tham số phổ biến cho QR code
        binary = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 5
        )
        # Chuyển về định dạng 1 (sáng) và 0 (tối) như logic cũ của bạn
        binary = (binary > 128).astype(np.uint8)
        self._trace("binarize_adaptive", binary.shape)
        return binary
    # ── finder pattern detection (1:1:3:1:1 scan) ──

    def _scan_line(self, line):
        """Return list of (center_pos, module_size) for 1:1:3:1:1 patterns found."""
        runs, prev, count = [], int(line[0]), 1
        for val in line[1:]:
            v = int(val)
            if v == prev:
                count += 1
            else:
                runs.append((prev, count))
                prev, count = v, 1
        runs.append((prev, count))

        hits = []
        for i in range(len(runs) - 4):
            colors  = [runs[i + j][0] for j in range(5)]
            lengths = [runs[i + j][1] for j in range(5)]
            if colors != [0, 1, 0, 1, 0]:          # dark-light-dark-light-dark
                continue
            total = sum(lengths)
            u = total / 7.0
            if all(abs(lengths[k] - u * e) < u * 0.6
                   for k, e in enumerate([1, 1, 3, 1, 1])):
                start = sum(runs[k][1] for k in range(i))
                center = start + lengths[0] + lengths[1] + lengths[2] // 2
                hits.append((center, u))
        return hits

    def _find_peaks(self, votes, n=3):
        """Return top-n peaks via iterative suppression."""
        v = votes.copy().astype(float)
        min_dist = max(v.shape) // 8
        peaks = []
        while len(peaks) < n * 2 and v.max() >= 2:
            idx = np.unravel_index(v.argmax(), v.shape)
            peaks.append(idx)
            r0 = max(0, idx[0] - min_dist); r1 = min(v.shape[0], idx[0] + min_dist)
            c0 = max(0, idx[1] - min_dist); c1 = min(v.shape[1], idx[1] + min_dist)
            v[r0:r1, c0:c1] = 0
        return peaks

    def _find_finder_patterns(self, binary):
        h, w = binary.shape
        votes = np.zeros((h, w), dtype=np.float32)

        for r in range(h):
            for center, _ in self._scan_line(binary[r]):
                c = int(round(center))
                if 0 <= c < w:
                    votes[r, c] += 1

        for c in range(w):
            for center, _ in self._scan_line(binary[:, c]):
                r = int(round(center))
                if 0 <= r < h:
                    votes[r, c] += 1

        peaks = self._find_peaks(votes, n=3)
        if len(peaks) < 3:
            raise ValueError(f"Only found {len(peaks)} finder pattern(s) — check image quality.")

        centers = [(float(r), float(c)) for r, c in peaks[:3]]
        self._trace("finder_patterns_raw", centers)
        return self._orient_finders(centers)

    def _orient_finders(self, centers):
            """Return (TL, TR, BL) bằng hình học không gian, chống nhiễu góc xoay."""
            # 1. Tính bình phương khoảng cách giữa các điểm
            def distSq(p1, p2):
                return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

            d01 = distSq(centers[0], centers[1])
            d12 = distSq(centers[1], centers[2])
            d02 = distSq(centers[0], centers[2])

            # 2. Góc vuông (Top-Left) luôn đối diện với cạnh huyền (cạnh dài nhất)
            if d01 >= d12 and d01 >= d02:
                tl, pA, pB = centers[2], centers[0], centers[1]
            elif d12 >= d01 and d12 >= d02:
                tl, pA, pB = centers[0], centers[1], centers[2]
            else:
                tl, pA, pB = centers[1], centers[0], centers[2]

            # 3. Phân biệt TR và BL bằng Tích có hướng (Cross Product)
            # Vì hệ toạ độ ảnh: y hướng xuống, x hướng sang phải.
            dyA, dxA = pA[0] - tl[0], pA[1] - tl[1]
            dyB, dxB = pB[0] - tl[0], pB[1] - tl[1]

            cross_product = dxA * dyB - dyA * dxB
            if cross_product > 0:
                tr, bl = pA, pB
            else:
                tr, bl = pB, pA

            self._trace("finder_patterns", {"TL": tl, "TR": tr, "BL": bl})
            return tl, tr, bl

    # ── version auto-detection ──

    def _estimate_version(self, tl, tr, bl, img_shape):
        """
        center-to-center distance = (4v + 10) modules.
        Full QR width = (4v + 17) modules.
        Pick version whose expected QR pixel size best matches the image size.
        """
        d_h = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        d_v = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
        avg_d = (d_h + d_v) / 2.0
        img_size = max(img_shape)

        best_v, best_err = 1, float('inf')
        for v in range(1, 11):
            module_size = avg_d / (4 * v + 10)
            expected    = module_size * (4 * v + 17)
            err = abs(expected - img_size)
            if err < best_err:
                best_err, best_v = err, v

        self._trace("estimated_version", best_v)
        return best_v

    # ── perspective correction ──

    def _compute_homography(self, src, dst):
        A = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            A += [[-sx, -sy, -1, 0, 0, 0, sx*dx, sy*dx, dx],
                  [0, 0, 0, -sx, -sy, -1, sx*dy, sy*dy, dy]]
        _, _, Vt = np.linalg.svd(np.array(A, dtype=float))
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def _warp(self, binary, H, size):
        H_inv = np.linalg.inv(H)
        ys, xs = np.mgrid[0:size, 0:size]
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(size * size)])
        src = H_inv @ pts
        src /= src[2]
        ix = np.round(src[0]).astype(int)
        iy = np.round(src[1]).astype(int)
        valid = (ix >= 0) & (ix < binary.shape[1]) & (iy >= 0) & (iy < binary.shape[0])
        flat  = np.ones(size * size, dtype=np.uint8)   # default = light
        flat[valid] = binary[iy[valid], ix[valid]]
        return flat.reshape(size, size)

    def _extract_grid(self, binary, tl, tr, bl, version):
        size = 21 + (version - 1) * 4
        br   = (tr[0] - tl[0] + bl[0], tr[1] - tl[1] + bl[1])

        # Finder centers are at module (3, 3), (3, size-4), (size-4, size-4), (size-4, 3)
        # src: (col=x, row=y), dst: (module_col, module_row)
        src = [(tl[1], tl[0]), (tr[1], tr[0]), (br[1],  br[0]),  (bl[1], bl[0])]
        dst = [(3, 3),         (size-4, 3),    (size-4, size-4), (3, size-4)]

        H = self._compute_homography(src, dst)
        warped = self._warp(binary, H, size)
        grid   = (warped > 0).astype(np.uint8)
        self._trace("grid_size", grid.shape)
        return grid

    # ── format information ──

    def _extract_format_info(self, grid):
        FORMAT_MASK = 0b101010000010010
        bits = ([grid[8, i] for i in range(6)]
                + [grid[8, 7], grid[8, 8], grid[7, 8]]
                + [grid[i, 8] for i in range(5, -1, -1)])
        fmt  = int(''.join(map(str, bits)), 2) ^ FORMAT_MASK
        ec   = {0b01: "L", 0b00: "M", 0b11: "Q", 0b10: "H"}.get((fmt >> 13) & 3, "Unknown")
        mask = (fmt >> 10) & 7
        info = {"error_correction": ec, "mask_pattern": mask}
        self._trace("format_info", info)
        return info

    # ── unmasking ──

    _MASK_FNS = [
        lambda r, c: (r + c) % 2 == 0,
        lambda r, c: r % 2 == 0,
        lambda r, c: c % 3 == 0,
        lambda r, c: (r + c) % 3 == 0,
        lambda r, c: (r // 2 + c // 3) % 2 == 0,
        lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
        lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
        lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
    ]

    def _is_function(self, r, c, size):
        if (r < 9 and c < 9) or (r < 9 and c >= size - 8) or (r >= size - 8 and c < 9):
            return True
        return r == 6 or c == 6

    def _unmask(self, grid, mask_pattern):
        size = grid.shape[0]
        fn   = self._MASK_FNS[mask_pattern]
        out  = grid.copy()
        for r in range(size):
            for c in range(size):
                if not self._is_function(r, c, size) and fn(r, c):
                    out[r, c] ^= 1
        return out

    # ── codeword extraction ──

    def _read_codewords(self, grid):
            size = grid.shape[0]
            codewords, bits = [],[]
            col = size - 1
            
            upward = True  # QR Code luôn bắt đầu đi từ dưới LÊN trên
            
            while col > 0:
                # Bỏ qua cột thứ 6 (chứa đường Timing Pattern dọc)
                if col == 6:
                    col -= 1
                    
                # Tuỳ theo biến upward để quyết định đọc từ dưới lên hay từ trên xuống
                rows = range(size - 1, -1, -1) if upward else range(size)
                
                for row in rows:
                    for c in [col, col - 1]:
                        if not self._is_function(row, c, size):
                            bits.append(grid[row, c])
                            
                            # Cứ gom đủ 8 bit thì tạo thành 1 byte (codeword)
                            if len(bits) == 8:
                                codewords.append(int(''.join(map(str, bits)), 2))
                                bits =[]  # Reset mảng bits để đọc byte tiếp theo
                
                # Đổi hướng sau khi quét xong 2 cột (Lên -> Xuống, Xuống -> Lên)
                upward = not upward
                # Nhảy sang 2 cột tiếp theo bên trái
                col -= 2
                
            # Nếu ở cuối cùng còn dư một vài bit (không đủ 8 bit), thêm '0' vào cho đủ byte
            if bits:
                codewords.append(int(''.join(map(str, bits)).ljust(8, '0'), 2))
                
            self._trace("codewords_count", len(codewords))
            return codewords

    # ── payload decoding ──

    def _decode_payload(self, bits, mode, version):
        CCB = {           # character-count bits per version range
            "numeric":      [10, 12, 14],
            "alphanumeric": [9,  11, 13],
            "byte":         [8,  16, 16],
            "kanji":        [8,  10, 12],
        }
        if mode not in CCB:
            return "[mode not supported]"

        nb = CCB[mode][0 if version <= 9 else 1 if version <= 26 else 2]
        n  = int(bits[4:4 + nb], 2)
        i  = 4 + nb

        try:
            if mode == "numeric":
                out = []
                while n > 0:
                    if n >= 3:   out.append(f"{int(bits[i:i+10],2):03d}"); i+=10; n-=3
                    elif n == 2: out.append(f"{int(bits[i:i+7], 2):02d}"); i+=7;  n-=2
                    else:        out.append(f"{int(bits[i:i+4], 2):01d}"); i+=4;  n-=1
                return ''.join(out)

            if mode == "alphanumeric":
                T = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
                out = []
                while n > 0:
                    if n >= 2:
                        p = int(bits[i:i+11], 2); out += [T[p//45], T[p%45]]; i+=11; n-=2
                    else:
                        out.append(T[int(bits[i:i+6], 2)]); i+=6; n-=1
                return ''.join(out)

            if mode == "byte":
                return ''.join(chr(int(bits[i + j*8: i + j*8 + 8], 2)) for j in range(n))

            if mode == "kanji":
                out = []
                for _ in range(n):
                    code = int(bits[i:i+13], 2); i += 13
                    code += 0xC140 if code >= 0x1F00 else 0x8140
                    out.append(bytes([(code>>8)&0xFF, code&0xFF]).decode('shift_jis', errors='replace'))
                return ''.join(out)

        except Exception as e:
            return f"[decoding error: {e}]"

    # ── public entry point ──

    def decode(self, image_input):
        binary = self._binarize(image_input)

        tl, tr, bl = self._find_finder_patterns(binary)
        version    = self._estimate_version(tl, tr, bl, binary.shape)
        grid       = self._extract_grid(binary, tl, tr, bl, version)

        fmt_info   = self._extract_format_info(grid)
        ec_level   = fmt_info["error_correction"]
        mask       = fmt_info["mask_pattern"]

        if version not in EC_CODEWORDS or ec_level not in EC_CODEWORDS[version]:
            raise ValueError(f"Unsupported version {version} or EC level '{ec_level}'")

        ec_bytes = EC_CODEWORDS[version][ec_level]
        self._trace("ec_bytes", ec_bytes)

        unmasked   = self._unmask(grid, mask)
        codewords  = self._read_codewords(unmasked)
        corrected  = self.rs.correct_errors(list(codewords), ec_bytes)
        self._trace("corrected_codewords", corrected)

        bits    = ''.join(f"{b:08b}" for b in corrected)
        mode    = {'0001': "numeric", '0010': "alphanumeric",
                   '0100': "byte",    '1000': "kanji"}.get(bits[:4], "unknown")
        self._trace("mode", mode)

        payload = self._decode_payload(bits, mode, version)
        self._trace("payload", payload)
        return payload


if __name__ == "__main__":
    decoder = QRCodeDecoder(trace_mode=True)
    result  = decoder.decode("../qr/valid/v2SMS-Text_png_jpg.rf.adcf9a65a3967bcf580a41b7abf76383.jpg")
    print("\nDecoded:", result)