import numpy as np


class QRDecode:
    def __init__(self, qr_code: np.ndarray, debug: bool = False, verbose: bool = False) -> None:
        self.matrix = qr_code
        self.debug = debug
        self.verbose = verbose

        # Validate input matrix
        if not qr_code.any() or any(len(row) != len(self.matrix) for row in self.matrix):
            raise ValueError("Invalid QR code matrix")

    def get_version(self) -> int:
        """Get the QR code version based on matrix size."""
        size = len(self.matrix)
        version = (size - 17) // 4
        if self.debug:
            print(f"[DEBUG] get_version: Matrix size = {size}, computed version = {version}")
            print(f"[DEBUG] get_version: Expected matrix size fo r this version = {17 + 4 * version}")
        return version

    def mark_function_modules(self) -> np.ndarray:
        """Mark all function modules in the QR code."""
        size = len(self.matrix)
        function_modules = np.zeros((size, size), dtype=bool)
        
        # Compute version before using it
        version = self.get_version()

        # Helper function to add a finder pattern and its separator at given position
        def add_finder_pattern(row, col):
            # Add the 7x7 finder pattern
            for r in range(7):
                for c in range(7):
                    if 0 <= row + r < size and 0 <= col + c < size:
                        function_modules[row + r, col + c] = True
            
            # Add separator (white border)
            # Mark the entire 8x8 region around the finder pattern
            for r in range(-1, 8):
                for c in range(-1, 8):
                    if 0 <= row + r < size and 0 <= col + c < size:
                        # Only mark if it's not part of the finder pattern itself
                        if r < 0 or r >= 7 or c < 0 or c >= 7:
                            function_modules[row + r, col + c] = True

        # Add finder patterns at the three corners
        add_finder_pattern(0, 0)        # Top-left
        add_finder_pattern(0, size - 7)   # Top-right
        add_finder_pattern(size - 7, 0)   # Bottom-left

        # Add timing patterns
        for i in range(size):
            function_modules[6, i] = True  # Horizontal timing pattern
            function_modules[i, 6] = True  # Vertical timing pattern

        # Add format information areas
        # Around top-left finder pattern
        for i in range(9):
            if i != 6:  # Skip timing pattern
                function_modules[8, i] = True  # Horizontal format info
                if i < 8:
                    function_modules[i, 8] = True  # Vertical format info

        # Around top-right finder pattern
        for i in range(8):
            function_modules[8, size - 1 - i] = True  # Horizontal format info

        # Around bottom-left finder pattern
        for i in range(8):
            function_modules[size - 1 - i, 8] = True  # Vertical format info

        # Dark module
        function_modules[size - 8, 8] = True

        # Mark alignment patterns for version >= 2
        if version >= 2:
            centers = self._get_alignment_pattern_centers()
            for r in centers:
                for c in centers:
                    # Skip the finder pattern regions
                    if (r, c) in [(6, 6), (6, size - 7), (size - 7, 6)]:
                        continue
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if 0 <= r + i < size and 0 <= c + j < size:
                                function_modules[r + i, c + j] = True

        # Add version information areas for version >= 7
        if version >= 7:
            # Add version information below bottom-left finder pattern
            for i in range(6):
                for j in range(3):
                    function_modules[size - 11 + j, i] = True  # Bottom-left version info
                    function_modules[i, size - 11 + j] = True  # Top-right version info

        if self.verbose:
            print("[DEBUG] mark_function_modules: Function modules matrix:")
            for i in range(size):
                row = ""
                for j in range(size):
                    row += "██" if function_modules[i, j] else "  "
                print(row)

        return function_modules

    def _get_alignment_pattern_centers(self) -> list:
        """Return a list of alignment pattern center positions for the QR code."""
        version = self.get_version()
        if version == 1:
            return []
        lookup = {
            2: [6, 18],
            3: [6, 22],
            4: [6, 26],
            5: [6, 30],
            6: [6, 34],
            7: [6, 22, 38],
            8: [6, 24, 42],
            9: [6, 26, 46],
            10: [6, 28, 50],
            11: [6, 30, 54],
            12: [6, 32, 58],
            13: [6, 34, 62],
            14: [6, 26, 46, 66],
            15: [6, 26, 48, 70],
            16: [6, 26, 50, 74],
            17: [6, 30, 54, 78],
            18: [6, 30, 56, 82],
            19: [6, 30, 58, 86],
            20: [6, 34, 62, 90],
            21: [6, 28, 50, 72, 94],
            22: [6, 26, 50, 74, 98],
            23: [6, 30, 54, 78, 102],
            24: [6, 28, 54, 80, 106],
            25: [6, 32, 58, 84, 110],
            26: [6, 30, 58, 86, 114],
            27: [6, 34, 62, 90, 118],
            28: [6, 26, 50, 74, 98, 122],
            29: [6, 30, 54, 78, 102, 126],
            30: [6, 26, 52, 78, 104, 130],
            31: [6, 30, 56, 82, 108, 134],
            32: [6, 34, 60, 86, 112, 138],
            33: [6, 30, 58, 86, 114, 142],
            34: [6, 34, 62, 90, 118, 146],
            35: [6, 30, 54, 78, 102, 126, 150],
            36: [6, 24, 50, 76, 102, 128, 154],
            37: [6, 28, 54, 80, 106, 132, 158],
            38: [6, 32, 58, 84, 110, 136, 162],
            39: [6, 26, 54, 82, 110, 138, 166],
            40: [6, 30, 58, 86, 114, 142, 170]
        }

        if version in lookup:
            return lookup[version]

        # Algorithm to compute alignment pattern positions for theoretical versions > 40:
        # 1. First position is always 6
        # 2. Last position is matrix_size - 7
        # 3. Number of alignment patterns increases with version
        # 4. Patterns are roughly evenly spaced
        def compute_alignment_positions(ver: int) -> list:
            matrix_size = 17 + 4 * ver
            end = matrix_size - 7
            
            # Number of alignment patterns between first and last
            # Increases by roughly 1 pattern every 7 versions
            count = (ver // 7) + 2
            
            if count == 2:
                return [6, end]
            
            # Calculate step size to evenly space patterns
            step = (end - 6) // (count - 1)
            
            # Generate positions
            positions = [6]  # First position is always 6
            for i in range(1, count - 1):
                positions.append(6 + i * step)
            positions.append(end)  # Last position
            
            return positions

        return compute_alignment_positions(version)

    def extract_format_information(self) -> tuple[str, int]:
        """Extract and BCH-correct format information from both format copies."""
        size = len(self.matrix)

        def read_bits(positions: list[tuple[int, int]]) -> int:
            value = 0
            for r, c in positions:
                value = (value << 1) | (self.matrix[r][c] & 1)
            return value

        def build_format_codeword(data_5bits: int) -> int:
            # BCH(15,5) using generator polynomial 0x537, then apply mask 0x5412.
            g = 0x537
            code = data_5bits << 10
            for i in range(14, 9, -1):
                if (code >> i) & 1:
                    code ^= g << (i - 10)
            remainder = code & 0x3FF
            return ((data_5bits << 10) | remainder) ^ 0x5412

        def hamming_distance(a: int, b: int) -> int:
            return (a ^ b).bit_count()

        # Copy near top-left, bit order follows QR spec (MSB -> LSB).
        tl_positions = [
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
            (8, 7), (8, 8), (7, 8),
            (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
        ]

        # Copy near top-right / bottom-left, same logical bit order.
        tr_bl_positions = [
            (size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
            (size - 5, 8), (size - 6, 8), (size - 7, 8),
            (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
            (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1),
        ]

        copy1 = read_bits(tl_positions)
        copy2 = read_bits(tr_bl_positions)

        if self.verbose:
            print(f"[DEBUG] extract_format_information: copy1={copy1:015b}, copy2={copy2:015b}")

        best_data = None
        best_dist = 1_000_000
        best_source = ""

        for data_bits in range(32):
            valid_codeword = build_format_codeword(data_bits)
            d1 = hamming_distance(copy1, valid_codeword)
            d2 = hamming_distance(copy2, valid_codeword)
            if d1 < best_dist:
                best_dist = d1
                best_data = data_bits
                best_source = "copy1"
            if d2 < best_dist:
                best_dist = d2
                best_data = data_bits
                best_source = "copy2"

        if best_data is None or best_dist > 3:
            # Fallback to copy1 raw decode if BCH correction fails.
            if self.debug:
                print("[DEBUG] extract_format_information: BCH correction failed, fallback to copy1 raw")
            format_info = copy1 ^ 0x5412
            best_data = (format_info >> 10) & 0b11111
        elif self.debug:
            print(f"[DEBUG] extract_format_information: selected {best_source}, distance={best_dist}")

        error_correction = {
            0b00: 'M',
            0b01: 'L',
            0b10: 'H',
            0b11: 'Q',
        }[(best_data >> 3) & 0b11]
        mask_pattern = best_data & 0b111

        if self.debug:
            print(f"[DEBUG] extract_format_information: Extracted error correction = {error_correction}, mask pattern = {mask_pattern}")

        return error_correction, mask_pattern

    def extract_version_information(self) -> int:
        """Extract version information from the QR code (for version >= 7)."""
        # For versions less than 7, just return the computed version
        version = self.get_version()
        if version < 7:
            return version

        size = len(self.matrix)
        raw_version = 0

        # Extract version information from bottom-left block
        # Read in the same order as written in QRCode._add_version_information
        for i in range(6):
            for j in range(3):
                bit = self.matrix[size - 11 + j][i] & 1
                raw_version |= (bit << (i * 3 + j))

        if self.debug:
            print(f"[DEBUG] extract_version_information: Raw version bits = {raw_version:018b}")

        corrected_version = self.correct_version_information(raw_version)
        return corrected_version

    def _compute_valid_version_info(self, version: int) -> int:
        """Compute the valid 18-bit version information codeword for the given version using BCH error correction."""
        generator = 0x1F25
        codeword = version << 12
        for i in range(17, 11, -1):
            if codeword & (1 << i):
                codeword ^= generator << (i - 12)
        valid_codeword = (version << 12) | (codeword & 0xFFF)
        if self.debug:
            print(f"[DEBUG] _compute_valid_version_info: version = {version}, valid_codeword = {valid_codeword:018b}")
        return valid_codeword

    def correct_version_information(self, raw_version: int) -> int:
        """Correct errors in raw version information bits using BCH code by brute-force matching valid codewords."""
        if self.debug:
            print(f"[DEBUG] correct_version_information: raw_version = {raw_version:018b}")
        for version in range(7, 41):
            valid_codeword = self._compute_valid_version_info(version)
            diff = bin(valid_codeword ^ raw_version).count('1')
            if self.debug:
                print(f"[DEBUG] correct_version_information: testing version {version} with diff = {diff}")
            if diff <= 3:
                if self.debug:
                    print(f"[DEBUG] correct_version_information: corrected to version {version}")
                return version
        raise ValueError("Failed to correct version information")

    def unmask_data(self, mask_pattern: int) -> None:
        """Unmask the data in the QR code by inverting bits in data modules as per the mask pattern."""
        size = len(self.matrix)
        function_modules = self.mark_function_modules()

        def mask_condition(i: int, j: int, mask: int) -> bool:
            if mask == 0:
                return (i + j) % 2 == 0
            elif mask == 1:
                return i % 2 == 0
            elif mask == 2:
                return j % 3 == 0
            elif mask == 3:
                return (i + j) % 3 == 0
            elif mask == 4:
                return ((i // 2) + (j // 3)) % 2 == 0
            elif mask == 5:
                return ((i * j) % 2 + (i * j) % 3) == 0
            elif mask == 6:
                return (((i * j) % 2 + (i * j) % 3) % 2) == 0
            elif mask == 7:
                return (((i + j) % 2 + (i * j) % 3) % 2) == 0
            else:
                raise ValueError(f"Invalid mask pattern: {mask}")

        for i in range(size):
            for j in range(size):
                # Only unmask data modules
                if not function_modules[i, j]:
                    if mask_condition(i, j, mask_pattern):
                        self.matrix[i][j] ^= 1
        if self.verbose:
            print(f"[DEBUG] unmask_data: Finished unmasking process. Matrix:")
            for row in self.matrix:
                print("".join("██" if bit == 1 else "  " for bit in row))
            print("[DEBUG] unmask_data: Finished unmasking process.")
        # Ensure matrix remains binary
        assert np.all((self.matrix == 0) | (self.matrix == 1)), "Matrix contains non-binary values after unmasking"

    def extract_data_bits(self) -> list[int]:
        """Extract the data bits from the QR code following the standard zigzag order."""
        if self.debug:
            print("[DEBUG] extract_data_bits: Starting extraction of data bits.")
        # Mark function modules to know which modules are reserved
        function_modules = self.mark_function_modules()
        n = len(self.matrix)
        data_bits = []

        col = n - 1
        upward = True
        while col > 0:
            # Skip the vertical timing pattern column
            if col == 6:
                col -= 1
            # Determine the row order based on scanning direction
            row_range = range(n-1, -1, -1) if upward else range(0, n)
            for row in row_range:
                for c in [col, col - 1]:
                    if c < 0 or c >= n:
                        continue
                    # Only extract the bit if it's not part of a function module
                    if not function_modules[row, c]:
                        bit = int(self.matrix[row][c])
                        data_bits.append(bit)
                        if self.verbose:
                            print(f"[DEBUG] extract_data_bits: Appended bit {bit} from position ({row},{c}).")
            upward = not upward
            col -= 2
        if self.debug:
            print(f"[DEBUG] extract_data_bits: Finished extraction, total bits = {len(data_bits)}")
        # Assert that all extracted bits are binary
        assert all(bit in (0, 1) for bit in data_bits), "Matrix contains non-binary values in data bits"
        return data_bits

    def decode_data(self, data_bits: list[int]) -> str:
        """Decode the data bits and return the message, supporting multiple segmentation modes."""
        pointer = 0
        result = ""
        version = self.get_version()
        if self.debug:
            print("[DEBUG] decode_data: Starting multi-segment decoding.")
        while pointer + 4 <= len(data_bits):
            mode_indicator = int(''.join(str(bit) for bit in data_bits[pointer:pointer+4]), 2)
            if self.debug:
                print(f"[DEBUG] decode_data: Mode indicator = {mode_indicator:04b} at pointer {pointer}")
            pointer += 4
            if mode_indicator == 0:  # Terminator
                if self.debug:
                    print("[DEBUG] decode_data: Encountered terminator mode indicator")
                break
            if mode_indicator == 4:  # Byte mode
                count_indicator_length = 8 if version < 10 else 16
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for byte mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Byte mode, character count = {count}")
                pointer += count_indicator_length
                for i in range(count):
                    if pointer >= len(data_bits):
                        if self.debug:
                            print(f"[DEBUG] decode_data: No bits remaining for byte {i+1}/{count}")
                        break
                    if pointer + 8 > len(data_bits):
                        remaining = data_bits[pointer:]
                        padded = remaining + [0] * (8 - len(remaining))
                        byte_val = int(''.join(str(bit) for bit in padded), 2)
                        if self.verbose:
                            print(f"[DEBUG] decode_data: Extracted padded byte value {byte_val} from bits at pointer {pointer}")
                        result += chr(byte_val)
                        pointer = len(data_bits)
                        break
                    byte_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+8]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Extracted byte value {byte_val} from bits at pointer {pointer}")
                    result += chr(byte_val)
                    pointer += 8
            elif mode_indicator == 1:  # Numeric mode
                count_indicator_length = 10 if version < 10 else (12 if version < 27 else 14)
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for numeric mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Numeric mode, digit count = {count}")
                pointer += count_indicator_length
                # Process groups of 3 digits
                while count >= 3:
                    if pointer + 10 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a group of 3 digits in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+10]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted group value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:03d}"
                    pointer += 10
                    count -= 3
                if count == 2:
                    if pointer + 7 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a group of 2 digits in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+7]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted 2-digit group value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:02d}"
                    pointer += 7
                elif count == 1:
                    if pointer + 4 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a single digit in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+4]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted single digit value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:d}"
                    pointer += 4
            elif mode_indicator == 2:  # Alphanumeric mode
                count_indicator_length = 9 if version < 10 else (11 if version < 27 else 13)
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for alphanumeric mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Alphanumeric mode, character count = {count}")
                pointer += count_indicator_length
                alphanum_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
                while count >= 2:
                    if pointer + 11 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a pair in alphanumeric mode")
                        break
                    value = int(''.join(str(bit) for bit in data_bits[pointer:pointer+11]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Alphanumeric mode: Extracted pair value {value} from bits at pointer {pointer}")
                    if self.debug:
                        print(result, " add ", value, " = ", alphanum_table[value // 45], alphanum_table[value % 45])
                    result += alphanum_table[value // 45] + alphanum_table[value % 45]
                    pointer += 11
                    count -= 2
                if count == 1:
                    if pointer + 6 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a single character in alphanumeric mode")
                        break
                    value = int(''.join(str(bit) for bit in data_bits[pointer:pointer+6]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Alphanumeric mode: Extracted single character value {value} from bits at pointer {pointer}")
                    result += alphanum_table[value]
                    pointer += 6
            else:
                if self.debug:
                    print(f"[DEBUG] decode_data: Unsupported mode indicator {mode_indicator:04b}")
                break
        if self.debug:
            print(f"[DEBUG] decode_data: Final decoded message = {result}")
        return result

    def _get_data_capacity(self) -> int:
        """Return the number of data bits (not including error correction) for the QR code."""
        version = self.get_version()
        error_correction, _ = self.extract_format_information()
        
        # Capacity lookup table for versions 1-40
        capacity_lookup = {
            'L': [19, 34, 55, 80, 108, 136, 156, 194, 232, 274, 324, 370, 428, 461, 523, 589, 647, 721, 795, 861, 932, 1006, 1094, 1174, 1276, 1370, 1468, 1531, 1631, 1735, 1843, 1955, 2071, 2191, 2306, 2434, 2566, 2702, 2812, 2956],
            'M': [16, 28, 44, 64, 86, 108, 124, 154, 182, 216, 254, 290, 334, 365, 415, 453, 507, 563, 627, 669, 714, 782, 860, 914, 1000, 1062, 1128, 1193, 1267, 1373, 1455, 1541, 1631, 1725, 1812, 1914, 1992, 2102, 2216, 2334],
            'Q': [13, 22, 34, 48, 62, 76, 88, 110, 132, 154, 180, 206, 244, 261, 295, 325, 367, 397, 445, 485, 512, 568, 614, 664, 718, 754, 808, 871, 911, 985, 1033, 1115, 1171, 1231, 1286, 1354, 1426, 1502, 1582, 1666],
            'H': [9, 16, 26, 36, 46, 60, 66, 86, 100, 122, 140, 158, 180, 197, 223, 253, 283, 313, 341, 385, 406, 442, 464, 514, 538, 596, 628, 661, 701, 745, 793, 845, 901, 961, 986, 1054, 1096, 1142, 1222, 1276]
        }

        # Algorithm to compute data capacity dynamically:
        # 1. Total codewords = (16*version + 128) * version + 64
        # 2. EC codewords per block from lookup or formula
        # 3. Number of blocks from lookup or formula
        # 4. Data codewords = Total codewords - (EC codewords per block * total blocks)
        def compute_capacity(ver: int, ec: str) -> int:
            total_codewords = (16 * ver + 128) * ver + 64
            
            # EC codewords per block (approximate formula)
            ec_per_block = {
                'L': lambda v: int(total_codewords * 0.07),
                'M': lambda v: int(total_codewords * 0.15),
                'Q': lambda v: int(total_codewords * 0.25),
                'H': lambda v: int(total_codewords * 0.30)
            }[ec](ver)
            
            # Number of blocks (approximate formula)
            blocks = max(1, ver // 2)
            if ec in ['Q', 'H']:
                blocks *= 2
            
            return total_codewords - (ec_per_block * blocks)

        return capacity_lookup[error_correction][version - 1] * 8  # Convert to bits

    def _get_block_info(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Get block information for the current QR version and error correction level.
        Returns (data_blocks, ec_blocks) where each block is a list of tuples (count, size)."""
        version = self.get_version()
        error_correction, _ = self.extract_format_information()
        
        # Corrected block_info lookup table for all versions 1-40
        block_info = {
            # Version 1
            (1, 'L'): ([(1, 19)], [(1, 7)]),
            (1, 'M'): ([(1, 16)], [(1, 10)]),
            (1, 'Q'): ([(1, 13)], [(1, 13)]),
            (1, 'H'): ([(1, 9)], [(1, 17)]),

            # Version 2
            (2, 'L'): ([(1, 34)], [(1, 10)]),
            (2, 'M'): ([(1, 28)], [(1, 16)]),
            (2, 'Q'): ([(1, 22)], [(1, 22)]),
            (2, 'H'): ([(1, 16)], [(1, 28)]),

            # Version 3
            (3, 'L'): ([(1, 55)], [(1, 15)]),
            (3, 'M'): ([(1, 44)], [(1, 26)]),
            (3, 'Q'): ([(2, 17)], [(2, 18)]),
            (3, 'H'): ([(2, 13)], [(2, 22)]),

            # Version 4
            (4, 'L'): ([(1, 80)], [(1, 20)]),
            (4, 'M'): ([(2, 32)], [(2, 18)]),
            (4, 'Q'): ([(2, 24)], [(2, 26)]),
            (4, 'H'): ([(4, 9)], [(4, 16)]),

            # Version 5
            (5, 'L'): ([(1, 108)], [(1, 26)]),
            (5, 'M'): ([(2, 43)], [(2, 24)]),
            (5, 'Q'): ([(2, 15), (2, 16)], [(2, 18), (2, 18)]),
            (5, 'H'): ([(2, 11), (2, 12)], [(2, 22), (2, 22)]),

            # Version 6
            (6, 'L'): ([(2, 68)], [(2, 18)]),
            (6, 'M'): ([(4, 27)], [(4, 16)]),
            (6, 'Q'): ([(4, 19)], [(4, 24)]),
            (6, 'H'): ([(4, 15)], [(4, 28)]),

            # Version 7
            (7, 'L'): ([(2, 78)], [(2, 20)]),
            (7, 'M'): ([(4, 31)], [(4, 18)]),
            (7, 'Q'): ([(2, 14), (4, 15)], [(2, 18), (4, 18)]),
            (7, 'H'): ([(4, 13), (1, 14)], [(4, 26), (1, 26)]),

            # Version 8
            (8, 'L'): ([(2, 97)], [(2, 24)]),
            (8, 'M'): ([(2, 38), (2, 39)], [(2, 22), (2, 22)]),
            (8, 'Q'): ([(4, 18), (2, 19)], [(4, 22), (2, 22)]),
            (8, 'H'): ([(4, 14), (2, 15)], [(4, 26), (2, 26)]),

            # Version 9
            (9, 'L'): ([(2, 116)], [(2, 30)]),
            (9, 'M'): ([(3, 36), (2, 37)], [(3, 22), (2, 22)]),
            (9, 'Q'): ([(4, 16), (4, 17)], [(4, 24), (4, 24)]),
            (9, 'H'): ([(4, 12), (4, 13)], [(4, 28), (4, 28)]),

            # Version 10
            (10, 'L'): ([(2, 68), (2, 69)], [(2, 18), (2, 18)]),
            (10, 'M'): ([(4, 43), (1, 44)], [(4, 26), (1, 26)]),
            (10, 'Q'): ([(6, 19), (2, 20)], [(6, 28), (2, 28)]),
            (10, 'H'): ([(6, 15), (2, 16)], [(6, 30), (2, 30)]),

            # Version 11
            (11, 'L'): ([(4, 81)], [(4, 20)]),
            (11, 'M'): ([(1, 50), (4, 51)], [(1, 24), (4, 24)]),
            (11, 'Q'): ([(4, 22), (4, 23)], [(4, 28), (4, 28)]),
            (11, 'H'): ([(3, 12), (8, 13)], [(3, 24), (8, 24)]),

            # Version 12
            (12, 'L'): ([(2, 92), (2, 93)], [(2, 24), (2, 24)]),
            (12, 'M'): ([(6, 36), (2, 37)], [(6, 26), (2, 26)]),
            (12, 'Q'): ([(4, 20), (6, 21)], [(4, 28), (6, 28)]),
            (12, 'H'): ([(7, 14), (4, 15)], [(7, 28), (4, 28)]),

            # Version 13
            (13, 'L'): ([(4, 107)], [(4, 26)]),
            (13, 'M'): ([(8, 37), (1, 38)], [(8, 24), (1, 24)]),
            (13, 'Q'): ([(8, 20), (4, 21)], [(8, 28), (4, 28)]),
            (13, 'H'): ([(12, 11), (4, 12)], [(12, 28), (4, 28)]),

            # Version 14
            (14, 'L'): ([(3, 115), (1, 116)], [(3, 30), (1, 30)]),
            (14, 'M'): ([(4, 40), (5, 41)], [(4, 28), (5, 28)]),
            (14, 'Q'): ([(11, 16), (5, 17)], [(11, 28), (5, 28)]),
            (14, 'H'): ([(11, 12), (5, 13)], [(11, 28), (5, 28)]),

            # Version 15
            (15, 'L'): ([(5, 87), (1, 88)], [(5, 22), (1, 22)]),
            (15, 'M'): ([(5, 41), (5, 42)], [(5, 28), (5, 28)]),
            (15, 'Q'): ([(5, 24), (7, 25)], [(5, 30), (7, 30)]),
            (15, 'H'): ([(11, 12), (7, 13)], [(11, 30), (7, 30)]),

            # Version 16
            (16, 'L'): ([(5, 98), (1, 99)], [(5, 24), (1, 24)]),
            (16, 'M'): ([(7, 45), (3, 46)], [(7, 28), (3, 28)]),
            (16, 'Q'): ([(15, 19), (2, 20)], [(15, 30), (2, 30)]),
            (16, 'H'): ([(3, 15), (13, 16)], [(3, 30), (13, 30)]),

            # Version 17
            (17, 'L'): ([(1, 107), (5, 108)], [(1, 28), (5, 28)]),
            (17, 'M'): ([(10, 46), (1, 47)], [(10, 28), (1, 28)]),
            (17, 'Q'): ([(1, 22), (15, 23)], [(1, 28), (15, 28)]),
            (17, 'H'): ([(2, 14), (17, 15)], [(2, 28), (17, 28)]),

            # Version 18
            (18, 'L'): ([(5, 120), (1, 121)], [(5, 30), (1, 30)]),
            (18, 'M'): ([(9, 43), (4, 44)], [(9, 30), (4, 30)]),
            (18, 'Q'): ([(17, 22), (1, 23)], [(17, 30), (1, 30)]),
            (18, 'H'): ([(2, 14), (19, 15)], [(2, 30), (19, 30)]),

            # Version 19
            (19, 'L'): ([(3, 113), (4, 114)], [(3, 28), (4, 28)]),
            (19, 'M'): ([(3, 44), (11, 45)], [(3, 28), (11, 28)]),
            (19, 'Q'): ([(17, 21), (4, 22)], [(17, 28), (4, 28)]),
            (19, 'H'): ([(9, 13), (16, 14)], [(9, 28), (16, 28)]),

            # Version 20
            (20, 'L'): ([(3, 107), (5, 108)], [(3, 28), (5, 28)]),
            (20, 'M'): ([(3, 41), (13, 42)], [(3, 28), (13, 28)]),
            (20, 'Q'): ([(15, 24), (5, 25)], [(15, 30), (5, 30)]),
            (20, 'H'): ([(15, 15), (10, 16)], [(15, 30), (10, 30)]),

            # Version 21
            (21, 'L'): ([(4, 116), (4, 117)], [(4, 28), (4, 28)]),
            (21, 'M'): ([(17, 42)], [(17, 28)]),
            (21, 'Q'): ([(17, 22), (6, 23)], [(17, 30), (6, 30)]),
            (21, 'H'): ([(19, 16), (6, 17)], [(19, 30), (6, 30)]),

            # Version 22
            (22, 'L'): ([(2, 111), (7, 112)], [(2, 28), (7, 28)]),
            (22, 'M'): ([(17, 46)], [(17, 28)]),
            (22, 'Q'): ([(7, 24), (16, 25)], [(7, 30), (16, 30)]),
            (22, 'H'): ([(34, 13)], [(34, 30)]),

            # Version 23
            (23, 'L'): ([(4, 121), (5, 122)], [(4, 30), (5, 30)]),
            (23, 'M'): ([(4, 47), (14, 48)], [(4, 30), (14, 30)]),
            (23, 'Q'): ([(11, 24), (14, 25)], [(11, 30), (14, 30)]),
            (23, 'H'): ([(16, 15), (14, 16)], [(16, 30), (14, 30)]),

            # Version 24
            (24, 'L'): ([(6, 117), (4, 118)], [(6, 30), (4, 30)]),
            (24, 'M'): ([(6, 45), (14, 46)], [(6, 30), (14, 30)]),
            (24, 'Q'): ([(11, 24), (16, 25)], [(11, 30), (16, 30)]),
            (24, 'H'): ([(30, 16), (2, 17)], [(30, 30), (2, 30)]),

            # Version 25
            (25, 'L'): ([(8, 106), (4, 107)], [(8, 26), (4, 26)]),
            (25, 'M'): ([(8, 47), (13, 48)], [(8, 28), (13, 28)]),
            (25, 'Q'): ([(7, 24), (22, 25)], [(7, 30), (22, 30)]),
            (25, 'H'): ([(22, 15), (13, 16)], [(22, 30), (13, 30)]),

            # Version 26
            (26, 'L'): ([(10, 114), (2, 115)], [(10, 28), (2, 28)]),
            (26, 'M'): ([(19, 46), (4, 47)], [(19, 28), (4, 28)]),
            (26, 'Q'): ([(28, 22), (6, 23)], [(28, 28), (6, 28)]),
            (26, 'H'): ([(33, 16), (4, 17)], [(33, 30), (4, 30)]),

            # Version 27
            (27, 'L'): ([(8, 122), (4, 123)], [(8, 30), (4, 30)]),
            (27, 'M'): ([(22, 45), (3, 46)], [(22, 30), (3, 30)]),
            (27, 'Q'): ([(8, 23), (26, 24)], [(8, 30), (26, 30)]),
            (27, 'H'): ([(12, 15), (28, 16)], [(12, 30), (28, 30)]),

            # Version 28
            (28, 'L'): ([(3, 117), (10, 118)], [(3, 30), (10, 30)]),
            (28, 'M'): ([(3, 45), (23, 46)], [(3, 30), (23, 30)]),
            (28, 'Q'): ([(4, 24), (31, 25)], [(4, 30), (31, 30)]),
            (28, 'H'): ([(11, 15), (31, 16)], [(11, 30), (31, 30)]),

            # Version 29
            (29, 'L'): ([(7, 116), (7, 117)], [(7, 30), (7, 30)]),
            (29, 'M'): ([(21, 45), (7, 46)], [(21, 30), (7, 30)]),
            (29, 'Q'): ([(1, 23), (37, 24)], [(1, 30), (37, 30)]),
            (29, 'H'): ([(19, 15), (26, 16)], [(19, 30), (26, 30)]),

            # Version 30
            (30, 'L'): ([(5, 115), (10, 116)], [(5, 30), (10, 30)]),
            (30, 'M'): ([(19, 47), (10, 48)], [(19, 30), (10, 30)]),
            (30, 'Q'): ([(15, 24), (25, 25)], [(15, 30), (25, 30)]),
            (30, 'H'): ([(23, 15), (25, 16)], [(23, 30), (25, 30)]),

            # Version 31
            (31, 'L'): ([(13, 115), (3, 116)], [(13, 30), (3, 30)]),
            (31, 'M'): ([(2, 46), (29, 47)], [(2, 30), (29, 30)]),
            (31, 'Q'): ([(42, 24), (1, 25)], [(42, 30), (1, 30)]),
            (31, 'H'): ([(23, 15), (28, 16)], [(23, 30), (28, 30)]),

            # Version 32
            (32, 'L'): ([(17, 115)], [(17, 30)]),
            (32, 'M'): ([(10, 46), (23, 47)], [(10, 30), (23, 30)]),
            (32, 'Q'): ([(10, 24), (35, 25)], [(10, 30), (35, 30)]),
            (32, 'H'): ([(19, 15), (35, 16)], [(19, 30), (35, 30)]),

            # Version 33
            (33, 'L'): ([(17, 115), (1, 116)], [(17, 30), (1, 30)]),
            (33, 'M'): ([(14, 46), (21, 47)], [(14, 30), (21, 30)]),
            (33, 'Q'): ([(29, 24), (19, 25)], [(29, 30), (19, 30)]),
            (33, 'H'): ([(11, 15), (46, 16)], [(11, 30), (46, 30)]),

            # Version 34
            (34, 'L'): ([(13, 115), (6, 116)], [(13, 30), (6, 30)]),
            (34, 'M'): ([(14, 46), (23, 47)], [(14, 30), (23, 30)]),
            (34, 'Q'): ([(44, 24), (7, 25)], [(44, 30), (7, 30)]),
            (34, 'H'): ([(59, 16), (1, 17)], [(59, 30), (1, 30)]),

            # Version 35
            (35, 'L'): ([(12, 121), (7, 122)], [(12, 30), (7, 30)]),
            (35, 'M'): ([(12, 47), (26, 48)], [(12, 30), (26, 30)]),
            (35, 'Q'): ([(39, 24), (14, 25)], [(39, 30), (14, 30)]),
            (35, 'H'): ([(22, 15), (41, 16)], [(22, 30), (41, 30)]),

            # Version 36
            (36, 'L'): ([(6, 121), (14, 122)], [(6, 30), (14, 30)]),
            (36, 'M'): ([(6, 47), (34, 48)], [(6, 30), (34, 30)]),
            (36, 'Q'): ([(46, 24), (10, 25)], [(46, 30), (10, 30)]),
            (36, 'H'): ([(2, 15), (64, 16)], [(2, 30), (64, 30)]),

            # Version 37
            (37, 'L'): ([(17, 122), (4, 123)], [(17, 30), (4, 30)]),
            (37, 'M'): ([(29, 46), (14, 47)], [(29, 30), (14, 30)]),
            (37, 'Q'): ([(49, 24), (10, 25)], [(49, 30), (10, 30)]),
            (37, 'H'): ([(24, 15), (46, 16)], [(24, 30), (46, 30)]),

            # Version 38
            (38, 'L'): ([(4, 122), (18, 123)], [(4, 30), (18, 30)]),
            (38, 'M'): ([(13, 46), (32, 47)], [(13, 30), (32, 30)]),
            (38, 'Q'): ([(48, 24), (14, 25)], [(48, 30), (14, 30)]),
            (38, 'H'): ([(42, 15), (32, 16)], [(42, 30), (32, 30)]),

            # Version 39
            (39, 'L'): ([(20, 117), (4, 118)], [(20, 30), (4, 30)]),
            (39, 'M'): ([(40, 47), (7, 48)], [(40, 30), (7, 30)]),
            (39, 'Q'): ([(43, 24), (22, 25)], [(43, 30), (22, 30)]),
            (39, 'H'): ([(10, 15), (67, 16)], [(10, 30), (67, 30)]),

            # Version 40
            (40, 'L'): ([(19, 118), (6, 119)], [(19, 30), (6, 30)]),
            (40, 'M'): ([(18, 47), (31, 48)], [(18, 30), (31, 30)]),
            (40, 'Q'): ([(34, 24), (34, 25)], [(34, 30), (34, 30)]),
            (40, 'H'): ([(20, 15), (61, 16)], [(20, 30), (61, 30)]),
        }

        # Algorithm to compute block information dynamically for versions > 40:
        def compute_block_info(ver: int, ec: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
            total_codewords = (16 * ver + 128) * ver + 64
            
            # EC codewords per block (approximate formula)
            ec_per_block = {
                'L': lambda v: int(total_codewords * 0.07),
                'M': lambda v: int(total_codewords * 0.15),
                'Q': lambda v: int(total_codewords * 0.25),
                'H': lambda v: int(total_codewords * 0.30)
            }[ec](ver)
            
            # Number of blocks (approximate formula)
            blocks = max(1, ver // 2)
            if ec in ['Q', 'H']:
                blocks *= 2
            
            data_per_block = (total_codewords - (ec_per_block * blocks)) // blocks
            
            # For simplicity in theoretical versions > 40, we use single block size
            return ([(blocks, data_per_block)], [(blocks, ec_per_block)])

        key = (version, error_correction)
        if key in block_info:
            return block_info[key]
        
        # For theoretical versions > 40, use computed values
        return compute_block_info(version, error_correction)

    def _extract_blocks(self, data_bits: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """Extract data and error correction blocks from the raw data bits."""
        data_blocks_info, ec_blocks_info = self._get_block_info()
        
        if self.debug:
            print(f"[DEBUG] _extract_blocks: Data blocks info: {data_blocks_info}")
            print(f"[DEBUG] _extract_blocks: EC blocks info: {ec_blocks_info}")
        
        # Convert raw codeword bits to bytes.
        raw_codewords = []
        for i in range(0, len(data_bits), 8):
            if i + 8 <= len(data_bits):
                byte = int(''.join(str(bit) for bit in data_bits[i:i+8]), 2)
            raw_codewords.append(byte)
        
        if self.debug:
            print(f"[DEBUG] _extract_blocks: Total codewords: {len(raw_codewords)}")

        total_blocks = sum(count for count, _ in data_blocks_info)

        data_sizes = []
        for count, size in data_blocks_info:
            data_sizes.extend([size] * count)

        ec_sizes = []
        for count, size in ec_blocks_info:
            ec_sizes.extend([size] * count)

        if len(data_sizes) != total_blocks or len(ec_sizes) != total_blocks:
            # Fallback for malformed metadata.
            if self.debug:
                print("[DEBUG] _extract_blocks: block metadata mismatch, fallback to single block")
            data_sizes = [len(raw_codewords)]
            ec_sizes = [0]
            total_blocks = 1

        data_blocks = [[] for _ in range(total_blocks)]
        ec_blocks = [[] for _ in range(total_blocks)]

        # De-interleave data codewords.
        pos = 0
        max_data_size = max(data_sizes) if data_sizes else 0
        for i in range(max_data_size):
            for j in range(total_blocks):
                if i < data_sizes[j] and pos < len(raw_codewords):
                    data_blocks[j].append(raw_codewords[pos])
                    pos += 1

        # De-interleave EC codewords.
        max_ec_size = max(ec_sizes) if ec_sizes else 0
        for i in range(max_ec_size):
            for j in range(total_blocks):
                if i < ec_sizes[j] and pos < len(raw_codewords):
                    ec_blocks[j].append(raw_codewords[pos])
                    pos += 1

        # Pad missing EC bytes if stream truncated.
        for i in range(total_blocks):
            missing = ec_sizes[i] - len(ec_blocks[i])
            if missing > 0:
                ec_blocks[i].extend([0] * missing)

        if self.debug:
            for i, (data, ec) in enumerate(zip(data_blocks, ec_blocks)):
                print(f"[DEBUG] _extract_blocks: Block {i}: {len(data)} data bytes, {len(ec)} EC bytes")
        
        return data_blocks, ec_blocks

    def _gf_mult(self, x: int, y: int, primitive: int = 0x11D) -> int:
        """Multiply two numbers in the Galois Field."""
        result = 0
        while y:
            if y & 1:
                result ^= x
            y >>= 1
            x <<= 1
            if x & 0x100:
                x ^= primitive
        return result

    def _gf_pow(self, x: int, power: int) -> int:
        """Raise x to the power in the Galois Field."""
        result = 1
        while power:
            if power & 1:
                result = self._gf_mult(result, x)
            x = self._gf_mult(x, x)
            power >>= 1
        return result

    def _gf_inverse(self, x: int) -> int:
        """Find the multiplicative inverse of x in the Galois Field."""
        return self._gf_pow(x, 255)  # In GF(2^8), x^255 = x^-1

    def _gf_poly_scale(self, p: list[int], x: int) -> list[int]:
        """Multiply polynomial by a scalar."""
        return [self._gf_mult(coef, x) for coef in p]

    def _gf_poly_add(self, p: list[int], q: list[int]) -> list[int]:
        """Add two polynomials in the Galois Field."""
        # Pad shorter polynomial with zeros
        if len(p) > len(q):
            q = q + [0] * (len(p) - len(q))
        else:
            p = p + [0] * (len(q) - len(p))
        # Add coefficients
        return [p[i] ^ q[i] for i in range(len(p))]

    def _gf_poly_mult(self, p: list[int], q: list[int]) -> list[int]:
        """Multiply two polynomials in the Galois Field."""
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self._gf_mult(p[i], q[j])
        return r

    def _gf_poly_eval(self, poly: list[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        y = poly[0]
        for i in range(1, len(poly)):
            y = self._gf_mult(y, x) ^ poly[i]
        return y

    def _rs_generator_poly(self, nsym: int) -> list[int]:
        """Generate Reed-Solomon generator polynomial."""
        g = [1]
        for i in range(nsym):
            g = self._gf_poly_mult(g, [1, self._gf_pow(2, i)])
        return g

    def _rs_decode(self, msg_in: list[int], nsym: int) -> list[int]:
        """Decode a Reed-Solomon encoded message."""
        if len(msg_in) <= nsym:
            return msg_in  # Not enough data to decode
            
        # Find error locations
        syndromes = [0] * nsym
        for i in range(nsym):
            syndromes[i] = self._gf_poly_eval(msg_in, self._gf_pow(2, i))
            
        if max(syndromes) == 0:
            return msg_in[:-nsym]  # No errors
            
        # Find error locator polynomial using Berlekamp-Massey
        err_loc = [1]
        old_loc = [1]
        
        for i in range(nsym):
            delta = syndromes[i]
            for j in range(1, len(err_loc)):
                delta ^= self._gf_mult(err_loc[-(j+1)], syndromes[i-j])
                
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self._gf_poly_scale(old_loc, delta)
                    old_loc = self._gf_poly_scale(err_loc, self._gf_inverse(delta))
                    err_loc = new_loc
                err_loc = self._gf_poly_add(err_loc, self._gf_poly_scale(old_loc, delta))
                
        # Find zeros of error polynomial
        err_pos = []
        for i in range(len(msg_in)):
            if self._gf_poly_eval(err_loc, self._gf_pow(2, 255-i)) == 0:
                err_pos.append(len(msg_in) - 1 - i)
                
        if len(err_pos) != len(err_loc) - 1:
            return msg_in[:-nsym]  # Too many errors to correct
            
        # Find error values and correct using Forney algorithm.
        msg_out = list(msg_in)

        # Error evaluator polynomial omega(x) = (S(x) * sigma(x)) mod x^nsym.
        eval_poly = self._gf_poly_mult(syndromes, err_loc)
        eval_poly = eval_poly[-nsym:] if len(eval_poly) > nsym else eval_poly

        # Formal derivative sigma'(x) in characteristic 2 (keep odd powers only).
        err_loc_prime = []
        degree = len(err_loc) - 1
        for idx, coef in enumerate(err_loc):
            power = degree - idx
            if power % 2 == 1:
                err_loc_prime.append(coef)

        if not err_loc_prime:
            err_loc_prime = [1]

        for i in range(len(err_pos)):
            pos = err_pos[i]

            xi_inv = self._gf_pow(2, 255-pos)
            numerator = self._gf_poly_eval(eval_poly, xi_inv)
            denominator = self._gf_poly_eval(err_loc_prime, xi_inv)
            if denominator == 0:
                continue
            err_val = self._gf_mult(numerator, self._gf_inverse(denominator))

            msg_out[pos] ^= err_val
            
        return msg_out[:-nsym]  # Remove error correction bytes

    def _apply_error_correction(self, data_blocks: list[list[int]], ec_blocks: list[list[int]]) -> list[int]:
        """Apply error correction to the data blocks using the error correction blocks."""
        if self.debug:
            print(f"[DEBUG] _apply_error_correction: Processing {len(data_blocks)} data blocks")
        
        corrected_data = []
        for i, (data, ec) in enumerate(zip(data_blocks, ec_blocks)):
            if self.debug:
                print(f"[DEBUG] _apply_error_correction: Block {i+1}: {len(data)} data bytes, {len(ec)} EC bytes")
            
            # Combine data and EC bytes for Reed-Solomon decoding
            block = data + ec
            if self.debug:
                print(f"[DEBUG] _apply_error_correction: Combined block length: {len(block)}")
            
            # Apply Reed-Solomon error correction
            try:
                corrected = self._rs_decode(block, len(ec))
                if self.debug:
                    print(f"[DEBUG] _apply_error_correction: Corrected block length: {len(corrected)}")
                corrected_data.extend(corrected)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] _apply_error_correction: Error in block {i+1}: {str(e)}")
                # If error correction fails, use original data
                corrected_data.extend(data)
        
        return corrected_data

    def decode(self) -> str:
        """Decode the QR code and return the message."""
        # 1. Get version
        version = self.get_version()
        if self.debug:
            print(f"[DEBUG] decode: QR code version = {version}")

        # 2. Extract format information
        error_correction, mask_pattern = self.extract_format_information()
        if self.debug:
            print(f"[DEBUG] decode: Error correction level = {error_correction}, Mask pattern = {mask_pattern}")

        # 3. Extract version information (for version >= 7)
        if version >= 7:
            version_info = self.extract_version_information()
            if self.debug:
                print(f"[DEBUG] decode: Version information = {version_info}")

        # 4. Unmask data
        self.unmask_data(mask_pattern)
        if self.debug:
            print("[DEBUG] decode: Data unmasked")

        # 5. Extract data bits
        data_bits = self.extract_data_bits()
        if self.debug:
            print(f"[DEBUG] decode: Extracted {len(data_bits)} data bits")

        # 6. Extract data and error correction blocks
        data_blocks, ec_blocks = self._extract_blocks(data_bits)
        if self.debug:
            print(f"[DEBUG] decode: Extracted {len(data_blocks)} data blocks and {len(ec_blocks)} EC blocks")

        # 7. Apply error correction
        corrected_data = self._apply_error_correction(data_blocks, ec_blocks)
        if self.debug:
            print(f"[DEBUG] decode: Applied error correction, got {len(corrected_data)} bytes")

        # 8. Convert corrected data back to bits for decoding
        corrected_bits = []
        for byte in corrected_data:
            bits = [(byte >> i) & 1 for i in range(7, -1, -1)]
            corrected_bits.extend(bits)

        # 9. Decode the corrected data bits
        return self.decode_data(corrected_bits)


if __name__ == "__main__":
    qr_code = QRCode('https://cs.unibuc.ro/~crusu/asc/index.html', version=-1, error_correction='L', mask=-1, debug=False, mode="auto")
    data_bits = qr_code.get_matrix()
    version = qr_code.get_version()
    error_correction = qr_code.get_error_correction()
    mask_pattern = qr_code.get_mask()
    
    print("Generated QR Code:")
    for row in data_bits:
        print("".join("██" if bit == 1 else "  " for bit in row))
    print(f"Generated version: {version}")
    print(f"Error correction: {error_correction}")
    print(f"Mask pattern: {mask_pattern}")


    qr_decode = QRDecode(data_bits, debug=True)
    decoded_text = qr_decode.decode()
    print(f"\nDecoded text: {decoded_text}")