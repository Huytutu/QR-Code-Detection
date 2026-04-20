# QR Code Detection & Decoding Submission

## Cấu trúc tệp

```
submission/
├── main.py              # Chương trình chính (bắt buộc)
├── pipeline.py          # QR detection pipeline
├── decoder.py           # QR decoder (from scratch, not using pyzbar)
├── decoder_wrapper.py   # Wrapper để gọi decoder
├── utils.py             # Utility functions (I/O)
├── requirements.txt     # Dependencies
├── pca_model.npz        # Pre-trained PCA model
├── evaluate.py          # Validate output.csv format
├── evaluate_iou.py      # Evaluate with IoU metrics (if ground truth available)
└── output.csv           # Output file (generated)
```

## Cài đặt

```bash
# Lần đầu, cài thư viện
pip install -r requirements.txt
```

## Chạy chương trình

### Cách 1: Chạy detection + decoding
```bash
python main.py --data <path_to_csv>
```

Ví dụ:
```bash
python main.py --data ../qr/public_valid.csv --decode=yes
```

### Cách 2: Chạy detection mà không decode
```bash
python main.py --data <path_to_csv> --decode=no
```

### Cách 3: Chạy trên tập private (khi có sẵn)
```bash
python main.py --data private.csv
```

## Định dạng Input (CSV)

File CSV phải có cấu trúc:
```csv
image_id,image_path
img_001,train/img_001.jpg
img_002,train/img_002.jpg
```

## Định dạng Output (output.csv)

```csv
image_id,qr_index,x0,y0,x1,y1,x2,y2,x3,y3,content
2656508531_jpg.rf,0,75,82,161,82,161,148,75,148,https://example.com
2879826877_jpg.rf,0,103,309,183,309,183,412,103,412,
2879826877_jpg.rf,1,315,419,403,419,403,529,315,529,Hello World
IMG_20220601_jpg.rf,,,,,,,,,,
```

**Quy ước tọa độ:**
```
(x0,y0) -------- (x1,y1)
   |                  |
   |      QR code     |
   |                  |
(x3,y3) -------- (x2,y2)
```

## Kiểm tra kết quả

### 1. Validate format
```bash
python evaluate.py
```

Kiểm tra:
- ✓ UTF-8 encoding
- ✓ CSV header đúng
- ✓ qr_index ordering
- ✓ Coordinates format
- ✓ Không có unnecessary quotes

### 2. Evaluate với IoU metrics (optional)
```bash
python evaluate_iou.py
```

Nếu có `ground_truth.csv`, sẽ compute:
- Precision, Recall, F1 Score
- TP, FP, FN counts
- IoU matching với threshold 0.5

## Tham số chương trình

### --data (bắt buộc)
Đường dẫn tới file CSV đầu vào

### --decode (tùy chọn, mặc định: yes)
- `--decode=yes`: Decode QR content
- `--decode=no`: Chỉ detect vị trí (nhanh hơn)

## Yêu cầu phần cứng

- **CPU**: Không cần GPU, chạy trên CPU
- **RAM**: ~1-2GB
- **Tốc độ**: ~0.05 giây/ảnh (detection), ~0.1 giây/ảnh (with decoding)

## Thư viện sử dụng

```
numpy          - Numerical operations
opencv-python  - Image processing
Pillow         - Image I/O
shapely        - Geometric operations (IoU calculation)
```

**Lưu ý**: Không sử dụng pyzbar hoặc bất kỳ thư viện sẵn có nào cho detection/decoding

## Troubleshooting

### Lỗi: "output.csv không được ghi"
- Đảm bảo folder submission có quyền ghi
- Kiểm tra file encoding là UTF-8

### Lỗi: "Ảnh không tìm thấy"
- Kiểm tra đường dẫn trong CSV là đúng
- Đường dẫn nên tính từ vị trí file CSV

### Lỗi: "decoder fail"
- Bình thường khi detected region không phải QR code hợp lệ
- Chương trình tự động trả về empty content, không fail

## Liên hệ

- Kiểm tra log output để debug
- Dùng `--decode=no` để test detection nhanh
- Dùng `evaluate.py` để validate format trước khi submit
