import os
import csv


def get_image_paths(csv_path):
    """Read image list from CSV."""
    base_dir = os.path.dirname(csv_path)

    if base_dir == "":
        base_dir = "."

    if not os.path.exists(csv_path):
        print(f"khong tim thay file {csv_path}.")
        return []

    data_list = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            img_id = row["image_id"].strip()
            img_path_relative = row["image_path"].strip()
            img_path = os.path.join(base_dir, img_path_relative)

            data_list.append({
                "image_id": img_id,
                "path": img_path,
            })

    return data_list


def write_output_csv(results, output_filename="output.csv"):
    headers = [
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
    ]

    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        writer.writerow(headers)

        for res in results:
            img_id = res["image_id"]
            qrs = res["qrs"]

            if len(qrs) == 0:
                writer.writerow([img_id, "", "", "", "", "", "", "", "", "", ""])
            else:
                for idx, qr in enumerate(qrs):
                    writer.writerow([
                        img_id,
                        idx,
                        int(round(qr["x0"])),
                        int(round(qr["y0"])),
                        int(round(qr["x1"])),
                        int(round(qr["y1"])),
                        int(round(qr["x2"])),
                        int(round(qr["y2"])),
                        int(round(qr["x3"])),
                        int(round(qr["y3"])),
                        qr.get("content", ""),
                    ])

    print(f"Ket qua da duoc ghi vao {output_filename}.")


def is_valid_qr_triangle(p1, p2, p3, tolerance=0.3):
    """Check whether 3 points form a right isosceles triangle."""
    d12 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    d23 = (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2
    d13 = (p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2

    d = sorted([d12, d23, d13])

    if d[0] == 0:
        return False

    ratio_sides = d[1] / d[0]
    if ratio_sides > (1.0 + tolerance):
        return False

    ratio_pythagoras = (d[0] + d[1]) / d[2]
    if not ((1.0 - tolerance) < ratio_pythagoras < (1.0 + tolerance)):
        return False

    return True
