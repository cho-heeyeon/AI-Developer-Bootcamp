import cv2
import numpy as np
from pathlib import Path


base_dir = Path("Day020-1")


samples = {
    "03": "sample_03_roi.jpg",
    "04": "sample_04_roi.jpg",
    "06": "sample_06_roi.jpg",
    "07": "sample_07_roi.jpg",
    "N01": "sample_N01_roi.jpg",
    "N02": "sample_N02_roi.jpg",
}


def subpixel_peak(profile, index):

    if index <= 0 or index >= len(profile) - 1:
        return float(index)

    y1 = profile[index - 1]
    y2 = profile[index]
    y3 = profile[index + 1]

    denominator = y1 - 2 * y2 + y3

    if denominator == 0:
        return float(index)

    offset = 0.5 * (y1 - y3) / denominator

    return index + offset


def measure_subpixel(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다 : {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    gradient_y = cv2.Sobel(
        blur,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_abs = np.abs(
        gradient_y
    )

    height, width = gray.shape

    center_y = height // 2

    x_start = int(width * 0.25)
    x_end = int(width * 0.75)

    x_positions = np.linspace(
        x_start,
        x_end,
        7,
        dtype=int
    )

    diameters = []


    for x in x_positions:

        x1 = max(0, x - 2)
        x2 = min(width, x + 3)

        strip = gradient_abs[:, x1:x2]

        profile = np.mean(
            strip,
            axis=1
        )


        # 상단 Sub-pixel Edge
        top_profile = profile[:center_y]

        top_index = np.argmax(
            top_profile
        )

        y_top = subpixel_peak(
            top_profile,
            top_index
        )


        # 하단 Sub-pixel Edge
        bottom_profile = profile[center_y:]

        bottom_index_local = np.argmax(
            bottom_profile
        )

        bottom_index = (
            center_y
            + bottom_index_local
        )

        y_bottom = subpixel_peak(
            profile,
            bottom_index
        )


        diameter = (
            y_bottom
            - y_top
        )

        diameters.append(
            diameter
        )


    median_pixel = np.median(
        diameters
    )

    mean_pixel = np.mean(
        diameters
    )

    std_pixel = np.std(
        diameters
    )

    return (
        median_pixel,
        mean_pixel,
        std_pixel
    )


print()
print("========== Sub-pixel 측정 결과 ==========")
print()


results = {}


for name, filename in samples.items():

    image_path = base_dir / filename

    median_px, mean_px, std_px = measure_subpixel(
        image_path
    )

    results[name] = median_px

    print(
        f"{name:>3} | "
        f"중앙값 : {median_px:.6f} px | "
        f"평균 : {mean_px:.6f} px | "
        f"표준편차 : {std_px:.6f} px"
    )


print()
print("========== Calibration용 ==========")

for name in ["03", "04", "06", "N01", "N02"]:

    print(
        name,
        ":",
        f"{results[name]:.6f}",
        "pixel"
    )


print()
print("========== 검증용 ==========")

print(
    "07 :",
    f"{results['07']:.6f}",
    "pixel"
)